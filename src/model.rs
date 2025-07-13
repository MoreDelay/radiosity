use std::path::Path;

use cgmath::{EuclideanSpace, InnerSpace, Zero};
use image::ImageReader;
use parser::{ObjVertexTriplet, ParsedObj};

use crate::{
    primitives::{Color, Vertex},
    render::layout::{
        GpuTransfer, GpuTransferRef, InstanceBufferRaw, InstanceRaw, PhongRaw, TextureRaw,
        TriangleBufferRaw,
    },
};

pub mod parser;

pub const NUM_INSTANCES_PER_ROW: u32 = 10;

#[derive(Copy, Clone, Debug)]
pub struct Instance {
    pub position: cgmath::Vector3<f32>,
    pub rotation: cgmath::Quaternion<f32>,
}

#[derive(Copy, Clone, Debug)]
pub struct Triplet(u32, u32, u32);

#[derive(Clone, Debug)]
pub struct ColorTexture(pub image::ImageBuffer<image::Rgba<u8>, Vec<u8>>);

#[derive(Clone, Debug)]
pub struct NormalTexture(pub image::ImageBuffer<image::Rgba<u8>, Vec<u8>>);

pub struct Model {
    pub mesh: Mesh,
    pub material: Option<Material>,
}

#[derive(Clone, Debug)]
pub struct PhongParameters {
    pub ambient_color: Color,
    pub diffuse_color: Color,
    pub specular_color: Color,
    pub specular_exponent: f32,
}

pub struct Material {
    #[expect(unused)]
    pub name: String,
    pub phong_params: Option<PhongParameters>,
    pub color_texture: Option<ColorTexture>,
    pub normal_texture: Option<NormalTexture>,
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub triangles: Vec<Triplet>,
    pub material_id: Option<usize>,
}

impl Material {
    pub fn load(root: &Path, mtl: &parser::ParsedMtl) -> anyhow::Result<Self> {
        let color_texture = match &mtl.map_kd {
            None => None,
            Some(parser::MtlMapKd(path)) => {
                let path = root.join(path);
                let image = ImageReader::open(path)?.decode()?;
                Some(ColorTexture(image.into()))
            }
        };
        let normal_texture = match &mtl.map_bump {
            None => None,
            Some(parser::MtlMapBump(path)) => {
                let path = root.join(path);
                let image = ImageReader::open(path)?.decode()?;
                Some(NormalTexture(image.into()))
            }
        };

        let white = Color {
            r: 255,
            g: 255,
            b: 255,
        };

        let ambient_color = mtl.ka.map(|c| c.into()).unwrap_or(white);
        let diffuse_color = mtl.kd.map(|c| c.into()).unwrap_or(white);
        let specular_color = mtl.ks.map(|c| c.into()).unwrap_or(white);
        let specular_exponent = mtl
            .ns
            .map(|parser::MtlNs(exponent)| exponent)
            .unwrap_or(10.);

        let phong_params = Some(PhongParameters {
            ambient_color,
            diffuse_color,
            specular_color,
            specular_exponent,
        });

        Ok(Self {
            name: mtl.name.to_string(),
            phong_params,
            color_texture,
            normal_texture,
        })
    }
}

impl Model {
    pub fn load(file_name: &Path) -> anyhow::Result<Self> {
        let (parsed_obj, materials) = parser::parse_obj(file_name)?;
        let mesh = Mesh::new(parsed_obj);
        let root = file_name.parent().expect("texture should not be root");
        let material = match mesh.material_id {
            Some(index) => Some(Material::load(root, &materials[index])?),
            _ => None,
        };
        Ok(Self { mesh, material })
    }
}

impl Mesh {
    fn new(parsed_obj: ParsedObj) -> Self {
        let ParsedObj {
            vertices,
            texture_coords,
            normals,
            faces,
            material_switches,
        } = parsed_obj;

        assert!(vertices.len() == normals.len());

        let mut vertices = vertices
            .into_iter()
            .map(|parser::ObjV { x, y, z, .. }| {
                let position = cgmath::Point3 { x, y, z };
                let tex_coords = cgmath::Point2::origin();
                let zero = cgmath::Vector3::zero();

                Vertex {
                    position,
                    tex_coords,
                    normal: zero,
                    tangent: zero,
                    bitangent: zero,
                }
            })
            .collect::<Vec<_>>();

        for face in faces.iter() {
            for triplet in face.triplets.iter() {
                let i_vertex = triplet.index_vertex;
                let Some(i_normal) = triplet.index_normal else {
                    continue;
                };
                let parser::ObjVn { i, j, k } = normals[i_normal];
                let normal = cgmath::Vector3 { x: i, y: j, z: k }.normalize();
                let unit = if (normal - cgmath::Vector3::unit_x()).magnitude() > 0.1 {
                    cgmath::Vector3::unit_x()
                } else {
                    cgmath::Vector3::unit_y()
                };
                let tangent = normal.cross(unit).normalize();
                let bitangent = normal.cross(tangent);

                vertices[i_vertex].normal = normal;
                vertices[i_vertex].tangent = tangent;
                vertices[i_vertex].bitangent = bitangent;
            }
        }
        for v in vertices.iter_mut() {
            v.normal = v.normal.normalize();
            v.tangent = v.tangent.normalize();
            v.bitangent = v.bitangent.normalize();
        }

        // set texture coordinates and normals as specified by faces
        for face in faces.iter() {
            for triplet in face.triplets.iter() {
                let ObjVertexTriplet {
                    index_vertex,
                    index_texture,
                    index_normal,
                } = triplet;

                if let Some(index_texture) = index_texture {
                    let tex_coords = texture_coords[*index_texture];
                    let parser::ObjVt { u, v, w: _ } = tex_coords;
                    let vertex_tex = &mut vertices[*index_vertex].tex_coords;
                    *vertex_tex = cgmath::Point2 { x: u, y: v };
                }

                if let Some(index_normal) = index_normal {
                    let normal = normals[*index_normal];
                    let parser::ObjVn { i, j, k } = normal;
                    let vertex_normal = &mut vertices[*index_vertex].normal;
                    *vertex_normal = cgmath::Vector3 { x: i, y: j, z: k };
                }
            }
        }

        // make all faces to triangles naively
        let triangles = faces
            .into_iter()
            .flat_map(|face| {
                assert!(face.triplets.len() >= 3, "too few vertices, not a face!");
                let ([first], other) = face.triplets.split_at(1) else {
                    unreachable!("checked above");
                };
                other
                    .windows(2)
                    .flat_map(<&[ObjVertexTriplet; 2]>::try_from)
                    .map(|[second, third]| {
                        Triplet(
                            first.index_vertex as u32,
                            second.index_vertex as u32,
                            third.index_vertex as u32,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // TODO: currently switching materials not supported as we drop all switches
        let material_id = material_switches.first().map(|v| v.material_index);

        Self {
            vertices,
            triangles,
            material_id,
        }
    }
}

impl GpuTransfer for Mesh {
    type Raw = TriangleBufferRaw;

    fn to_raw(&self) -> Self::Raw {
        assert!(self.vertices.len() <= <u32>::MAX as usize);
        let upper_bound = self.vertices.len() as u32;
        let indices = self
            .triangles
            .iter()
            .flat_map(|&Triplet(i, j, k)| {
                assert!(i < upper_bound);
                assert!(j < upper_bound);
                assert!(k < upper_bound);
                [i, j, k]
            })
            .collect();
        let vertices = self.vertices.iter().map(|v| v.to_raw()).collect();
        TriangleBufferRaw { vertices, indices }
    }
}

impl GpuTransfer for Vec<Instance> {
    type Raw = InstanceBufferRaw;

    fn to_raw(&self) -> Self::Raw {
        let instances = self.iter().map(|i| i.to_raw()).collect();
        InstanceBufferRaw { instances }
    }
}

impl<'a> GpuTransferRef<'a> for ColorTexture {
    type Raw = TextureRaw<'a>;

    fn to_raw(&'a self) -> TextureRaw<'a> {
        let Self(image) = self;
        let data = &image;
        let dimensions = image.dimensions();
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        TextureRaw { data, format, size }
    }
}

impl<'a> GpuTransferRef<'a> for NormalTexture {
    type Raw = TextureRaw<'a>;

    fn to_raw(&'a self) -> TextureRaw<'a> {
        let Self(image) = self;
        let data = &image;
        let dimensions = image.dimensions();
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        TextureRaw { data, format, size }
    }
}

impl GpuTransfer for Instance {
    type Raw = InstanceRaw;
    fn to_raw(&self) -> Self::Raw {
        let model = (cgmath::Matrix4::from_translation(self.position)
            * cgmath::Matrix4::from(self.rotation))
        .into();
        let normal = cgmath::Matrix3::from(self.rotation).into();
        InstanceRaw { model, normal }
    }
}

impl From<Triplet> for (u32, u32, u32) {
    fn from(Triplet(i, j, k): Triplet) -> Self {
        (i, j, k)
    }
}

impl GpuTransfer for PhongParameters {
    type Raw = PhongRaw;

    fn to_raw(&self) -> Self::Raw {
        PhongRaw {
            specular_color: self.specular_color.into(),
            specular_exponent: self.specular_exponent,
            diffuse_color: self.diffuse_color.into(),
            _padding1: 0,
            ambient_color: self.ambient_color.into(),
            _padding2: 0,
        }
    }
}
