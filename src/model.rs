use std::path::Path;

use cgmath::{EuclideanSpace, InnerSpace, Zero};
use image::ImageReader;
use parser::{ObjVertexTriplet, ParsedObj};

use crate::render::layout::{
    GpuTransfer, GpuTransferRef, InstanceBufferRaw, InstanceRaw, TextureRaw, TriangleBufferRaw,
    VertexRaw,
};

mod parser;

pub const NUM_INSTANCES_PER_ROW: u32 = 10;

#[derive(Copy, Clone, Debug)]
pub struct Instance {
    pub position: cgmath::Vector3<f32>,
    pub rotation: cgmath::Quaternion<f32>,
}

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: cgmath::Point3<f32>,
    pub tex_coords: cgmath::Point2<f32>,
    pub normal: cgmath::Vector3<f32>,
    pub tangent: cgmath::Vector3<f32>,
    pub bitangent: cgmath::Vector3<f32>,
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

pub struct Material {
    #[expect(unused)]
    pub name: String,
    pub color_texture: Option<ColorTexture>,
    pub normal_texture: Option<NormalTexture>,
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub triangles: Vec<Triplet>,
    #[expect(unused)]
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

        Ok(Self {
            name: mtl.name.to_string(),
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
        let material = match &materials[..] {
            [mtl, ..] => Some(Material::load(root, mtl)?),
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

        Self {
            vertices,
            triangles,
            material_id: None,
        }
    }
}

// Safety: Triangles consist of 3 indices, and we check that all indices stay within vertex slice.
unsafe impl GpuTransfer for Mesh {
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
        let data = self.vertices.iter().map(|v| v.to_raw()).collect();
        TriangleBufferRaw {
            vertices: data,
            indices,
        }
    }
}

// Safety: No additional expectations beyond those of `InstanceRaw`.
unsafe impl GpuTransfer for Vec<Instance> {
    type Raw = InstanceBufferRaw;

    fn to_raw(&self) -> Self::Raw {
        let data = self.iter().map(|i| i.to_raw()).collect();
        InstanceBufferRaw { instances: data }
    }
}

// Safety: set size to the packed dimensions of the image, format matches with stored RGBA pixels.
unsafe impl<'a> GpuTransferRef<'a> for ColorTexture {
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

// Safety: set size to the packed dimensions of the image, format matches with stored RGBA pixels
unsafe impl<'a> GpuTransferRef<'a> for NormalTexture {
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

unsafe impl GpuTransfer for Vertex {
    type Raw = VertexRaw;
    fn to_raw(&self) -> Self::Raw {
        Self::Raw {
            position: self.position.into(),
            tex_coords: self.tex_coords.into(),
            normal: self.normal.into(),
            tangent: self.tangent.into(),
            bitangent: self.bitangent.into(),
        }
    }
}

// Safety: InstanceRaw restricts model to a row-major matrix with bottom right != 0
// and normal to the the same as top left 3x3 matrix of model
unsafe impl GpuTransfer for Instance {
    type Raw = InstanceRaw;
    fn to_raw(&self) -> Self::Raw {
        let model = (cgmath::Matrix4::from_translation(self.position)
            * cgmath::Matrix4::from(self.rotation))
        .into();
        let normal = cgmath::Matrix3::from(self.rotation).into();
        Self::Raw { model, normal }
    }
}

impl From<Triplet> for (u32, u32, u32) {
    fn from(Triplet(i, j, k): Triplet) -> Self {
        (i, j, k)
    }
}
