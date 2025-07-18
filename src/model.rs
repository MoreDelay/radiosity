use std::path::{Path, PathBuf};

use cgmath::{EuclideanSpace, InnerSpace, Zero};
use image::ImageReader;

use crate::{
    model::parser::{mtl, obj},
    primitives::{Color, Vertex},
    render::layout::{
        GpuTransfer, GpuTransferRef, InstanceBufferRaw, InstanceRaw, PhongRaw, TextureRaw,
        TriangleBufferRaw,
    },
};

pub mod parser;

pub const NUM_INSTANCES_PER_ROW: u32 = 10;

pub struct ModelStorage {
    current_root: Option<PathBuf>,
    meshes: Vec<Mesh>,
    materials: Vec<Material>,
    name_to_mesh: Vec<(String, usize)>,
    name_to_mtl: Vec<(String, usize)>,
    map_mtl_to_mesh: Vec<Vec<usize>>,
    meshes_without_mtl: Vec<usize>,
}

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

impl ModelStorage {
    #[expect(unused)]
    fn new() -> Self {
        Self {
            current_root: None,
            meshes: Vec::new(),
            materials: Vec::new(),
            name_to_mesh: Vec::new(),
            name_to_mtl: Vec::new(),
            map_mtl_to_mesh: Vec::new(),
            meshes_without_mtl: Vec::new(),
        }
    }

    #[expect(unused)]
    fn load_mesh(&mut self, path: &Path) -> anyhow::Result<()> {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let already_loaded = self.name_to_mesh.iter().any(|(n, _)| n == &name);
        if already_loaded {
            return Ok(());
        }

        let root = path.parent().unwrap().to_path_buf();
        self.current_root = Some(root);

        let loaded_obj = obj::load_obj(path, self)?;
        let mesh = Mesh::new(loaded_obj);

        let next_index = self.meshes.len();
        match mesh.material_id {
            Some(index) => self.map_mtl_to_mesh[index].push(next_index),
            None => self.meshes_without_mtl.push(next_index),
        }

        self.name_to_mesh.push((name, next_index));
        self.meshes.push(mesh);
        Ok(())
    }
}

impl parser::MtlManager for ModelStorage {
    fn request_load(&mut self, name: &str) -> Result<Vec<String>, parser::mtl::MtlError> {
        let root = self.current_root.as_ref().unwrap();
        let path = root.join(name);
        let loaded_mtls = parser::mtl::parse_mtl(&path)?;
        let mut mtls_in_this_lib = Vec::new();

        for mtl in loaded_mtls {
            mtls_in_this_lib.push(mtl.name.clone());

            let already_loaded = self.name_to_mtl.iter().any(|(n, _)| n == &mtl.name);
            if already_loaded {
                continue;
            }
            let next_index = self.materials.len();
            self.name_to_mtl.push((mtl.name.clone(), next_index));
            let new_material = Material::load(root, &mtl).unwrap();
            self.materials.push(new_material);
        }

        Ok(mtls_in_this_lib)
    }

    fn request_index(&self, name: &str) -> usize {
        self.name_to_mtl
            .iter()
            .find_map(|(n, i)| if n == name { Some(*i) } else { None })
            .unwrap()
    }
}

impl Material {
    pub fn load(root: &Path, mtl: &mtl::ParsedMtl) -> anyhow::Result<Self> {
        let color_texture = match &mtl.map_kd {
            None => None,
            Some(mtl::MtlMapKd(path)) => {
                let path = root.join(path);
                let image = ImageReader::open(path)?.decode()?;
                Some(ColorTexture(image.into()))
            }
        };
        let normal_texture = match &mtl.map_bump {
            None => None,
            Some(mtl::MtlMapBump(path)) => {
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
        let specular_exponent = mtl.ns.map(|mtl::MtlNs(exponent)| exponent).unwrap_or(10.);

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
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let abs_dir = path.canonicalize().unwrap().parent().unwrap().to_path_buf();
        let mut mtl_manager = parser::SimpleMtlManager::new(abs_dir.clone());

        let parsed_obj = parser::obj::load_obj(path, &mut mtl_manager)?;

        let mesh = Mesh::new(parsed_obj);
        let material = match mesh.material_id {
            Some(index) => Some(Material::load(&abs_dir, mtl_manager.get(index))?),
            _ => None,
        };
        Ok(Self { mesh, material })
    }
}

impl Mesh {
    fn new(parsed_obj: obj::ParsedObj) -> Self {
        let obj::ParsedObj {
            vertices,
            texture_coords,
            normals,
            faces,
            material_switches,
        } = parsed_obj;

        assert!(vertices.len() == normals.len());

        let mut vertices = vertices
            .into_iter()
            .map(|obj::ObjV { x, y, z, .. }| {
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
                let obj::ObjVn { i, j, k } = normals[i_normal];
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
                let obj::ObjVertexTriplet {
                    index_vertex,
                    index_texture,
                    index_normal,
                } = triplet;

                if let Some(index_texture) = index_texture {
                    let tex_coords = texture_coords[*index_texture];
                    let obj::ObjVt { u, v, w: _ } = tex_coords;
                    let vertex_tex = &mut vertices[*index_vertex].tex_coords;
                    *vertex_tex = cgmath::Point2 { x: u, y: v };
                }

                if let Some(index_normal) = index_normal {
                    let normal = normals[*index_normal];
                    let obj::ObjVn { i, j, k } = normal;
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
                    .flat_map(<&[obj::ObjVertexTriplet; 2]>::try_from)
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
