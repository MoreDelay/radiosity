use std::{ops::Deref, path::Path};

use cgmath::{EuclideanSpace, InnerSpace, Zero};
use image::ImageReader;

use crate::{
    model::parser::{mtl, obj},
    primitives::{Color, Vertex},
    render,
};

pub mod parser;

pub const NUM_INSTANCES_PER_ROW: u32 = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshIndex {
    index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialIndex {
    index: usize,
}

pub struct ModelStorage {
    meshes: Vec<(String, Mesh)>,
    materials: Vec<(String, Material)>,
}

struct ModelStorageInserter<'a> {
    root_dir: &'a Path,
    storage: &'a mut ModelStorage,
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
    pub phong_params: PhongParameters,
    pub color_texture: Option<ColorTexture>,
    pub normal_texture: Option<NormalTexture>,
}

pub struct MaterialSwitch {
    pub first_face: usize,
    pub material: MaterialIndex,
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub triangles: Vec<Triplet>,
    pub material_switches: Vec<MaterialSwitch>,
}

impl ModelStorage {
    pub fn new() -> Self {
        Self {
            meshes: Vec::new(),
            materials: Vec::new(),
        }
    }

    pub fn get_mesh(&self, MeshIndex { index }: MeshIndex) -> &Mesh {
        let (_, mesh) = &self.meshes[index];
        mesh
    }

    pub fn get_material(&self, MaterialIndex { index }: MaterialIndex) -> &Material {
        let (_, material) = &self.materials[index];
        material
    }

    pub fn load_mesh(&mut self, path: &Path) -> anyhow::Result<MeshIndex> {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let already_loaded = self
            .meshes
            .iter()
            .enumerate()
            .find_map(|(i, (n, _))| (n == &name).then_some(i));
        if let Some(index) = already_loaded {
            return Ok(MeshIndex { index });
        }

        let mut inserter = ModelStorageInserter {
            root_dir: path.parent().expect("file should have a parent directory"),
            storage: self,
        };

        let loaded_obj = obj::load_obj(path, &mut inserter)?;
        let mesh = Mesh::new(loaded_obj);

        let mesh_index = self.meshes.len();

        self.meshes.push((name, mesh));
        Ok(MeshIndex { index: mesh_index })
    }
}

impl<'a> parser::MtlManager for ModelStorageInserter<'a> {
    fn request_mtl_load(&mut self, filename: &str) -> Result<Vec<String>, parser::mtl::MtlError> {
        let path = self.root_dir.join(filename);
        let loaded_mtls = parser::mtl::parse_mtl(&path)?;
        let mut mtls_in_this_lib = Vec::new();

        for mtl in loaded_mtls {
            assert!(!mtls_in_this_lib.contains(&mtl.name), "duplicate material");
            mtls_in_this_lib.push(mtl.name.clone());

            let already_loaded = self.storage.materials.iter().any(|(n, _)| n == &mtl.name);
            if already_loaded {
                continue;
            }
            let new_material = Material::load(self.root_dir, &mtl).unwrap();
            self.storage.materials.push((mtl.name, new_material));
        }

        Ok(mtls_in_this_lib)
    }

    fn request_mtl_index(&self, name: &str) -> Option<usize> {
        self.storage
            .materials
            .iter()
            .enumerate()
            .find_map(|(index, (mesh_name, _))| if mesh_name == name { Some(index) } else { None })
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

        let ambient_color = mtl
            .ka
            .map(|c| c.into())
            .unwrap_or(mtl::MtlKa::default().into());
        let diffuse_color = mtl
            .kd
            .map(|c| c.into())
            .unwrap_or(mtl::MtlKd::default().into());
        let specular_color = mtl
            .ks
            .map(|c| c.into())
            .unwrap_or(mtl::MtlKs::default().into());
        let specular_exponent = mtl
            .ns
            .map(|mtl::MtlNs(exponent)| exponent)
            .unwrap_or(mtl::MtlNs::default().0);

        let phong_params = PhongParameters {
            ambient_color,
            diffuse_color,
            specular_color,
            specular_exponent,
        };

        Ok(Self {
            name: mtl.name.to_string(),
            phong_params,
            color_texture,
            normal_texture,
        })
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

        // TODO: need to adapt first face for materials after triangulation
        let material_switches = material_switches
            .into_iter()
            .map(|switch| MaterialSwitch {
                first_face: switch.first_face,
                material: MaterialIndex {
                    index: switch.material_index,
                },
            })
            .collect();
        Self {
            vertices,
            triangles,
            material_switches,
        }
    }
}

impl Default for PhongParameters {
    fn default() -> Self {
        Self {
            ambient_color: Color::from(mtl::MtlKa::default()),
            diffuse_color: Color::from(mtl::MtlKd::default()),
            specular_color: Color::from(mtl::MtlKs::default()),
            specular_exponent: mtl::MtlNs::default().0,
        }
    }
}

impl render::GpuTransfer for Mesh {
    type Raw = render::TriangleBufferRaw;

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
        render::TriangleBufferRaw { vertices, indices }
    }
}

impl render::GpuTransfer for &[Instance] {
    type Raw = render::InstanceBufferRaw;

    fn to_raw(&self) -> Self::Raw {
        let instances = self.iter().map(|i| i.to_raw()).collect();
        render::InstanceBufferRaw { instances }
    }
}

impl<'a> render::GpuTransferRef<'a> for ColorTexture {
    type Raw = render::TextureRaw<'a>;

    fn to_raw(&'a self) -> render::TextureRaw<'a> {
        let Self(image) = self;
        let data = &image;
        let dimensions = image.dimensions();
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        render::TextureRaw { data, format, size }
    }
}

impl<'a> render::GpuTransferRef<'a> for NormalTexture {
    type Raw = render::TextureRaw<'a>;

    fn to_raw(&'a self) -> render::TextureRaw<'a> {
        let Self(image) = self;
        let data = &image;
        let dimensions = image.dimensions();
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        render::TextureRaw { data, format, size }
    }
}

impl render::GpuTransfer for Instance {
    type Raw = render::InstanceRaw;
    fn to_raw(&self) -> Self::Raw {
        let model = (cgmath::Matrix4::from_translation(self.position)
            * cgmath::Matrix4::from(self.rotation))
        .into();
        let normal = cgmath::Matrix3::from(self.rotation).into();
        render::InstanceRaw { model, normal }
    }
}

impl From<Triplet> for (u32, u32, u32) {
    fn from(Triplet(i, j, k): Triplet) -> Self {
        (i, j, k)
    }
}

impl render::GpuTransfer for PhongParameters {
    type Raw = render::PhongRaw;

    fn to_raw(&self) -> Self::Raw {
        render::PhongRaw {
            specular_color: self.specular_color.into(),
            specular_exponent: self.specular_exponent,
            diffuse_color: self.diffuse_color.into(),
            _padding1: 0,
            ambient_color: self.ambient_color.into(),
            _padding2: 0,
        }
    }
}

impl Deref for MeshIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}

impl Deref for MaterialIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}
