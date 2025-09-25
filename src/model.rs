use std::{
    collections::HashMap,
    ops::{Deref, Range},
    path::Path,
};

use bitvec::prelude::*;
use image::ImageReader;
use nalgebra::{self as na, Unit};

use crate::{
    model::parser::{mtl, obj},
    render,
};

pub mod parser;
pub mod primitives;

// reexport
pub use primitives::*;

#[expect(unused)]
const NUM_INSTANCES_PER_ROW: u32 = 10;
#[expect(unused)]
const SPACE_BETWEEN: f32 = 3.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshIndex {
    index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialIndex {
    index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexIndex {
    index: u32,
}

pub struct ModelStorage {
    meshes: Vec<Mesh>,
    materials: Vec<Material>,
}

struct ModelStorageInserter<'a> {
    root_dir: &'a Path,
    storage: &'a mut ModelStorage,
}

#[derive(Copy, Clone, Debug)]
pub struct Instance {
    pub position: na::Vector3<f32>,
    pub rotation: na::UnitQuaternion<f32>,
}

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
    pub name: String,
    pub phong_params: PhongParameters,
    pub color_texture: Option<ColorTexture>,
    pub normal_texture: Option<NormalTexture>,
}

#[derive(Debug, Clone)]
pub struct MaterialRanges {
    pub ranges: Vec<Range<u32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Triangle(VertexIndex, VertexIndex, VertexIndex);

pub struct Mesh {
    #[expect(unused)]
    pub name: Option<String>,
    pub vertices: Vec<Vertex>,
    pub triangles: Vec<Triangle>,
    pub mtl_ranges: HashMap<Option<MaterialIndex>, MaterialRanges>,
}

pub struct Model {
    pub meshes: Vec<Mesh>,
}

impl ModelStorage {
    pub fn new() -> Self {
        Self {
            meshes: Vec::new(),
            materials: Vec::new(),
        }
    }

    pub fn get_mesh(&self, MeshIndex { index }: MeshIndex) -> &Mesh {
        &self.meshes[index as usize]
    }

    pub fn get_material(&self, MaterialIndex { index }: MaterialIndex) -> &Material {
        &self.materials[index as usize]
    }

    pub fn load_meshes(&mut self, path: &Path) -> anyhow::Result<Vec<MeshIndex>> {
        let mut inserter = ModelStorageInserter {
            root_dir: path.parent().expect("file should have a parent directory"),
            storage: self,
        };

        let loaded_obj = obj::load_obj(path, &mut inserter)?;
        let model = Model::new(loaded_obj);

        let first_new_index = self.meshes.len() as u32;
        let new_indices = (0..model.meshes.len() as u32)
            .into_iter()
            .map(|v| MeshIndex {
                index: v + first_new_index,
            })
            .collect();

        self.meshes.extend(model.meshes);
        Ok(new_indices)
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

            let already_loaded = self.storage.materials.iter().any(|m| m.name == mtl.name);
            if already_loaded {
                continue;
            }
            let new_material = Material::load(self.root_dir, &mtl).unwrap();
            self.storage.materials.push(new_material);
        }

        Ok(mtls_in_this_lib)
    }

    fn request_mtl_index(&self, name: &str) -> Option<u32> {
        self.storage
            .materials
            .iter()
            .enumerate()
            .find_map(|(index, mtl)| (mtl.name == name).then_some(index as u32))
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

#[derive(Debug, Clone)]
struct Object {
    pub name: Option<String>,
    pub faces: Vec<obj::F>,
    #[expect(unused)]
    pub groups: HashMap<String, obj::FaceRanges>,
    pub mtls: HashMap<Option<u32>, obj::FaceRanges>,
    pub geo_vertices: Vec<obj::V>,
    pub tex_vertices: Vec<obj::Vt>,
    pub vertex_normals: Vec<obj::Vn>,
}

#[derive(Debug, Copy, Clone)]
struct SkipNode {
    start: u32,
    skipped: u32,
}

#[derive(Debug, Clone)]
struct SkipSequence {
    seq: Vec<SkipNode>,
}

impl SkipSequence {
    fn new(mut bitslice: &BitSlice) -> SkipSequence {
        let mut seq = vec![];

        let mut total = 0;
        let mut zeros = 0;

        loop {
            let Some(index) = bitslice.first_one() else {
                break;
            };

            let (_, rest) = bitslice.split_at(index);
            bitslice = rest;
            zeros += index as u32;
            total += index as u32;

            seq.push(SkipNode {
                start: total,
                skipped: zeros,
            });

            let Some(index) = bitslice.first_zero() else {
                break;
            };

            let (_, rest) = bitslice.split_at(index);
            bitslice = rest;
            total += index as u32;
        }

        SkipSequence { seq }
    }

    fn get_compacted_index(&self, old_index: u32) -> u32 {
        if self.seq.is_empty() {
            return old_index;
        }

        // binary search in skip sequence
        let mut seq_index = self.seq.len() / 2;

        loop {
            if self.seq[seq_index].start > old_index {
                seq_index = seq_index / 2;
                continue;
            }
            if self
                .seq
                .get(seq_index + 1)
                .is_some_and(|entry| entry.start <= old_index)
            {
                seq_index += (seq_index + 1) / 2;
                continue;
            }
            break;
        }

        let new_index = old_index - self.seq[seq_index].skipped;
        new_index
    }
}

impl Object {
    fn separate_objects(parsed_obj: obj::ParsedObj) -> Vec<Self> {
        let obj::ParsedObj {
            geo_vertices,
            tex_vertices,
            vertex_normals,
            objects,
        } = parsed_obj;

        objects
            .into_iter()
            .map(|obj| Object::create_single(obj, &geo_vertices, &tex_vertices, &vertex_normals))
            .collect()
    }

    fn create_single(
        obj: obj::Object,
        geo_vertices: &[obj::V],
        tex_vertices: &[obj::Vt],
        vertex_normals: &[obj::Vn],
    ) -> Object {
        let obj::Object {
            name,
            faces,
            groups,
            mtls,
        } = obj;

        let mut geo_bitvec = bitvec![0; geo_vertices.len()];
        let mut tex_bitvec = bitvec![0; tex_vertices.len()];
        let mut norm_bitvec = bitvec![0; vertex_normals.len()];

        for triplet in faces.iter().map(|f| &f.triplets).flatten() {
            let &obj::FTriplet {
                index_vertex,
                index_texture,
                index_normal,
            } = triplet;
            geo_bitvec.set(index_vertex as usize, true);
            if let Some(index_texture) = index_texture {
                tex_bitvec.set(index_texture as usize, true);
            }
            if let Some(index_normal) = index_normal {
                norm_bitvec.set(index_normal as usize, true);
            }
        }

        let geo_missing = SkipSequence::new(&geo_bitvec);
        let tex_missing = SkipSequence::new(&tex_bitvec);
        let norm_missing = SkipSequence::new(&norm_bitvec);

        let geo_vertices = geo_vertices
            .iter()
            .zip(geo_bitvec.iter())
            .filter_map(|(v, b)| b.then_some(v))
            .copied()
            .collect();

        let tex_vertices = tex_vertices
            .iter()
            .zip(tex_bitvec.iter())
            .filter_map(|(v, b)| b.then_some(v))
            .copied()
            .collect();

        let vertex_normals = vertex_normals
            .iter()
            .zip(norm_bitvec.iter())
            .filter_map(|(v, b)| b.then_some(v))
            .copied()
            .collect();

        let faces = faces
            .into_iter()
            .map(|f| {
                let triplets = f
                    .triplets
                    .into_iter()
                    .map(|t| {
                        let index_vertex = geo_missing.get_compacted_index(t.index_vertex);
                        let index_texture =
                            t.index_texture.map(|i| tex_missing.get_compacted_index(i));
                        let index_normal =
                            t.index_normal.map(|i| norm_missing.get_compacted_index(i));

                        obj::FTriplet {
                            index_vertex,
                            index_texture,
                            index_normal,
                        }
                    })
                    .collect();
                obj::F { triplets }
            })
            .collect();

        Object {
            name,
            faces,
            groups,
            mtls,
            geo_vertices,
            tex_vertices,
            vertex_normals,
        }
    }
}

struct TempVertex {
    position: na::Vector3<f32>,
    tex_coords: na::Vector2<f32>,
    normal: na::Vector3<f32>,
    tangent: na::Vector3<f32>,
    bitangent: na::Vector3<f32>,
}

struct InterleavedVertices {
    vertices: Vec<TempVertex>,
    map: HashMap<obj::FTriplet, u32>,
}

impl InterleavedVertices {
    fn new(faces: &[obj::F], geo: &[obj::V], tex: &[obj::Vt], norm: &[obj::Vn]) -> Self {
        let mut vertices = Vec::new();
        let mut map = HashMap::<obj::FTriplet, u32>::new();

        for obj::F { triplets } in faces.iter() {
            for &triplet in triplets.iter() {
                use std::collections::hash_map::Entry;

                match map.entry(triplet) {
                    Entry::Occupied(_) => (), // already present
                    Entry::Vacant(entry) => {
                        let index = vertices.len() as u32;
                        let position = {
                            let obj::V { x, y, z, w: _ } = geo[triplet.index_vertex as usize];
                            na::Vector3::new(x, y, z)
                        };
                        let tex_coords = match triplet.index_texture {
                            Some(index_texture) => {
                                let obj::Vt { u, v, w: _ } = tex[index_texture as usize];
                                na::Vector2::new(u, v)
                            }
                            None => na::Vector2::zeros(),
                        };
                        let normal = match triplet.index_normal {
                            Some(index_normal) => {
                                let obj::Vn { i, j, k } = norm[index_normal as usize];
                                na::Vector3::new(i, j, k)
                            }
                            None => na::Vector3::zeros(),
                        };
                        let new_v = TempVertex {
                            position,
                            tex_coords,
                            normal,
                            tangent: na::Vector3::zeros(),
                            bitangent: na::Vector3::zeros(),
                        };
                        vertices.push(new_v);
                        entry.insert(index);
                    }
                }
            }
        }

        Self { vertices, map }
    }
}

impl Mesh {
    fn new(obj: Object) -> Self {
        let Object {
            name,
            faces,
            groups: _,
            mtls: old_ranges,
            geo_vertices,
            tex_vertices,
            vertex_normals,
        } = obj;

        let mut interleaved =
            InterleavedVertices::new(&faces, &geo_vertices, &tex_vertices, &vertex_normals);

        // make all faces to triangles naively, and update ranges correspondingly
        struct ActiveMaterial {
            index: Option<u32>,
            range: Range<u32>,
            new_start: u32,
        }

        let find_next_active_material = |face_index: u32| -> ActiveMaterial {
            let (index, range) = old_ranges
                .iter()
                .find_map(|(index, face_ranges)| {
                    if let Some(slice) = face_ranges
                        .slices
                        .iter()
                        .find(|Range { start, .. }| *start == face_index)
                    {
                        Some((*index, slice.clone()))
                    } else {
                        None
                    }
                })
                .expect("all faces must be assigned to a material");

            ActiveMaterial {
                index,
                range,
                new_start: face_index,
            }
        };

        let mut new_ranges: HashMap<Option<MaterialIndex>, MaterialRanges> = HashMap::new();
        let mut active_mtl = find_next_active_material(0);
        let mut triangles = Vec::new();
        for (old_index, face) in faces.into_iter().enumerate() {
            let old_index = old_index as u32;
            assert!(face.triplets.len() >= 3, "too few vertices, not a face!");

            if active_mtl.range.end == old_index {
                // update last material and find next one
                if active_mtl.new_start != old_index {
                    let new_slice = active_mtl.new_start..triangles.len() as u32;
                    let index = active_mtl.index.map(|index| MaterialIndex { index });
                    new_ranges
                        .entry(index)
                        .and_modify(|e| e.ranges.push(new_slice.clone()))
                        .or_insert_with(|| MaterialRanges {
                            ranges: vec![new_slice],
                        });
                }

                active_mtl = find_next_active_material(old_index);
            }

            let ([first], other) = face.triplets.split_at(1) else {
                unreachable!("checked above");
            };
            let first = VertexIndex {
                index: interleaved.map[first],
            };
            let new_triangles = other
                .windows(2)
                .flat_map(<&[obj::FTriplet; 2]>::try_from)
                .map(|[second, third]| {
                    let second = VertexIndex {
                        index: interleaved.map[second],
                    };
                    let third = VertexIndex {
                        index: interleaved.map[third],
                    };
                    Triangle(first, second, third)
                });
            triangles.extend(new_triangles);
        }
        // let triangles = faces
        //     .into_iter()
        //     .enumerate()
        //     .flat_map(|(orig_f_index, face)| {})
        //     .collect::<Vec<_>>();

        // add last texture
        let last_slice = active_mtl.new_start..triangles.len() as u32;
        let index = active_mtl.index.map(|index| MaterialIndex { index });
        new_ranges
            .entry(index)
            .and_modify(|e| e.ranges.push(last_slice.clone()))
            .or_insert_with(|| MaterialRanges {
                ranges: vec![last_slice],
            });

        // compute normals and tangents
        for Triangle(first, second, third) in triangles.iter() {
            let normal =
                if interleaved.vertices[first.index as usize].normal == na::Vector3::zeros() {
                    let first = interleaved.vertices[first.index as usize].position;
                    let second = interleaved.vertices[second.index as usize].position;
                    let third = interleaved.vertices[third.index as usize].position;

                    let v0 = second - first;
                    let v1 = third - first;
                    Some(na::Unit::new_normalize(v0.cross(&v1)))
                } else {
                    None
                };

            for VertexIndex { index: vertex } in [first, second, third] {
                let normal =
                    if interleaved.vertices[first.index as usize].normal == na::Vector3::zeros() {
                        normal.expect("computed before")
                    } else {
                        let normal = interleaved.vertices[*vertex as usize].normal;
                        na::Unit::new_normalize(normal)
                    };

                let unit = if (*normal - *na::Vector3::x_axis()).magnitude() > 0.1 {
                    na::Vector3::x_axis()
                } else {
                    na::Vector3::y_axis()
                };
                let tangent = Unit::new_normalize(normal.cross(&unit));
                let bitangent = Unit::new_normalize(normal.cross(&tangent));

                interleaved.vertices[*vertex as usize].normal = *normal;
                interleaved.vertices[*vertex as usize].tangent = *tangent;
                interleaved.vertices[*vertex as usize].bitangent = *bitangent;
            }
        }

        let vertices = interleaved
            .vertices
            .into_iter()
            .map(|v: TempVertex| {
                let TempVertex {
                    position,
                    tex_coords,
                    normal,
                    tangent,
                    bitangent,
                } = v;
                Vertex {
                    position,
                    tex_coords,
                    // normalized above
                    normal: na::Unit::new_unchecked(normal),
                    tangent: na::Unit::new_unchecked(tangent),
                    bitangent: na::Unit::new_unchecked(bitangent),
                }
            })
            .collect();

        Mesh {
            name,
            triangles,
            mtl_ranges: new_ranges,
            vertices,
        }
    }
}

impl Model {
    fn new(parsed_obj: obj::ParsedObj) -> Self {
        let meshes = Object::separate_objects(parsed_obj)
            .into_iter()
            .map(|obj| Mesh::new(obj))
            .collect();

        Self { meshes }
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
            .flat_map(|&Triangle(i, j, k)| {
                assert!(i.index < upper_bound);
                assert!(j.index < upper_bound);
                assert!(k.index < upper_bound);
                [i.index, j.index, k.index]
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
        let model = na::Translation::from(self.position) * self.rotation;
        let model = model.to_matrix().into();
        let normal = self.rotation.to_rotation_matrix();
        let normal = normal.matrix().clone().into();
        render::InstanceRaw { model, normal }
    }
}

impl From<Triangle> for (u32, u32, u32) {
    fn from(Triangle(i, j, k): Triangle) -> Self {
        (i.index, j.index, k.index)
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
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}

impl Deref for MaterialIndex {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}
