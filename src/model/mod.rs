use std::{collections::HashMap, ops::Range, path::PathBuf};

use bitvec::prelude::*;
use image::GenericImageView;
use nalgebra as na;

use crate::{
    model::parser::{mtl, obj},
    render,
};

pub mod parser;

pub struct IndexBuffer {
    pub triangles: Vec<[u32; 3]>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct IndexBufferIndex(u32);

#[derive(Clone, Debug)]
pub struct IndexBufferView {
    pub buffer: IndexBufferIndex,
    pub range: Range<u32>,
}

#[derive(Clone)]
pub struct PositionBuffer {
    pub positions: Vec<[f32; 3]>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PositionBufferIndex(u32);

pub struct TexCoordBuffer {
    pub tex_coords: Vec<[f32; 2]>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TexCoordBufferIndex(u32);

pub struct NormalBuffer {
    pub normals: Vec<[f32; 3]>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct NormalBufferIndex(u32);

pub struct ComputedNormals(NormalBuffer);

pub struct TangentBuffer {
    pub tangents: Vec<[f32; 3]>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TangentBufferIndex(u32);

#[derive(Clone, Debug)]
pub struct BlinnPhong {
    pub ambient_base: Color,
    pub diffuse_base: Color,
    pub specular_base: Color,
    pub specular_exponent: f32,
    pub diffuse_map: Option<TextureIndex>,
}

impl BlinnPhong {
    pub fn to_raw(&self) -> render::PhongRaw {
        render::PhongRaw {
            specular_color: self.specular_base.into(),
            specular_exponent: self.specular_exponent,
            diffuse_color: self.diffuse_base.into(),
            _padding1: 0,
            ambient_color: self.ambient_base.into(),
            _padding2: 0,
        }
    }
}

impl Default for BlinnPhong {
    fn default() -> Self {
        Self {
            ambient_base: Color::from(mtl::MtlKa::default()),
            diffuse_base: Color::from(mtl::MtlKd::default()),
            diffuse_map: None,
            specular_base: Color::from(mtl::MtlKs::default()),
            specular_exponent: mtl::MtlNs::default().0,
        }
    }
}

pub enum MaterialType {
    BlinnPhong(BlinnPhong),
    // TODO: Add Metallic Roughness type
}

pub struct Material {
    pub data: MaterialType,
    pub normal: Option<TextureIndex>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct MaterialIndex(u32);

#[derive(Debug, Copy, Clone, Default)]
pub struct Sampler {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub mipmap_filter: Filter,
    pub wrap_s: WrapMode,
    pub wrap_t: WrapMode,
}

impl Sampler {
    pub fn to_desc(self, label: Option<&str>) -> wgpu::SamplerDescriptor<'_> {
        wgpu::SamplerDescriptor {
            label,
            address_mode_u: self.wrap_s.into(),
            address_mode_v: self.wrap_t.into(),
            mag_filter: self.mag_filter.into(),
            min_filter: self.min_filter.into(),
            mipmap_filter: self.mipmap_filter.into(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Texture {
    pub sampler: Sampler,
    pub image: ImageIndex,
}

impl Texture {
    pub fn to_raw(&self, storage: &Storage, format: wgpu::TextureFormat) -> render::TextureRaw {
        let &Texture { sampler, image } = self;
        let image = storage.image(image);
        let (data, dims) = match image {
            Image::Path(path) => {
                let image = image::ImageReader::open(path)
                    .expect("path must exist")
                    .decode()
                    .expect("image must be valid");
                let dims = image.dimensions();
                let image = image.into();
                (image, dims)
            }
            Image::Data { data, .. } => {
                let dims = data.dimensions();
                let data = data.clone();
                (data, dims)
            }
        };
        let data = data.into_vec().into();
        let (width, height) = dims;
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        render::TextureRaw {
            data,
            format,
            size,
            sampler,
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TextureIndex(u32);

#[derive(Debug)]
pub enum Image {
    Path(PathBuf),
    #[expect(unused)]
    Data {
        data: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
        path: Option<PathBuf>,
    },
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ImageIndex(u32);

#[derive(Debug, Clone)]
pub struct PrimitiveData {
    pub position: PositionBufferIndex,
    pub normal: Option<(NormalBufferIndex, TangentBufferIndex)>,
    pub tex_coord: Option<TexCoordBufferIndex>,
}

#[derive(Debug, Clone)]
pub struct Primitive {
    pub data: PrimitiveData,
    pub indices: IndexBufferView,
    pub material: MaterialIndex,
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub primitives: Vec<Primitive>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct MeshIndex(u32);

pub struct Storage {
    index_buffers: Vec<IndexBuffer>,
    vertex_buffers: Vec<PositionBuffer>,
    tex_coord_buffers: Vec<TexCoordBuffer>,
    normal_buffers: Vec<NormalBuffer>,
    tangent_buffers: Vec<TangentBuffer>,
    materials: Vec<Material>,
    textures: Vec<Texture>,
    images: Vec<Image>,
    meshes: Vec<Mesh>,

    obj_default_material: Option<MaterialIndex>,
}

impl Storage {
    pub fn new() -> Self {
        Self {
            index_buffers: Vec::new(),
            vertex_buffers: Vec::new(),
            tex_coord_buffers: Vec::new(),
            normal_buffers: Vec::new(),
            tangent_buffers: Vec::new(),
            materials: Vec::new(),
            textures: Vec::new(),
            images: Vec::new(),
            meshes: Vec::new(),
            obj_default_material: None,
        }
    }

    pub fn index_buffer(&self, index: IndexBufferIndex) -> &IndexBuffer {
        &self.index_buffers[index.0 as usize]
    }

    pub fn position_buffer(&self, index: PositionBufferIndex) -> &PositionBuffer {
        &self.vertex_buffers[index.0 as usize]
    }

    pub fn tex_coord_buffer(&self, index: TexCoordBufferIndex) -> &TexCoordBuffer {
        &self.tex_coord_buffers[index.0 as usize]
    }

    pub fn normal_buffer(&self, index: NormalBufferIndex) -> &NormalBuffer {
        &self.normal_buffers[index.0 as usize]
    }

    pub fn tangent_buffer(&self, index: TangentBufferIndex) -> &TangentBuffer {
        &self.tangent_buffers[index.0 as usize]
    }

    pub fn material(&self, index: MaterialIndex) -> &Material {
        &self.materials[index.0 as usize]
    }

    pub fn texture(&self, index: TextureIndex) -> &Texture {
        &self.textures[index.0 as usize]
    }

    pub fn image(&self, index: ImageIndex) -> &Image {
        &self.images[index.0 as usize]
    }

    pub fn mesh(&self, index: MeshIndex) -> &Mesh {
        &self.meshes[index.0 as usize]
    }
}

impl Storage {
    pub fn store_obj(
        &mut self,
        obj: parser::ParsedObj,
        mtls: impl IntoIterator<Item = parser::ParsedMtl>,
    ) -> MeshIndex {
        let material_indices = self.store_mtls(mtls);
        let objects = CompactedObject::separate_objects(obj);

        let primitives = objects
            .into_iter()
            .flat_map(MeshCombined::new)
            .flat_map(|object| {
                let normal_buffer = if let Some(normals) = object.normal_buffer_specified {
                    normals
                } else {
                    object.compute_normals().0
                };

                let tangent_buffer = MeshCombined::compute_tangents(&normal_buffer);

                let MeshCombined {
                    _name: _,
                    vertex_buffer,
                    tex_coord_buffer,
                    materials,
                    normal_buffer_specified: _,
                    index_buffer,
                } = object;

                let indices = self.store_index_buffer(index_buffer);
                let vertex = self.store_position_buffer(vertex_buffer);
                let tex_coord = tex_coord_buffer.map(|uv| self.store_tex_coord_buffer(uv));
                let normal = self.store_normal_buffer(normal_buffer);
                let tangent = self.store_tangent_buffer(tangent_buffer);

                let normal = Some((normal, tangent));

                materials
                    .into_iter()
                    .map(|(mtl_index, face_range)| {
                        let data = PrimitiveData {
                            position: vertex,
                            normal,
                            tex_coord,
                        };

                        let indices = IndexBufferView {
                            buffer: indices,
                            range: face_range.0,
                        };

                        let material = if let Some(old_index) = mtl_index {
                            material_indices[old_index as usize]
                        } else {
                            self.use_obj_default_material()
                        };

                        Primitive {
                            data,
                            indices,
                            material,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let mesh = Mesh { primitives };
        self.store_mesh(mesh)
    }

    fn store_mtls(
        &mut self,
        mtls: impl IntoIterator<Item = parser::ParsedMtl>,
    ) -> Vec<MaterialIndex> {
        let mut texture_map: HashMap<PathBuf, TextureIndex> = HashMap::new();

        let mut to_index = |inner: &mut Self, path: PathBuf| {
            use std::collections::hash_map::Entry;

            let entry = match texture_map.entry(path) {
                Entry::Occupied(entry) => return *entry.get(),
                Entry::Vacant(entry) => entry,
            };

            let path = entry.key().clone();
            let image = Image::Path(path);
            let index = inner.store_image(image);
            let texture = Texture {
                sampler: Sampler::default(),
                image: index,
            };
            let index = inner.store_texture(texture);
            entry.insert(index);
            index
        };

        mtls.into_iter()
            .map(|mtl| {
                let mtl::ParsedMtl {
                    ka,
                    kd,
                    ks,
                    ns,
                    map_bump,
                    map_kd,
                    ..
                } = mtl;

                let ka = ka.unwrap_or_else(parser::mtl::MtlKa::default);
                let kd = kd.unwrap_or_else(parser::mtl::MtlKd::default);
                let ks = ks.unwrap_or_else(parser::mtl::MtlKs::default);
                let ns = ns.unwrap_or_else(parser::mtl::MtlNs::default);

                let ambient_base = Color {
                    r: ka.0,
                    g: ka.1,
                    b: ka.2,
                };
                let diffuse_base = Color {
                    r: kd.0,
                    g: kd.1,
                    b: kd.2,
                };
                let specular_base = Color {
                    r: ks.0,
                    g: ks.1,
                    b: ks.2,
                };
                let specular_exponent = ns.0;

                let diffuse_map = map_kd.map(|path| to_index(self, path.0));
                let normal_map = map_bump.map(|path| to_index(self, path.0));

                let mtl = BlinnPhong {
                    ambient_base,
                    diffuse_base,
                    specular_base,
                    specular_exponent,
                    diffuse_map,
                };
                let mtl = Material {
                    data: MaterialType::BlinnPhong(mtl),
                    normal: normal_map,
                };

                self.store_material(mtl)
            })
            .collect()
    }

    fn use_obj_default_material(&mut self) -> MaterialIndex {
        if let Some(index) = self.obj_default_material {
            return index;
        }

        let data = BlinnPhong::default();
        let data = MaterialType::BlinnPhong(data);
        let default = Material { data, normal: None };
        let index = self.store_material(default);
        self.obj_default_material = Some(index);
        index
    }

    fn store_index_buffer(&mut self, buffer: IndexBuffer) -> IndexBufferIndex {
        let index = IndexBufferIndex(self.vertex_buffers.len() as u32);
        self.index_buffers.push(buffer);
        index
    }

    pub fn store_position_buffer(&mut self, buffer: PositionBuffer) -> PositionBufferIndex {
        let index = PositionBufferIndex(self.vertex_buffers.len() as u32);
        self.vertex_buffers.push(buffer);
        index
    }

    fn store_tex_coord_buffer(&mut self, buffer: TexCoordBuffer) -> TexCoordBufferIndex {
        let index = TexCoordBufferIndex(self.tex_coord_buffers.len() as u32);
        self.tex_coord_buffers.push(buffer);
        index
    }

    fn store_normal_buffer(&mut self, buffer: NormalBuffer) -> NormalBufferIndex {
        let index = NormalBufferIndex(self.normal_buffers.len() as u32);
        self.normal_buffers.push(buffer);
        index
    }

    fn store_tangent_buffer(&mut self, buffer: TangentBuffer) -> TangentBufferIndex {
        let index = TangentBufferIndex(self.tangent_buffers.len() as u32);
        self.tangent_buffers.push(buffer);
        index
    }

    pub fn store_material(&mut self, material: Material) -> MaterialIndex {
        let index = MaterialIndex(self.materials.len() as u32);
        self.materials.push(material);
        index
    }

    fn store_texture(&mut self, texture: Texture) -> TextureIndex {
        let index = TextureIndex(self.textures.len() as u32);
        self.textures.push(texture);
        index
    }

    fn store_image(&mut self, image: Image) -> ImageIndex {
        let index = ImageIndex(self.images.len() as u32);
        self.images.push(image);
        index
    }

    pub fn store_mesh(&mut self, mesh: Mesh) -> MeshIndex {
        let index = MeshIndex(self.meshes.len() as u32);
        self.meshes.push(mesh);
        index
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum Filter {
    Nearest,
    #[default]
    Linear,
}

impl From<Filter> for wgpu::FilterMode {
    fn from(value: Filter) -> Self {
        match value {
            Filter::Nearest => wgpu::FilterMode::Nearest,
            Filter::Linear => wgpu::FilterMode::Linear,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum WrapMode {
    ClampToEdge,
    MirroredRepeat,
    #[default]
    Repeat,
}

impl From<WrapMode> for wgpu::AddressMode {
    fn from(value: WrapMode) -> Self {
        match value {
            WrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            WrapMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
            WrapMode::Repeat => wgpu::AddressMode::Repeat,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl From<Color> for [f32; 3] {
    fn from(val: Color) -> Self {
        let Color { r, g, b, .. } = val;
        [r, g, b]
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Instance {
    pub position: na::Vector3<f32>,
    pub rotation: na::UnitQuaternion<f32>,
}

pub struct MeshCombined {
    pub _name: Option<String>,
    pub vertex_buffer: PositionBuffer,
    pub tex_coord_buffer: Option<TexCoordBuffer>,
    pub materials: Vec<(Option<u32>, obj::FaceRange)>,
    pub normal_buffer_specified: Option<NormalBuffer>,
    pub index_buffer: IndexBuffer,
}

impl MeshCombined {
    fn compute_tangents(normals: &NormalBuffer) -> TangentBuffer {
        let tangents = normals
            .normals
            .iter()
            .map(|n| {
                let n = na::Unit::new_normalize(na::Vector3::from_row_slice(n));
                let basis = crate::math::orthonormal_basis_for_normal(&n);
                let t = basis.column(1);
                [t[0], t[1], t[2]]
            })
            .collect();
        TangentBuffer { tangents }
    }
}

#[derive(Debug, Clone)]
struct CompactedObject {
    pub name: Option<String>,
    pub faces: Vec<obj::F>,
    pub _groups: Vec<(String, obj::FaceRange)>,
    pub mtls: Vec<(Option<u32>, obj::FaceRange)>,
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
                seq_index /= 2;
                continue;
            }
            if self
                .seq
                .get(seq_index + 1)
                .is_some_and(|entry| entry.start <= old_index)
            {
                seq_index += seq_index.div_ceil(2);
                continue;
            }
            break;
        }

        old_index - self.seq[seq_index].skipped
    }
}

impl CompactedObject {
    fn separate_objects(parsed_obj: obj::ParsedObj) -> Vec<Self> {
        let obj::ParsedObj {
            geo_vertices,
            tex_vertices,
            vertex_normals,
            objects,
        } = parsed_obj;

        objects
            .into_iter()
            .flat_map(|obj| {
                CompactedObject::create_single(obj, &geo_vertices, &tex_vertices, &vertex_normals)
            })
            .collect()
    }

    fn create_single(
        obj: obj::Object,
        geo_vertices: &[obj::V],
        tex_vertices: &[obj::Vt],
        vertex_normals: &[obj::Vn],
    ) -> Option<CompactedObject> {
        let obj::Object {
            name,
            faces,
            groups,
            mtls,
        } = obj;

        if faces.is_empty() {
            return None;
        }

        let mut geo_bitvec = bitvec![0; geo_vertices.len()];
        let mut tex_bitvec = bitvec![0; tex_vertices.len()];
        let mut norm_bitvec = bitvec![0; vertex_normals.len()];

        for triplet in faces.iter().flat_map(|f| &f.triplets) {
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

        Some(CompactedObject {
            name,
            faces,
            _groups: groups,
            mtls,
            geo_vertices,
            tex_vertices,
            vertex_normals,
        })
    }
}

impl MeshCombined {
    fn new(obj: CompactedObject) -> Option<Self> {
        let CompactedObject {
            name,
            faces,
            _groups: _,
            mut mtls,
            geo_vertices: old_geo,
            tex_vertices: old_tex,
            vertex_normals: old_normals,
        } = obj;

        if faces.is_empty() {
            return None;
        }

        use std::collections::hash_map::Entry;

        let mut triplet_to_index = HashMap::<obj::FTriplet, u32>::new();
        let mut has_normals = true;
        let mut has_tex = true;

        for obj::F { triplets } in faces.iter() {
            for &triplet in triplets.iter() {
                has_normals &= triplet.index_normal.is_some();
                has_tex &= triplet.index_texture.is_some();

                let index = triplet_to_index.len() as u32;

                match triplet_to_index.entry(triplet) {
                    Entry::Occupied(_) => (), // already present
                    Entry::Vacant(entry) => {
                        entry.insert(index);
                    }
                }
            }
        }

        let n_vertices = triplet_to_index.len();

        let mut new_vertices = Vec::with_capacity(n_vertices);
        new_vertices.resize_with(n_vertices, || [0f32; 3]);

        let mut new_normals = has_normals.then(|| {
            let mut vec = Vec::with_capacity(n_vertices);
            vec.resize_with(n_vertices, na::Vector3::zeros);
            vec
        });

        let mut new_tex = has_tex.then(|| {
            let mut vec = Vec::with_capacity(n_vertices);
            vec.resize_with(n_vertices, || [0f32; 2]);
            vec
        });

        for (triplet, index) in triplet_to_index.iter() {
            let &obj::FTriplet {
                index_vertex,
                index_texture,
                index_normal,
            } = triplet;

            let obj::V { x, y, z, w: _ } = old_geo[index_vertex as usize];
            new_vertices[*index as usize] = [x, y, z];

            if let Some(new_normals) = new_normals.as_mut() {
                let index_normal = index_normal.expect("checked before");
                let obj::Vn { i, j, k } = old_normals[index_normal as usize];
                let normal = na::Vector3::new(i, j, k);
                new_normals[*index as usize] = normal;
            }

            if let Some(new_tex) = new_tex.as_mut() {
                let index_texture = index_texture.expect("checked before");
                let obj::Vt { u, v, w: _ } = old_tex[index_texture as usize];
                new_tex[*index as usize] = [u, v];
            }
        }

        let new_vertices = PositionBuffer {
            positions: new_vertices,
        };
        let new_tex = new_tex.map(|vertices| TexCoordBuffer {
            tex_coords: vertices,
        });

        // make triangles out of polygons naively
        let mut mtl_iter = mtls.iter_mut();
        let (_, cur_range_ref) = mtl_iter.next().expect("at least one material (default)");
        let mut cur_range_ref = cur_range_ref;
        let mut new_start = 0;
        let mut new_index = 0;

        let triangles = faces
            .into_iter()
            .enumerate()
            .flat_map(|(face_index, face)| {
                assert!(face.triplets.len() >= 3, "too few vertices, not a face!");

                let face_index = face_index as u32;
                if face_index == cur_range_ref.0.end {
                    // update previous material range
                    let range = new_start..new_index;
                    *cur_range_ref = obj::FaceRange(range);

                    // keep track of next material
                    new_start = new_index;
                    let (_, next_range_ref) = mtl_iter.next().expect("not at end yet");
                    cur_range_ref = next_range_ref;
                }

                let first = *face.triplets.first().expect("checked above");
                let first = triplet_to_index[&first];
                face.triplets[1..]
                    .windows(2)
                    .flat_map(<&[obj::FTriplet; 2]>::try_from)
                    .map(|[second, third]| {
                        new_index += 1;

                        let second = triplet_to_index[second];
                        let third = triplet_to_index[third];

                        [first, second, third]
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // complete last material
        let new_end = triangles.len() as u32;
        let range = new_start..new_end;
        *cur_range_ref = obj::FaceRange(range);

        let triangles = IndexBuffer { triangles };

        let new_normals = new_normals.map(|new_normals| {
            new_normals
                .into_iter()
                .map(|n| {
                    let n = na::Unit::new_normalize(n);
                    [n[0], n[1], n[2]]
                })
                .collect()
        });
        let new_normals = new_normals.map(|normals| NormalBuffer { normals });

        let materials = if has_tex { mtls } else { Vec::new() };

        Some(Self {
            _name: name,
            vertex_buffer: new_vertices,
            tex_coord_buffer: new_tex,
            materials,
            normal_buffer_specified: new_normals,
            index_buffer: triangles,
        })
    }

    fn compute_normals(&self) -> ComputedNormals {
        let mut normals_computed = Vec::new();
        normals_computed.resize_with(self.vertex_buffer.positions.len(), na::Vector3::zeros);

        for &[i0, i1, i2] in self.index_buffer.triangles.iter() {
            let p0 = &self.vertex_buffer.positions[i0 as usize];
            let p1 = &self.vertex_buffer.positions[i1 as usize];
            let p2 = &self.vertex_buffer.positions[i2 as usize];

            let p0 = na::Vector3::from_column_slice(p0);
            let p1 = na::Vector3::from_column_slice(p1);
            let p2 = na::Vector3::from_column_slice(p2);

            let v0 = p1 - p0;
            let v1 = p2 - p0;
            let normal = na::Unit::new_normalize(v0.cross(&v1));

            normals_computed[i0 as usize] += *normal;
            normals_computed[i1 as usize] += *normal;
            normals_computed[i2 as usize] += *normal;
        }

        let normals = normals_computed
            .into_iter()
            .map(|n| {
                let n = na::Unit::new_normalize(n);
                [n[0], n[1], n[2]]
            })
            .collect();

        ComputedNormals(NormalBuffer { normals })
    }
}

impl Instance {
    pub fn to_raw(self) -> render::InstanceRaw {
        let model = na::Translation::from(self.position) * self.rotation;
        let model = model.to_matrix().into();
        render::InstanceRaw { model }
    }
}
