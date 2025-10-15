use std::{
    collections::{HashMap, HashSet},
    ops::{Deref, Range},
    path::{Path, PathBuf},
    sync::LazyLock,
};

use bitvec::prelude::*;
use image::ImageReader;
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

#[expect(unused)]
#[derive(Clone, Debug)]
pub struct IndexBufferView {
    pub buffer: IndexBufferIndex,
    pub range: Range<u32>,
}

pub struct VertexBuffer {
    pub vertices: Vec<[f32; 3]>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct VertexBufferIndex(u32);

#[expect(unused)]
pub struct TexCoordBuffer {
    pub vertices: Vec<[f32; 2]>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TexCoordBufferIndex(u32);

pub struct NormalBuffer {
    pub normals: Vec<[f32; 3]>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct NormalBufferIndex(u32);

#[expect(unused)]
pub struct TangentBuffer {
    pub normals: Vec<[f32; 3]>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TangentBufferIndex(u32);

#[expect(unused)]
#[derive(Clone, Debug)]
pub struct BlinnPhong {
    pub ambient_base: Color,
    pub diffuse_base: Color,
    pub specular_base: Color,
    pub specular_exponent: f32,
    pub diffuse_map: Option<TextureIndex>,
}

#[expect(unused)]
pub enum MaterialType {
    BlinnPhong(BlinnPhong),
    // TODO: Add Metallic Roughness type
}

#[expect(unused)]
pub struct MaterialNew {
    pub data: MaterialType,
    pub normal: Option<TextureIndex>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct MaterialNewIndex(u32);

#[expect(unused)]
#[derive(Debug, Clone, Default)]
pub struct Sampler {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub mipmap_filter: Filter,
    pub wrap_s: WrapMode,
    pub wrap_t: WrapMode,
}

#[expect(unused)]
#[derive(Debug, Clone)]
pub struct Texture {
    pub sampler: Sampler,
    pub image: ImageIndex,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TextureIndex(u32);

#[expect(unused)]
pub enum Image {
    Path(PathBuf),
    Data {
        data: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
        path: Option<PathBuf>,
    },
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ImageIndex(u32);

#[expect(unused)]
pub struct PrimitiveData {
    pub vertex: VertexBufferIndex,
    pub normal: Option<(NormalBufferIndex, TangentBufferIndex)>,
    pub tex_coords: Option<TexCoordBufferIndex>,
}

#[expect(unused)]
pub struct PrimitivesNew {
    pub data: PrimitiveData,
    pub indices: IndexBufferView,
    pub material: Option<MaterialNewIndex>,
}

#[expect(unused)]
pub struct MeshNew {
    pub primitives: Vec<PrimitivesNew>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct MeshNewIndex(u32);

pub struct Storage {
    index_buffers: Vec<IndexBuffer>,
    vertex_buffers: Vec<VertexBuffer>,
    tex_coord_buffers: Vec<TexCoordBuffer>,
    normal_buffers: Vec<NormalBuffer>,
    tangent_buffers: Vec<TangentBuffer>,
    materials: Vec<MaterialNew>,
    textures: Vec<Texture>,
    images: Vec<Image>,
    meshes: Vec<MeshNew>,
}

impl Storage {
    #[expect(unused)]
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
        }
    }

    #[expect(unused)]
    pub fn index_buffer(&self, index: IndexBufferIndex) -> &IndexBuffer {
        &self.index_buffers[index.0 as usize]
    }

    #[expect(unused)]
    pub fn vertex_buffer(&self, index: VertexBufferIndex) -> &VertexBuffer {
        &self.vertex_buffers[index.0 as usize]
    }

    #[expect(unused)]
    pub fn tex_coord_buffer(&self, index: TexCoordBufferIndex) -> &TexCoordBuffer {
        &self.tex_coord_buffers[index.0 as usize]
    }

    #[expect(unused)]
    pub fn normal_buffer(&self, index: NormalBufferIndex) -> &NormalBuffer {
        &self.normal_buffers[index.0 as usize]
    }

    #[expect(unused)]
    pub fn tangent_buffer(&self, index: TangentBufferIndex) -> &TangentBuffer {
        &self.tangent_buffers[index.0 as usize]
    }

    #[expect(unused)]
    pub fn material(&self, index: MaterialNewIndex) -> &MaterialNew {
        &self.materials[index.0 as usize]
    }

    #[expect(unused)]
    pub fn texture(&self, index: TextureIndex) -> &Texture {
        &self.textures[index.0 as usize]
    }

    #[expect(unused)]
    pub fn image(&self, index: ImageIndex) -> &Image {
        &self.images[index.0 as usize]
    }

    #[expect(unused)]
    pub fn mesh(&self, index: MeshNewIndex) -> &MeshNew {
        &self.meshes[index.0 as usize]
    }
}

impl Storage {
    #[expect(unused)]
    pub fn store_obj(
        &mut self,
        obj: parser::ParsedObj,
        mtls: impl IntoIterator<Item = parser::ParsedMtl>,
    ) -> MeshNewIndex {
        let texture_map = self.store_mtls(mtls);
        let objects = CompactedObject::separate_objects(obj);

        for mut object in objects.into_iter().flat_map(MeshCombined::new) {
            object.compute_normals();
            let MeshCombined {
                name,
                vertex_buffer: vertices,
                tex_coord_buffer: uv,
                material,
                normal_buffer_computed: normals_computed,
                normal_buffer_specified: normals_specified,
                index_buffer: triangles,
            } = object;

            let index_buffer = self.store_index_buffer(triangles);
            let vertex_buffer = self.store_vertex_buffer(vertices);
            let tex_coord_buffer = uv.map(|uv| self.store_tex_coord_buffer(uv));
            let normal_buffer_computed = self.store_normal_buffer(normals_computed.unwrap());
            let normal_buffer_specified = normals_specified.map(|n| self.store_normal_buffer(n));

            // TODO: create index buffer slices from materials (first need to keep slices in
            // MeshCombined)
            todo!()
        }
        todo!()
    }

    #[expect(unused)]
    fn store_mtls(
        &mut self,
        mtls: impl IntoIterator<Item = parser::ParsedMtl>,
    ) -> HashMap<PathBuf, TextureIndex> {
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
            return index;
        };

        for mtl in mtls {
            let mtl::ParsedMtl {
                name,
                ka,
                kd,
                ks,
                ns,
                map_bump,
                map_kd,
                ..
            } = mtl;

            let ka = ka.unwrap_or_else(|| parser::mtl::MtlKa::default());
            let kd = kd.unwrap_or_else(|| parser::mtl::MtlKd::default());
            let ks = ks.unwrap_or_else(|| parser::mtl::MtlKs::default());
            let ns = ns.unwrap_or_else(|| parser::mtl::MtlNs::default());

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
                g: kd.1,
                b: kd.2,
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
            let mtl = MaterialNew {
                data: MaterialType::BlinnPhong(mtl),
                normal: normal_map,
            };

            self.store_material(mtl);
        }

        texture_map
    }

    fn store_index_buffer(&mut self, buffer: IndexBuffer) -> IndexBufferIndex {
        let index = IndexBufferIndex(self.vertex_buffers.len() as u32);
        self.index_buffers.push(buffer);
        index
    }

    fn store_vertex_buffer(&mut self, buffer: VertexBuffer) -> VertexBufferIndex {
        let index = VertexBufferIndex(self.vertex_buffers.len() as u32);
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

    #[expect(unused)]
    fn store_tangent_buffer(&mut self, buffer: TangentBuffer) -> TangentBufferIndex {
        let index = TangentBufferIndex(self.tangent_buffers.len() as u32);
        self.tangent_buffers.push(buffer);
        index
    }

    fn store_material(&mut self, material: MaterialNew) -> MaterialNewIndex {
        let index = MaterialNewIndex(self.materials.len() as u32);
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

    #[expect(unused)]
    fn store_mesh(&mut self, mesh: MeshNew) -> MeshNewIndex {
        let index = MeshNewIndex(self.meshes.len() as u32);
        self.meshes.push(mesh);
        index
    }
}

/// GLTF Buffer
#[expect(unused)]
pub enum Buffer {
    Bytes {
        data: Vec<u8>,
        origin: Option<PathBuf>,
    },
    Path {
        path: PathBuf,
        bytes_length: u32,
    },
}

#[derive(Copy, Clone, Debug)]
pub struct BufferIndex(usize);

pub struct Buffers(Vec<Buffer>);

impl Buffers {
    #[expect(unused)]
    pub fn get(&self, index: BufferIndex) -> Option<&Buffer> {
        self.0.get(index.0)
    }
}

/// GLTF BufferView
#[expect(unused)]
pub struct BufferView {
    pub buffer: BufferIndex,
    pub byte_offset: usize,
    pub byte_length: usize,
    pub byte_stride: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct BufferViewIndex(usize);

pub struct BufferViews(Vec<BufferView>);

impl BufferViews {
    #[expect(unused)]
    pub fn get(&self, index: BufferViewIndex) -> Option<&BufferView> {
        self.0.get(index.0)
    }
}

#[expect(unused)]
pub enum CompositeType {
    Scalar,
    Vec2,
    Vec3,
    Vec4,
    Mat2,
    Mat3,
    Mat4,
}

#[expect(unused)]
pub enum ComponentType {
    I8,
    U8,
    I16,
    U16,
    U32,
    F32,
}

/// GLTF Accessor
#[expect(unused)]
pub struct Accessor {
    pub buffer_view: BufferViewIndex,
    pub byte_offset: usize,
    pub composite_type: CompositeType,
    pub component_type: ComponentType,
    pub count: usize,
}

pub struct Accessors(Vec<Accessor>);

#[derive(Copy, Clone, Debug)]
pub struct AccessorIndex(usize);

impl Accessors {
    #[expect(unused)]
    pub fn get(&self, index: AccessorIndex) -> Option<&Accessor> {
        self.0.get(index.0)
    }
}

#[expect(unused)]
pub enum TopologyType {
    Point,
    Line,
    Triangle,
}

/// GLTF attributes for mesh primitives
#[expect(unused)]
pub struct PrimitiveAttributes {
    pub position: AccessorIndex,
    pub normal: Option<AccessorIndex>,
    pub tangent: Option<AccessorIndex>,
    tex_coords: Vec<AccessorIndex>,
}

#[derive(Copy, Clone, Debug)]
pub struct TexCoordsIndex(usize);

impl PrimitiveAttributes {
    #[expect(unused)]
    pub fn get_tex_coords(&self, index: TexCoordsIndex) -> Option<AccessorIndex> {
        self.tex_coords.get(index.0).copied()
    }
}

/// GLTF Entry in "primitives" array for a mesh
#[expect(unused)]
pub struct Primitives {
    pub attributes: PrimitiveAttributes,
    pub mode: TopologyType,
    pub indices: AccessorIndex,
    pub material: Option<usize>,
}

/// GLTF Mesh
#[expect(unused)]
pub struct Mesh {
    pub primitives: Vec<Primitives>,
}

#[derive(Copy, Clone, Debug)]
pub struct MeshIndex(usize);

pub struct Meshes(Vec<Mesh>);

impl Meshes {
    #[expect(unused)]
    pub fn get(&self, index: MeshIndex) -> Option<&Mesh> {
        self.0.get(index.0)
    }
}

pub struct Images(Vec<Image>);

impl Images {
    #[expect(unused)]
    pub fn get(&self, index: ImageIndex) -> Option<&Image> {
        self.0.get(index.0 as usize)
    }
}

#[expect(unused)]
#[derive(Debug, Clone, Copy, Default)]
pub enum Filter {
    Nearest,
    #[default]
    Linear,
}

#[expect(unused)]
#[derive(Debug, Clone, Copy, Default)]
pub enum WrapMode {
    ClampToEdge,
    MirroredRepeat,
    #[default]
    Repeat,
}

pub struct Textures(Vec<Texture>);

impl Textures {
    fn from_mtl(mtls: &[parser::ParsedMtl]) -> (Images, Textures, HashMap<PathBuf, TextureIndex>) {
        let mut images = Images(Vec::new());
        let mut textures = Textures(Vec::new());
        let mut texture_map: HashMap<PathBuf, TextureIndex> = HashMap::new();

        let mut add_to_textures = |path: &Path| {
            use std::collections::hash_map::Entry;

            let entry = match texture_map.entry(path.to_path_buf()) {
                Entry::Occupied(_) => return,
                Entry::Vacant(entry) => entry,
            };

            let n_images = images.0.len();
            assert_eq!(n_images, textures.0.len());

            // mtl can not specify sampler, so images and textures have 1:1 correspondence
            let image_index = ImageIndex(n_images as u32);
            let texture_index = TextureIndex(n_images as u32);

            let image = Image::Path(path.to_path_buf());
            let texture = Texture {
                sampler: Sampler::default(),
                image: image_index,
            };

            images.0.push(image);
            textures.0.push(texture);
            entry.insert(texture_index);
        };

        for mtl in mtls.iter() {
            if let Some(mtl::MtlMapBump(path)) = &mtl.map_bump {
                add_to_textures(path);
            }
            if let Some(mtl::MtlMapKa(path)) = &mtl.map_ka {
                add_to_textures(path);
            }
            if let Some(mtl::MtlMapKd(path)) = &mtl.map_kd {
                add_to_textures(path);
            }
        }

        (images, textures, texture_map)
    }

    #[expect(unused)]
    pub fn get(&self, index: TextureIndex) -> Option<&Texture> {
        self.0.get(index.0 as usize)
    }
}

#[expect(unused)]
pub struct BaseColorTexture {
    pub index: TextureIndex,
    // TODO: I don't understand this one yet:
    // pub tex_coords: Option<AccessorIndex>,
}

#[expect(unused)]
pub struct MetallicRoughnessTexture {
    pub index: TextureIndex,
    // TODO: I don't understand this one yet:
    // pub tex_coords: Option<AccessorIndex>,
}

#[expect(unused)]
pub struct NormalTexture {
    pub index: TextureIndex,
    pub scale: f32,
    // TODO: I don't understand this one yet:
    // pub tex_coords: Option<AccessorIndex>,
}

/// GLTF Metallic Roughness Model properties
#[expect(unused)]
pub struct PbrMetallicRoughness {
    pub base_color_factors: na::Vector4<f32>,
    pub base_color_texture: Option<BaseColorTexture>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_texture: Option<MetallicRoughnessTexture>,
}

/// GLTF Material
#[expect(unused)]
pub struct Material {
    pub name: Option<String>,
    pub pbr_metallic_roughness: PbrMetallicRoughness,
    pub normal_texture: Option<NormalTexture>,
    pub emmisive_factor: na::Vector3<f32>,
}

impl Material {
    #[expect(unused)]
    fn from_mtl(mtl: parser::ParsedMtl, path_map: &HashMap<PathBuf, TextureIndex>) -> Self {
        let mtl::ParsedMtl {
            name,
            ka,
            kd,
            ks,
            ns,
            illum,
            map_ka,
            map_bump,
            map_kd,
            ..
        } = mtl;

        let base_color_factors = if let Some(mtl::MtlKd(r, g, b)) = kd {
            [r, g, b, 1.].into()
        } else if let Some(mtl::MtlKa(r, g, b)) = ka {
            [r, g, b, 1.].into()
        } else {
            [1., 1., 1., 1.].into()
        };

        let base_color_texture = map_kd
            .map(|mtl::MtlMapKd(path)| path)
            .or(map_ka.map(|mtl::MtlMapKa(path)| path))
            .map(|path| {
                let index = *path_map
                    .get(&path)
                    .expect("used path must be present in mapping");
                BaseColorTexture { index }
            });

        let roughness_factor = ns
            .map(|mtl::MtlNs(ns)| {
                if ns <= 1. {
                    return 1.;
                }

                const NS_MAX: f32 = 1000.;

                let ns = ns.clamp(1., NS_MAX);
                let ns = ns.ln();
                let ns = ns / NS_MAX.ln();
                1. - ns
            })
            .unwrap_or(1.);

        let pbr_metallic_roughness = PbrMetallicRoughness {
            base_color_factors,
            base_color_texture,
            metallic_factor: 1.,
            roughness_factor,
            metallic_roughness_texture: None,
        };

        let normal_texture = map_bump.map(|mtl::MtlMapBump(path)| {
            let index = *path_map
                .get(&path)
                .expect("used path must be present in mapping");
            NormalTexture { index, scale: 1. }
        });

        Self {
            name: Some(name),
            pbr_metallic_roughness,
            normal_texture,
            emmisive_factor: [0., 0., 0.].into(),
        }
    }
}

pub struct Materials(Vec<Material>);

#[derive(Copy, Clone, Debug)]
pub struct MaterialIndex(usize);

impl Materials {
    #[expect(unused)]
    pub fn get(&self, index: MaterialIndex) -> Option<&Material> {
        self.0.get(index.0)
    }
}

/// GLTF collection of all referenced data
#[expect(unused)]
pub struct Model {
    pub buffers: Buffers,
    pub buffer_views: BufferViews,
    pub accessors: Accessors,
    pub meshes: Meshes,
    pub materials: Materials,
    pub textures: Textures,
    pub images: Images,
}

impl Model {
    #[expect(unused)]
    pub fn new(obj: parser::ParsedObj, mtl: Vec<parser::ParsedMtl>) -> Self {
        let (images, textures, texture_map) = Textures::from_mtl(&mtl);

        let mtl = mtl
            .into_iter()
            .map(|mtl| Material::from_mtl(mtl, &texture_map))
            .collect();
        let mtl = Materials(mtl);

        let objects = CompactedObject::separate_objects(obj);

        // TODO: make vertex buffer interleaved with uv coordinates
        let mut buffers = Buffers(Vec::new());
        for obj in objects {
            let CompactedObject {
                name,
                faces,
                groups,
                mtls,
                geo_vertices,
                tex_vertices,
                vertex_normals,
            } = obj;

            assert!(tex_vertices.len() == 0 || geo_vertices.len() == tex_vertices.len());
            assert!(vertex_normals.len() == 0 || geo_vertices.len() == vertex_normals.len());

            todo!()
        }

        todo!()
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshIndexOld {
    index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialIndexOld {
    index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexIndex {
    index: u32,
}

pub struct ModelStorage {
    meshes: Vec<MeshCombined>,
    materials: Vec<MaterialOld>,
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
pub struct NormalTextureOld(pub image::ImageBuffer<image::Rgba<u8>, Vec<u8>>);

#[derive(Clone, Debug)]
pub struct BlinnPhongOld {
    pub ambient_base: Color,
    pub diffuse_color: Color,
    pub specular_color: Color,
    pub specular_exponent: f32,
    pub diffuse_map: Option<ColorTexture>,
}

#[derive(Debug, Clone)]
pub struct MaterialOld {
    pub name: String,
    pub phong_params: BlinnPhongOld,
    pub normal_texture: Option<NormalTextureOld>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Triangle(VertexIndex, VertexIndex, VertexIndex);

pub struct MeshCombined {
    pub name: Option<String>,
    pub vertex_buffer: VertexBuffer,
    pub tex_coord_buffer: Option<TexCoordBuffer>,
    pub material: Option<MaterialIndexOld>,
    pub normal_buffer_computed: Option<NormalBuffer>,
    pub normal_buffer_specified: Option<NormalBuffer>,
    pub index_buffer: IndexBuffer,
}

pub struct ModelOld {
    pub meshes: Vec<MeshCombined>,
}

impl ModelStorage {
    pub fn new() -> Self {
        Self {
            meshes: Vec::new(),
            materials: Vec::new(),
        }
    }

    pub fn get_mesh(&self, MeshIndexOld { index }: MeshIndexOld) -> &MeshCombined {
        &self.meshes[index as usize]
    }

    pub fn get_material(&self, index: Option<MaterialIndexOld>) -> &MaterialOld {
        static DEFAULT: LazyLock<MaterialOld> = LazyLock::new(|| MaterialOld {
            name: String::from("Default"),
            phong_params: Default::default(),
            normal_texture: None,
        });

        if let Some(MaterialIndexOld { index }) = index {
            &self.materials[index as usize]
        } else {
            &DEFAULT
        }
    }

    pub fn load_meshes(&mut self, path: &Path) -> anyhow::Result<Vec<MeshIndexOld>> {
        let mut inserter = ModelStorageInserter {
            root_dir: path.parent().expect("file should have a parent directory"),
            storage: self,
        };

        let loaded_obj = obj::load_obj(path, &mut inserter)?;
        let model = ModelOld::new(loaded_obj);

        let first_new_index = self.meshes.len() as u32;
        let new_indices = (0..model.meshes.len() as u32)
            .map(|v| MeshIndexOld {
                index: v + first_new_index,
            })
            .collect();

        self.meshes.extend(model.meshes);
        Ok(new_indices)
    }
}

impl<'a> parser::MtlManager for ModelStorageInserter<'a> {
    fn request_mtl_load(
        &mut self,
        filename: &str,
    ) -> Result<HashSet<String>, parser::mtl::MtlError> {
        let path = self.root_dir.join(filename);
        let loaded_mtls = parser::mtl::parse_mtl(&path)?;
        let mut mtls_in_this_lib = HashSet::new();

        for mtl in loaded_mtls {
            assert!(!mtls_in_this_lib.contains(&mtl.name), "duplicate material");
            mtls_in_this_lib.insert(mtl.name.clone());

            let already_loaded = self.storage.materials.iter().any(|m| m.name == mtl.name);
            if already_loaded {
                continue;
            }
            let new_material = MaterialOld::load(self.root_dir, &mtl).unwrap();
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

impl MaterialOld {
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
                Some(NormalTextureOld(image.into()))
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

        let phong_params = BlinnPhongOld {
            ambient_base: ambient_color,
            diffuse_color,
            diffuse_map: color_texture,
            specular_color,
            specular_exponent,
        };

        Ok(Self {
            name: mtl.name.to_string(),
            phong_params,
            normal_texture,
        })
    }
}

#[derive(Debug, Clone)]
struct CompactedObject {
    pub name: Option<String>,
    pub faces: Vec<obj::F>,
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
            groups,
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
            groups: _,
            mtls,
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
        let mut has_tex = mtls.len() == 1; // only consider meshes with single material

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
            vec.resize_with(n_vertices, || na::Vector3::zeros());
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

        let new_vertices = VertexBuffer {
            vertices: new_vertices,
        };
        let new_tex = new_tex.map(|vertices| TexCoordBuffer { vertices });

        // make triangles out of polygons naively
        let triangles = faces
            .into_iter()
            .flat_map(|face| {
                assert!(face.triplets.len() >= 3, "too few vertices, not a face!");

                let first = *face.triplets.first().expect("checked above");
                let first = triplet_to_index[&first];
                face.triplets[1..]
                    .windows(2)
                    .flat_map(<&[obj::FTriplet; 2]>::try_from)
                    .map(|[second, third]| {
                        let second = triplet_to_index[second];
                        let third = triplet_to_index[third];

                        [first, second, third]
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
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

        let material = has_tex
            .then(|| {
                assert!(mtls.len() == 1);
                let (index, face_ranges) = mtls.into_iter().next().unwrap();
                assert!(face_ranges.slices.len() == 1);
                let Range { start, end } = face_ranges.slices.into_iter().next().unwrap();
                assert!(start == 0 && end as usize == n_vertices);
                index.map(|index| MaterialIndexOld { index })
            })
            .flatten();

        Some(Self {
            name,
            vertex_buffer: new_vertices,
            tex_coord_buffer: new_tex,
            material,
            normal_buffer_computed: None,
            normal_buffer_specified: new_normals,
            index_buffer: triangles,
        })
    }

    fn compute_normals(&mut self) {
        if self.normal_buffer_computed.is_some() {
            return;
        }

        let mut normals_computed = Vec::new();
        normals_computed.resize_with(self.vertex_buffer.vertices.len(), na::Vector3::zeros);

        for &[i0, i1, i2] in self.index_buffer.triangles.iter() {
            let p0 = &self.vertex_buffer.vertices[i0 as usize];
            let p1 = &self.vertex_buffer.vertices[i1 as usize];
            let p2 = &self.vertex_buffer.vertices[i2 as usize];

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

        let normals_computed = normals_computed
            .into_iter()
            .map(|n| {
                let n = na::Unit::new_normalize(n);
                [n[0], n[1], n[2]]
            })
            .collect();

        self.normal_buffer_computed = Some(NormalBuffer {
            normals: normals_computed,
        });
    }
}

impl ModelOld {
    fn new(parsed_obj: obj::ParsedObj) -> Self {
        let meshes = CompactedObject::separate_objects(parsed_obj)
            .into_iter()
            .flat_map(MeshCombined::new)
            .map(|mut m| {
                m.compute_normals();
                m
            })
            .collect();

        Self { meshes }
    }
}

impl Default for BlinnPhongOld {
    fn default() -> Self {
        Self {
            ambient_base: Color::from(mtl::MtlKa::default()),
            diffuse_color: Color::from(mtl::MtlKd::default()),
            diffuse_map: None,
            specular_color: Color::from(mtl::MtlKs::default()),
            specular_exponent: mtl::MtlNs::default().0,
        }
    }
}

impl render::GpuTransfer for MeshCombined {
    type Raw = render::TriangleBufferRaw;

    fn to_raw(&self) -> Self::Raw {
        assert!(self.vertex_buffer.vertices.len() <= <u32>::MAX as usize);
        let indices = self
            .index_buffer
            .triangles
            .iter()
            .flat_map(|&[i, j, k]| [i, j, k])
            .collect();
        // let normals = if !self.normals_specified.is_empty() {
        //     &self.normals_specified
        // } else {
        //     &self.normals_computed
        // };
        let normal_buffer = self.normal_buffer_computed.as_ref().unwrap();
        assert!(normal_buffer.normals.len() == self.vertex_buffer.vertices.len());

        let vertices = self
            .vertex_buffer
            .vertices
            .iter()
            .zip(normal_buffer.normals.iter())
            .map(|(v, n)| {
                let n = na::Vector3::from_column_slice(n);
                let n = na::Unit::new_normalize(n);
                let basis = crate::math::orthonormal_basis_for_normal(&n);
                render::VertexRaw {
                    position: *v,
                    tex_coords: [0., 0.],
                    normal: basis.column(0).into(),
                    tangent: basis.column(1).into(),
                    bitangent: basis.column(2).into(),
                }
            })
            .collect();
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

impl<'a> render::GpuTransferRef<'a> for NormalTextureOld {
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
        let normal = *self.rotation.to_rotation_matrix().matrix();
        let normal = normal.into();
        render::InstanceRaw { model, normal }
    }
}

impl From<Triangle> for (u32, u32, u32) {
    fn from(Triangle(i, j, k): Triangle) -> Self {
        (i.index, j.index, k.index)
    }
}

impl render::GpuTransfer for BlinnPhongOld {
    type Raw = render::PhongRaw;

    fn to_raw(&self) -> Self::Raw {
        render::PhongRaw {
            specular_color: self.specular_color.into(),
            specular_exponent: self.specular_exponent,
            diffuse_color: self.diffuse_color.into(),
            _padding1: 0,
            ambient_color: self.ambient_base.into(),
            _padding2: 0,
        }
    }
}

impl Deref for MeshIndexOld {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}

impl Deref for MaterialIndexOld {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.index
    }
}
