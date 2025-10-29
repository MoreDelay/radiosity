use std::{
    cell::RefCell,
    collections::{HashMap, hash_map},
    ops::Range,
    rc::Rc,
};

use nalgebra as na;

use crate::{model, render};

#[derive(Debug)]
pub struct IndexInfo {
    pub index: render::IndexBufferIndex,
    pub slice: Range<u32>,
}

#[derive(Debug)]
pub struct PrimitiveData {
    pub position: render::PositionBufferIndex,
    pub tex_coord: render::TexCoordBufferIndex,
    pub normal: render::NormalBufferIndex,
    pub tangent: render::TangentBufferIndex,
    pub bi_tangent: render::BiTangentBufferIndex,
}

#[derive(Debug)]
pub struct PrimitiveInfo {
    pub data: PrimitiveData,
    pub indices: IndexInfo,
    pub material: render::MaterialBindingIndex,
}

#[derive(Debug)]
pub struct MeshInfo {
    pub primitives: Vec<PrimitiveInfo>,
    pub instance: render::InstanceBufferIndex,
}

pub struct DrawManager {
    render_state: Rc<RefCell<render::RenderState>>,
    meshes: HashMap<model::MeshIndex, MeshInfo>,

    // translations between model asset and render resource handles
    map_index: HashMap<model::IndexBufferIndex, render::IndexBufferIndex>,
    map_position: HashMap<model::VertexBufferIndex, render::PositionBufferIndex>,
    map_tex_coord: HashMap<model::TexCoordBufferIndex, render::TexCoordBufferIndex>,
    map_normal: HashMap<model::NormalBufferIndex, render::NormalBufferIndex>,
    map_tangent: HashMap<model::TangentBufferIndex, render::TangentBufferIndex>,
    map_bi_tangent: HashMap<
        (model::NormalBufferIndex, model::TangentBufferIndex),
        render::BiTangentBufferIndex,
    >,
    map_material: HashMap<model::MaterialIndex, render::MaterialBindingIndex>,
}

impl DrawManager {
    pub fn new(render_state: Rc<RefCell<render::RenderState>>) -> Self {
        Self {
            render_state,
            meshes: HashMap::new(),
            map_index: HashMap::new(),
            map_position: HashMap::new(),
            map_tex_coord: HashMap::new(),
            map_normal: HashMap::new(),
            map_tangent: HashMap::new(),
            map_bi_tangent: HashMap::new(),
            map_material: HashMap::new(),
        }
    }

    pub fn add_mesh(
        &mut self,
        storage: &model::Storage,
        mesh_index: model::MeshIndex,
        instance: render::InstanceBufferIndex,
        label: Option<&str>,
    ) {
        let mut render_state = self.render_state.borrow_mut();

        let mesh = storage.mesh(mesh_index);

        let primitives = mesh
            .primitives
            .iter()
            .map(|p| {
                let model::Primitive {
                    data,
                    indices,
                    material,
                } = p;

                let model::PrimitiveData {
                    vertex,
                    normal,
                    tex_coord,
                } = data;

                let position = match self.map_position.entry(*vertex) {
                    hash_map::Entry::Occupied(entry) => *entry.get(),
                    hash_map::Entry::Vacant(entry) => {
                        let data = &storage.vertex_buffer(*entry.key()).vertices;
                        let data = bytemuck::cast_slice(data);
                        let index = render_state.upload_position_buffer(data, label);
                        *entry.insert(index)
                    }
                };

                let tex_coord = tex_coord.unwrap();
                let tex_coord = match self.map_tex_coord.entry(tex_coord) {
                    hash_map::Entry::Occupied(entry) => *entry.get(),
                    hash_map::Entry::Vacant(entry) => {
                        let data = &storage.tex_coord_buffer(*entry.key()).vertices;
                        let data = bytemuck::cast_slice(data);
                        let index = render_state.upload_tex_coord_buffer(data, label);
                        *entry.insert(index)
                    }
                };

                let (normal_index, tangent_index) = normal.unwrap();
                let normal = match self.map_normal.entry(normal_index) {
                    hash_map::Entry::Occupied(entry) => *entry.get(),
                    hash_map::Entry::Vacant(entry) => {
                        let data = &storage.normal_buffer(*entry.key()).normals;
                        let data = bytemuck::cast_slice(data);
                        let index = render_state.upload_normal_buffer(data, label);
                        *entry.insert(index)
                    }
                };
                let tangent = match self.map_tangent.entry(tangent_index) {
                    hash_map::Entry::Occupied(entry) => *entry.get(),
                    hash_map::Entry::Vacant(entry) => {
                        let data = &storage.tangent_buffer(*entry.key()).tangents;
                        let data = bytemuck::cast_slice(data);
                        let index = render_state.upload_tangent_buffer(data, label);
                        *entry.insert(index)
                    }
                };
                let bi_tangent = match self.map_bi_tangent.entry((normal_index, tangent_index)) {
                    hash_map::Entry::Occupied(entry) => *entry.get(),
                    hash_map::Entry::Vacant(entry) => {
                        let (n, t) = *entry.key();
                        let n = storage.normal_buffer(n);
                        let t = storage.tangent_buffer(t);
                        let data = n
                            .normals
                            .iter()
                            .zip(t.tangents.iter())
                            .map(|(n, t)| {
                                let n = na::Vector3::from_column_slice(n);
                                let t = na::Vector3::from_column_slice(t);
                                let b = na::Unit::new_normalize(n.cross(&t));
                                [b[0], b[1], b[2]]
                            })
                            .collect::<Vec<_>>();
                        let data = bytemuck::cast_slice(&data);
                        let index = render_state.upload_bi_tangent_buffer(data, label);
                        *entry.insert(index)
                    }
                };

                let data = PrimitiveData {
                    position,
                    tex_coord,
                    normal,
                    tangent,
                    bi_tangent,
                };

                let index = match self.map_index.entry(indices.buffer) {
                    hash_map::Entry::Occupied(entry) => *entry.get(),
                    hash_map::Entry::Vacant(entry) => {
                        let data = storage
                            .index_buffer(*entry.key())
                            .triangles
                            .iter()
                            .flatten()
                            .copied()
                            .collect::<Vec<_>>();
                        let index = render_state.upload_index_buffer(&data, label);
                        *entry.insert(index)
                    }
                };

                let Range { start, end } = indices.range;
                let slice = start * 3..end * 3;

                let indices = IndexInfo { index, slice };

                let material = match self.map_material.entry(*material) {
                    hash_map::Entry::Occupied(entry) => *entry.get(),
                    hash_map::Entry::Vacant(entry) => {
                        let data = storage.material(*entry.key());
                        let model::MaterialType::BlinnPhong(phong_params) = &data.data;
                        let diffuse_map = phong_params.diffuse_map.map(|i| {
                            let texture = storage.texture(i);
                            storage.image(texture.image)
                        });

                        let normal_texture = data.normal.map(|i| {
                            let texture = storage.texture(i);
                            storage.image(texture.image)
                        });

                        let index = render_state.add_material(
                            &phong_params.to_raw(),
                            diffuse_map,
                            normal_texture,
                            label,
                        );
                        *entry.insert(index)
                    }
                };

                PrimitiveInfo {
                    data,
                    indices,
                    material,
                }
            })
            .collect();

        let mesh_info = MeshInfo {
            primitives,
            instance,
        };
        self.meshes.insert(mesh_index, mesh_info);
    }

    pub fn create_draw(&self, mode: render::PipelineMode) -> render::DrawWorld {
        let caps_filter = match mode {
            render::PipelineMode::Flat => render::PhongCapabilites {
                color_map: false,
                normal_map: false,
            },
            render::PipelineMode::Color => render::PhongCapabilites {
                color_map: true,
                normal_map: false,
            },
            render::PipelineMode::Normal => render::PhongCapabilites {
                color_map: false,
                normal_map: true,
            },
            render::PipelineMode::ColorNormal => render::PhongCapabilites {
                color_map: true,
                normal_map: true,
            },
        };

        let draw_calls = self
            .meshes
            .values()
            .flat_map(|mesh| {
                let instance = mesh.instance;
                mesh.primitives.iter().map(move |prim| {
                    let material = prim.material;
                    let IndexInfo { index, ref slice } = prim.indices;
                    let slice = slice.clone();

                    let PrimitiveData {
                        position,
                        tex_coord,
                        normal,
                        tangent,
                        bi_tangent,
                    } = prim.data;

                    render::DrawCall {
                        material,
                        caps_filter,
                        instance,
                        index,
                        slice,
                        position,
                        tex_coord,
                        normal,
                        tangent,
                        bi_tangent,
                    }
                })
            })
            .collect();

        render::DrawWorld { draw_calls }
    }
}
