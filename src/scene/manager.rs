use std::{
    cell::RefCell,
    collections::{HashMap, hash_map},
    ops::Range,
    rc::Rc,
};

use crate::{model, render};

#[derive(Debug)]
pub struct IndexInfo {
    pub index: render::IndexBufferIndex,
    pub slice: Range<u32>,
}

#[derive(Debug)]
pub struct PrimitiveData {
    pub position: render::PositionBufferIndex,
    pub tex_coord: Option<render::TexCoordBufferIndex>,
    pub normal: Option<render::NormalBufferIndex>,
    pub tangent: Option<render::TangentBufferIndex>,
}

#[derive(Debug)]
pub struct PrimitiveInfo {
    pub data: PrimitiveData,
    pub indices: IndexInfo,
    pub material: render::MaterialBindingIndex,
}

#[derive(Debug, Copy, Clone, Default)]
pub enum PipelineType {
    #[default]
    Render,
    Light,
}

#[derive(Debug)]
pub struct MeshInfo {
    pub primitives: Vec<PrimitiveInfo>,
    pub instance: render::InstanceBufferIndex,
    pub draw_type: PipelineType,
}

pub struct DrawManager {
    render_state: Rc<RefCell<render::RenderState>>,
    meshes: HashMap<model::MeshIndex, MeshInfo>,

    // translations between model asset and render resource handles
    map_index: HashMap<model::IndexBufferIndex, render::IndexBufferIndex>,
    map_position: HashMap<model::PositionBufferIndex, render::PositionBufferIndex>,
    map_tex_coord: HashMap<model::TexCoordBufferIndex, render::TexCoordBufferIndex>,
    map_normal: HashMap<model::NormalBufferIndex, render::NormalBufferIndex>,
    map_tangent: HashMap<model::TangentBufferIndex, render::TangentBufferIndex>,
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
            map_material: HashMap::new(),
        }
    }

    pub fn add_mesh(
        &mut self,
        storage: &model::Storage,
        mesh_index: model::MeshIndex,
        instance: render::InstanceBufferIndex,
        draw_type: PipelineType,
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
                    position: vertex,
                    normal,
                    tex_coord,
                } = data;

                let position = match self.map_position.entry(*vertex) {
                    hash_map::Entry::Occupied(entry) => *entry.get(),
                    hash_map::Entry::Vacant(entry) => {
                        let data = &storage.position_buffer(*entry.key()).positions;
                        let data = bytemuck::cast_slice(data);
                        let index = render_state.upload_position_buffer(data, label);
                        *entry.insert(index)
                    }
                };

                let tex_coord =
                    tex_coord.map(|tex_coord| match self.map_tex_coord.entry(tex_coord) {
                        hash_map::Entry::Occupied(entry) => *entry.get(),
                        hash_map::Entry::Vacant(entry) => {
                            let data = &storage.tex_coord_buffer(*entry.key()).tex_coords;
                            let data = bytemuck::cast_slice(data);
                            let index = render_state.upload_tex_coord_buffer(data, label);
                            *entry.insert(index)
                        }
                    });

                let (normal, tangent) = match *normal {
                    None => (None, None),
                    Some((normal_index, tangent_index)) => {
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
                        (Some(normal), Some(tangent))
                    }
                };

                let data = PrimitiveData {
                    position,
                    tex_coord,
                    normal,
                    tangent,
                };

                let index = match self.map_index.entry(indices.buffer) {
                    hash_map::Entry::Occupied(entry) => *entry.get(),
                    hash_map::Entry::Vacant(entry) => {
                        let data = &storage.index_buffer(*entry.key()).triangles;
                        let index = render_state.upload_index_buffer(data, label);
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
                        let diffuse_map = phong_params
                            .diffuse_map
                            .map(|i| storage.texture(i).to_raw(storage));

                        let normal_texture =
                            data.normal.map(|i| storage.texture(i).to_raw(storage));

                        let index = render_state.add_material(
                            &phong_params.to_raw(),
                            diffuse_map.as_ref(),
                            normal_texture.as_ref(),
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
            draw_type,
        };
        self.meshes.insert(mesh_index, mesh_info);
    }

    pub fn create_draw(&self, caps: render::PhongCapabilites) -> render::DrawWorld {
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
                    } = prim.data;

                    let draw_type = match mesh.draw_type {
                        PipelineType::Render => render::DrawType::Render(caps),
                        PipelineType::Light => render::DrawType::Light,
                    };

                    render::DrawCall {
                        material,
                        draw_type,
                        instance,
                        index,
                        slice,
                        position,
                        tex_coord,
                        normal,
                        tangent,
                    }
                })
            })
            .collect();

        render::DrawWorld { draw_calls }
    }
}
