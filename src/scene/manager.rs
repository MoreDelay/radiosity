use std::{
    cell::RefCell,
    collections::{HashMap, hash_map},
    ops::Range,
    rc::Rc,
};

use crate::{model, render};

#[derive(Debug)]
struct FaceIndexSlice(Range<u32>);

#[derive(Debug, Clone, Copy)]
struct MeshInfo {
    buffer_index: render::MeshBufferIndex,
    instance_index: Option<render::InstanceBufferIndex>,
    visible: bool,
    pipeline_mode: render::PipelineMode,
}

#[derive(Debug)]
struct MaterialSubscription {
    slice: FaceIndexSlice,
    mesh_index: model::MeshIndexOld,
}

#[derive(Debug)]
struct MaterialInfo {
    binding_index: render::MaterialBindingIndex,
    subscribed_meshes: Vec<MaterialSubscription>,
}

pub struct DrawManager {
    render_state: Rc<RefCell<render::RenderState>>,
    materials: HashMap<Option<model::MaterialIndexOld>, MaterialInfo>,
    meshes: HashMap<model::MeshIndexOld, MeshInfo>,
}

impl DrawManager {
    pub fn new(render_state: Rc<RefCell<render::RenderState>>) -> Self {
        Self {
            render_state,
            materials: HashMap::new(),
            meshes: HashMap::new(),
        }
    }

    pub fn draw_iter(&self) -> DrawIterator<'_> {
        let iterator = self.materials.iter();
        DrawIterator {
            manager: self,
            iterator,
        }
    }

    pub fn set_pipeline(&mut self, pipeline: render::PipelineMode, index: model::MeshIndexOld) {
        let mesh_info = self
            .meshes
            .get_mut(&index)
            .expect("only set pipeline for drawn meshes");
        mesh_info.pipeline_mode = pipeline;
    }

    fn add_material(
        &mut self,
        storage: &model::ModelStorage,
        material_index: Option<model::MaterialIndexOld>,
        label: Option<&str>,
    ) {
        if self.materials.contains_key(&material_index) {
            return;
        }

        let material = storage.get_material(material_index);
        let binding_index = self.render_state.borrow_mut().add_material(material, label);
        let material_info = MaterialInfo {
            binding_index,
            subscribed_meshes: Vec::new(),
        };
        self.materials.insert(material_index, material_info);
    }

    pub fn add_mesh(
        &mut self,
        storage: &model::ModelStorage,
        mesh_index: model::MeshIndexOld,
        instance_index: Option<render::InstanceBufferIndex>,
        label: Option<&str>,
    ) {
        let mesh = storage.get_mesh(mesh_index);
        assert!(!mesh.index_buffer.triangles.is_empty());
        let buffer_index = self.render_state.borrow_mut().add_mesh_buffer(mesh, label);

        let mesh_info = MeshInfo {
            buffer_index,
            instance_index,
            visible: true,
            pipeline_mode: render::PipelineMode::Flat,
        };
        self.meshes.insert(mesh_index, mesh_info);

        self.add_material(storage, mesh.material, label);

        let slice = 0..mesh.index_buffer.triangles.len() as u32;
        let slice = FaceIndexSlice(slice);
        let subscription = MaterialSubscription { slice, mesh_index };
        self.materials
            .get_mut(&mesh.material)
            .expect("made sure it exists before")
            .subscribed_meshes
            .push(subscription);
    }

    pub fn get_buffer_index(&self, index: model::MeshIndexOld) -> Option<render::MeshBufferIndex> {
        let mesh_info = self.meshes.get(&index);
        mesh_info.map(|info| info.buffer_index)
    }
}

#[derive(Clone)]
pub struct DrawIterator<'a> {
    manager: &'a DrawManager,
    iterator: hash_map::Iter<'a, Option<model::MaterialIndexOld>, MaterialInfo>,
}

#[derive(Clone)]
pub struct DrawMaterialIterator<'a> {
    manager: &'a DrawManager,
    material_info: &'a MaterialInfo,
    iterator: std::slice::Iter<'a, MaterialSubscription>,
}

impl<'a> Iterator for DrawIterator<'a> {
    type Item = DrawMaterialIterator<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let (_, material_info) = self.iterator.next()?;
        let iterator = material_info.subscribed_meshes.iter();

        Some(DrawMaterialIterator {
            manager: self.manager,
            material_info,
            iterator,
        })
    }
}

impl<'a> Iterator for DrawMaterialIterator<'a> {
    type Item = render::DrawSlice;

    fn next(&mut self) -> Option<Self::Item> {
        for subscription in self.iterator.by_ref() {
            let MaterialSubscription { slice, mesh_index } = subscription;
            let mesh_info = self
                .manager
                .meshes
                .get(mesh_index)
                .expect("subscribed mesh should be actual datastructure");

            if !mesh_info.visible {
                continue;
            }

            let &MeshInfo {
                buffer_index,
                instance_index,
                visible: _,
                pipeline_mode,
            } = mesh_info;

            let slice = slice.0.clone();
            let slice = slice.start * 3..slice.end * 3;
            let draw_slice = render::DrawSlice {
                buffer_index,
                slice,
                instance_index,
                pipeline_mode,
            };
            return Some(draw_slice);
        }
        None
    }
}

impl<'a> render::DrawMaterial for DrawMaterialIterator<'a> {
    fn get_index(&self) -> render::MaterialBindingIndex {
        self.material_info.binding_index
    }
}
