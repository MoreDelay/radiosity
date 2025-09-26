use std::{
    cell::RefCell,
    collections::{HashMap, hash_map},
    ops::Range,
    rc::Rc,
};

use crate::{model, render};

#[derive(Debug)]
struct FaceIndexSlice(Range<u32>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MeshInfoIndex(usize);

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
    mesh_index: model::MeshIndex,
}

#[derive(Debug)]
struct MaterialInfo {
    binding_index: render::MaterialBindingIndex,
    subscribed_meshes: HashMap<MeshInfoIndex, MaterialSubscription>,
}

pub struct DrawManager {
    render_state: Rc<RefCell<render::RenderState>>,
    materials: HashMap<Option<model::MaterialIndex>, MaterialInfo>,
    meshes: HashMap<model::MeshIndex, MeshInfo>,
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

    pub fn set_pipeline(&mut self, pipeline: render::PipelineMode, index: model::MeshIndex) {
        let mesh_info = self
            .meshes
            .get_mut(&index)
            .expect("only set pipeline for drawn meshes");
        mesh_info.pipeline_mode = pipeline;
    }

    fn add_material(
        &mut self,
        storage: &model::ModelStorage,
        material_index: Option<model::MaterialIndex>,
        label: Option<&str>,
    ) {
        if self.materials.contains_key(&material_index) {
            return;
        }

        let material = storage.get_material(material_index);
        let binding_index = self.render_state.borrow_mut().add_material(material, label);
        let material_info = MaterialInfo {
            binding_index,
            subscribed_meshes: HashMap::new(),
        };
        self.materials.insert(material_index, material_info);
    }

    pub fn add_mesh(
        &mut self,
        storage: &model::ModelStorage,
        mesh_index: model::MeshIndex,
        instance_index: Option<render::InstanceBufferIndex>,
        label: Option<&str>,
    ) {
        let mesh = storage.get_mesh(mesh_index);
        assert!(!mesh.triangles.is_empty());
        let buffer_index = self.render_state.borrow_mut().add_mesh_buffer(mesh, label);

        let mesh_info_index = self.meshes.len();
        let mesh_info_index = MeshInfoIndex(mesh_info_index);
        let mesh_info = MeshInfo {
            buffer_index,
            instance_index,
            visible: true,
            pipeline_mode: render::PipelineMode::Flat,
        };
        self.meshes.insert(mesh_index, mesh_info);

        for (&mtl_index, model::MaterialRanges { ranges }) in mesh.mtl_ranges.iter() {
            let label = label.map(|l| format!("{l}-Material"));
            self.add_material(storage, mtl_index, label.as_deref());

            for range in ranges.iter() {
                let start = range.start;
                let end = range.end;
                let slice = FaceIndexSlice(start * 3..end * 3);
                let subscription = MaterialSubscription { slice, mesh_index };
                let material_info = self
                    .materials
                    .get_mut(&mtl_index)
                    .expect("made sure it exists before");
                material_info
                    .subscribed_meshes
                    .insert(mesh_info_index, subscription);
            }
        }
    }

    pub fn get_buffer_index(&self, index: model::MeshIndex) -> Option<render::MeshBufferIndex> {
        let mesh_info = self.meshes.get(&index);
        mesh_info.map(|info| info.buffer_index)
    }
}

#[derive(Clone)]
pub struct DrawIterator<'a> {
    manager: &'a DrawManager,
    iterator: hash_map::Iter<'a, Option<model::MaterialIndex>, MaterialInfo>,
}

#[derive(Clone)]
pub struct DrawMaterialIterator<'a> {
    manager: &'a DrawManager,
    material_info: &'a MaterialInfo,
    iterator: hash_map::Iter<'a, MeshInfoIndex, MaterialSubscription>,
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
        for (_, subscription) in self.iterator.by_ref() {
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
