use std::{
    ops::{Deref, Range},
    sync::Arc,
};

use anyhow::Result;
use winit::window::Window;

use crate::{camera, model, render::resource::ModelResourceStorage};

mod layout;
mod pipeline;
mod resource;

pub use layout::*;
pub use resource::{InstanceBufferIndex, MaterialBindingIndex, MeshBufferIndex};

#[derive(Copy, Clone, Debug)]
pub enum PipelineMode {
    Flat,
    Color,
    Normal,
}

pub struct RenderStateInit {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    depth_texture: resource::Texture,
}

pub struct RenderState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    phong_layout: resource::PhongBindGroupLayout,
    texture_layouts: resource::TextureBindGroupLayouts,
    texture_pipelines: pipeline::TexturePipelines,
    #[expect(unused)]
    mesh_buffers: Vec<resource::MeshBuffer>,
    #[expect(unused)]
    material_bindings: Vec<resource::MaterialBinding>,
    #[expect(unused)]
    instance_buffers: Vec<resource::InstanceBuffer>,
    depth_texture: resource::Texture,
    camera_binding: resource::CameraBinding,
    light_binding: resource::LightBinding,
    light_pipeline: pipeline::LightPipeline,
    model_resource_storage: ModelResourceStorage,
}

pub trait DrawMaterial: Iterator<Item = DrawSlice> {
    fn get_index(&self) -> MaterialBindingIndex;
}

pub struct DrawSlice {
    pub buffer_index: MeshBufferIndex,
    pub slice: Range<u32>,
    pub instance_index: Option<InstanceBufferIndex>,
    pub pipeline_mode: PipelineMode,
}

impl RenderStateInit {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        // handle to create adapters and surfaces
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        // part of the window we can draw to
        let surface = instance.create_surface(window.clone()).unwrap();

        // handle to graphics card
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false, // always run on hardware or fail
            })
            .await
            .expect("no hardware available to render");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                // list available features with adapter.features() or device.features()
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: None,
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            // assume srgb surface here
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, // render surface to screen
            format: surface_format,                        // data format
            width: size.width,
            height: size.height,
            // sync strategy with display, always has at least Fifo
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![], // texture formats made available to create
            desired_maximum_frame_latency: 2,
        };

        let depth_texture =
            resource::Texture::create_depth_texture(&device, &config, "depth_texture");
        Self {
            surface,
            device,
            queue,
            config,
            size,
            depth_texture,
        }
    }

    pub fn get_frame_dim(&self) -> camera::FrameDim {
        (self.config.width, self.config.height).into()
    }
}

impl RenderState {
    #[allow(clippy::too_many_arguments)]
    pub fn create<C, L>(
        init: RenderStateInit,
        camera: &C,
        light: &L,
        // phong: &P,
        // mesh: &M,
        // instances: &I,
        // color_texture: Option<&T>,
        // normal_texture: Option<&N>,
    ) -> anyhow::Result<Self>
    where
        C: GpuTransfer<Raw = CameraRaw>,
        L: GpuTransfer<Raw = LightRaw>,
        // P: GpuTransfer<Raw = PhongRaw>,
        // M: GpuTransfer<Raw = TriangleBufferRaw>,
        // I: GpuTransfer<Raw = InstanceBufferRaw>,
        // T: GpuTransferTexture,
        // N: GpuTransferTexture,
    {
        let RenderStateInit {
            surface,
            device,
            queue,
            config,
            size,
            depth_texture,
        } = init;

        let texture_layouts = resource::TextureBindGroupLayouts::new(&device);

        let camera_layout = resource::CameraBindGroupLayout::new(&device);
        let camera_binding =
            resource::CameraBinding::new(&device, &camera_layout, camera, Some("Single"));

        let light_layout = resource::LightBindGroupLayout::new(&device);
        let light_binding =
            resource::LightBinding::new(&device, &light_layout, light, Some("Single"));

        let phong_layout = resource::PhongBindGroupLayout::new(&device);

        let texture_pipelines = pipeline::TexturePipelines::new(
            &device,
            config.format,
            &texture_layouts,
            &camera_layout,
            &light_layout,
            &phong_layout,
        )?;

        let light_pipeline =
            pipeline::LightPipeline::new(&device, config.format, &camera_layout, &light_layout)?;

        // let material_binding = resource::MaterialBinding::new(
        //     &device,
        //     &queue,
        //     &texture_layout,
        //     // color_texture,
        //     // normal_texture,
        //     Some("Cube"),
        // );

        // let mesh_buffer = resource::MeshBuffer::new(&device, mesh, Some("Single"));
        // let instance_buffer = resource::InstanceBuffer::new(&device, instances, Some("Grid"));

        let model_resource_storage = ModelResourceStorage::new();

        let mesh_buffers = Vec::new();
        let material_bindings = Vec::new();
        let instance_buffers = Vec::new();

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            depth_texture,
            phong_layout,
            camera_binding,
            light_binding,
            texture_layouts,
            texture_pipelines,
            light_pipeline,
            model_resource_storage,
            mesh_buffers,
            material_bindings,
            instance_buffers,
        })
    }

    pub fn resize(&mut self, new_size: Option<winit::dpi::PhysicalSize<u32>>) {
        let new_size = new_size.unwrap_or(self.size);
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            // resize depth texture
            self.depth_texture = resource::Texture::create_depth_texture(
                &self.device,
                &self.config,
                "depth_texture",
            );
        }
    }

    pub fn update_light<L: GpuTransfer<Raw = LightRaw>>(&self, data: &L) {
        self.light_binding.update(&self.queue, data);
    }

    pub fn update_camera<C: GpuTransfer<Raw = CameraRaw>>(&self, data: &C) {
        self.camera_binding.update(&self.queue, data);
    }

    // pub fn draw(&mut self, pipeline_mode: PipelineMode) -> Result<(), wgpu::SurfaceError> {
    pub fn draw<'a, I, D>(
        &mut self,
        mtl_iter: I,
        light_mesh_index: Option<MeshBufferIndex>,
    ) -> Result<(), wgpu::SurfaceError>
    where
        I: Iterator<Item = D>,
        D: DrawMaterial,
    {
        // blocks until surface provides render target
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // command buffer for GPU
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RenderCommandEncoder"),
            });
        // clear out window by writing a color

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RenderPass"),
            color_attachments: &[
                // target for fragment shader @location(0)
                Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None, // used for multi-sampling
                    ops: wgpu::Operations {
                        // can skip clear if rendering will cover whole surface anyway
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_bind_group(0, &self.camera_binding.bind_group, &[]);
        render_pass.set_bind_group(1, &self.light_binding.bind_group, &[]);

        for mesh_iter in mtl_iter {
            let material = self
                .model_resource_storage
                .get_material_binding(mesh_iter.get_index());
            // let pipeline = match (&material.color, &material.normal) {
            //     (None, None) => self.texture_pipelines.get_flat().deref(),
            //     (None, Some(_)) => unreachable!(),
            //     (Some(_), None) => self.texture_pipelines.get_color().deref(),
            //     (Some(_), Some(_)) => self.texture_pipelines.get_normal().deref(),
            // };
            // render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(2, &material.phong_binding.bind_group, &[]);
            render_pass.set_bind_group(3, &material.texture_bind_group, &[]);

            for draw_slice in mesh_iter {
                let DrawSlice {
                    buffer_index,
                    slice,
                    instance_index,
                    pipeline_mode,
                } = draw_slice;

                let pipeline = match pipeline_mode {
                    PipelineMode::Flat => self.texture_pipelines.get_flat().deref(),
                    PipelineMode::Color => self.texture_pipelines.get_color().deref(),
                    PipelineMode::Normal => self.texture_pipelines.get_normal().deref(),
                };
                render_pass.set_pipeline(pipeline);

                let mesh = self.model_resource_storage.get_mesh_buffer(buffer_index);

                let instance_slice = if let Some(instance_index) = instance_index {
                    let instances = self
                        .model_resource_storage
                        .get_instance_buffer(instance_index);

                    render_pass.set_vertex_buffer(1, instances.buffer.slice(..));
                    0..instances.num_instances
                } else {
                    0..1 // default to use when no instances are set, per wgpu docs
                };

                render_pass.set_vertex_buffer(0, mesh.vertices.slice(..));

                render_pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);

                render_pass.draw_indexed(slice, 0, instance_slice);
            }
        }

        // render_pass.set_vertex_buffer(1, self.instance_buffer.buffer.slice(..));
        //
        // // scene pass
        // match pipeline_mode {
        //     PipelineMode::Flat => render_pass.set_pipeline(&self.texture_pipelines.get_flat()),
        //     PipelineMode::Color if self.color_pipeline.is_some() => {
        //         let color_pipeline = self.color_pipeline.as_ref().unwrap();
        //         render_pass.set_pipeline(&color_pipeline.0);
        //     }
        //     PipelineMode::Normal if self.normal_pipeline.is_some() => {
        //         let normal_pipeline = &self.normal_pipeline.as_ref().unwrap();
        //         render_pass.set_pipeline(&normal_pipeline.0);
        //     }
        //     _ => render_pass.set_pipeline(&self.flat_pipeline.0),
        // }
        // render_pass.set_vertex_buffer(0, self.mesh_buffer.vertices.slice(..));
        // render_pass.set_index_buffer(
        //     self.mesh_buffer.indices.slice(..),
        //     wgpu::IndexFormat::Uint32,
        // );
        // render_pass.set_bind_group(0, &self.camera_binding.bind_group, &[]);
        // render_pass.set_bind_group(1, &self.light_binding.bind_group, &[]);
        // render_pass.set_bind_group(2, &self.phong_binding.bind_group, &[]);
        // render_pass.set_bind_group(3, &self.material_binding.bind_group, &[]);
        // render_pass.draw_indexed(
        //     0..self.mesh_buffer.num_indices,
        //     0,
        //     0..self.instance_buffer.num_instances,
        // );

        // light pass
        if let Some(index) = light_mesh_index {
            render_pass.set_pipeline(self.light_pipeline.deref());
            let buffer = self.model_resource_storage.get_mesh_buffer(index);
            render_pass.set_vertex_buffer(0, buffer.vertices.slice(..));
            render_pass.draw_indexed(0..buffer.num_indices, 0, 0..1);
        }
        // render_pass.set_pipeline(self.light_pipeline.deref());
        // // reuse bindings from above, see:
        // // https://toji.dev/webgpu-best-practices/bind-groups#reusing-pipeline-layouts
        // render_pass.draw_indexed(0..self.mesh_buffer.num_indices, 0, 0..1);

        // render pass recording ends when dropped
        drop(render_pass);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub fn add_material(
        &mut self,
        material: &model::Material,
        label: Option<&str>,
    ) -> MaterialBindingIndex {
        self.model_resource_storage.upload_material(
            &self.device,
            &self.queue,
            &self.phong_layout,
            &self.texture_layouts,
            &material.phong_params,
            material.color_texture.as_ref(),
            material.normal_texture.as_ref(),
            label,
        )
    }

    pub fn add_mesh_buffer(&mut self, mesh: &model::Mesh, label: Option<&str>) -> MeshBufferIndex {
        self.model_resource_storage
            .upload_mesh(&self.device, mesh, label)
    }

    pub fn add_instance_buffer(
        &mut self,
        instances: &[model::Instance],
        label: Option<&str>,
    ) -> InstanceBufferIndex {
        self.model_resource_storage
            .upload_instance(&self.device, &instances, label)
    }
}
