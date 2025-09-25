use std::{
    num::NonZeroU32,
    ops::{Deref, Range},
    sync::Arc,
};

use anyhow::Result;
use winit::window::Window;

use crate::{
    camera, model,
    render::resource::{ModelResourceStorage, TextureDims},
};

mod pipeline;
mod raw;
mod resource;

pub use raw::*;
pub use resource::{InstanceBufferIndex, MaterialBindingIndex, MeshBufferIndex};

#[derive(Copy, Clone, Debug)]
pub enum PipelineMode {
    Flat,
    Color,
    Normal,
    ColorNormal,
}

pub struct RenderStateInit {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
}

pub struct RenderState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    phong_layout: resource::PhongBindGroupLayout,
    texture_layout: resource::TextureBindGroupLayout,
    texture_pipelines: pipeline::TexturePipelines,
    #[expect(unused)]
    mesh_buffers: Vec<resource::MeshBuffer>,
    #[expect(unused)]
    material_bindings: Vec<resource::MaterialBindings>,
    #[expect(unused)]
    instance_buffers: Vec<resource::InstanceBuffer>,
    depth_texture: resource::DepthTexture,
    shadow_binding: resource::ShadowBindings,
    shadow_pipeline: pipeline::ShadowPipeline,
    shadow_depth_texture: resource::DepthTexture,
    camera_binding: resource::CameraBinding,
    light_binding: resource::LightBinding,
    #[expect(unused)]
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

        let required_limits = wgpu::Limits {
            max_bind_groups: 5,
            ..Default::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                // list available features with adapter.features() or device.features()
                required_features: wgpu::Features::empty(),
                required_limits,
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

        Self {
            surface,
            device,
            queue,
            config,
            size,
        }
    }

    pub fn get_frame_dim(&self) -> camera::FrameDim {
        (self.config.width, self.config.height).into()
    }
}

impl RenderState {
    #[allow(clippy::too_many_arguments)]
    pub fn create<C, L>(init: RenderStateInit, camera: &C, light: &L) -> anyhow::Result<Self>
    where
        C: GpuTransfer<Raw = CameraRaw>,
        L: GpuTransfer<Raw = LightRaw>,
    {
        let RenderStateInit {
            surface,
            device,
            queue,
            config,
            size,
        } = init;

        let depth_dims = TextureDims {
            width: NonZeroU32::new(config.width.max(1)).unwrap(),
            height: NonZeroU32::new(config.height.max(1)).unwrap(),
        };
        let depth_texture = resource::DepthTexture::new(&device, depth_dims, Some("depth_texture"));

        let texture_layout = resource::TextureBindGroupLayout::new(&device);

        let camera_layout = resource::CameraBindGroupLayout::new(&device);
        let camera_binding =
            resource::CameraBinding::new(&device, &camera_layout, camera, Some("Single"));

        let light_layout = resource::LightBindGroupLayout::new(&device);
        let light_binding =
            resource::LightBinding::new(&device, &light_layout, light, Some("Single"));

        let phong_layout = resource::PhongBindGroupLayout::new(&device);

        let dims = TextureDims {
            width: NonZeroU32::new(1024).unwrap(),
            height: NonZeroU32::new(1024).unwrap(),
        };
        let shadow_layout = resource::ShadowLayouts::new(&device);
        let shadow_binding = resource::ShadowBindings::new(
            &device,
            &shadow_layout,
            dims,
            &light.to_raw(),
            Some("Shadow"),
        );
        let shadow_depth_texture =
            resource::DepthTexture::new(&device, dims, Some("shadow_depth_texture"));

        let texture_pipelines = pipeline::TexturePipelines::new(
            &device,
            config.format,
            &texture_layout,
            &camera_layout,
            &light_layout,
            &shadow_layout,
            &phong_layout,
        )?;
        let shadow_pipeline =
            pipeline::ShadowPipeline::new(&device, &shadow_layout, &light_layout)?;

        let light_pipeline =
            pipeline::LightPipeline::new(&device, config.format, &camera_layout, &light_layout)?;

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
            shadow_binding,
            shadow_pipeline,
            shadow_depth_texture,
            phong_layout,
            camera_binding,
            light_binding,
            texture_layout,
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
            let dims = TextureDims {
                width: NonZeroU32::new(new_size.width).unwrap(),
                height: NonZeroU32::new(new_size.height).unwrap(),
            };
            self.depth_texture =
                resource::DepthTexture::new(&self.device, dims, Some("depth_texture"));
        }
    }

    pub fn update_light<L: GpuTransfer<Raw = LightRaw>>(&self, data: &L) {
        self.light_binding.update(&self.queue, data);
        self.shadow_binding.update(&self.queue, &data.to_raw());
    }

    pub fn update_camera<C: GpuTransfer<Raw = CameraRaw>>(&self, data: &C) {
        self.camera_binding.update(&self.queue, data);
    }

    pub fn draw<I, D>(
        &mut self,
        mtl_iter: I,
        #[expect(unused)] light_mesh_index: Option<MeshBufferIndex>,
    ) -> Result<(), wgpu::SurfaceError>
    where
        I: Iterator<Item = D> + Clone,
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

        // create shadow map
        let resource::ShadowBindings {
            transform_binds,
            layer_views,
            ..
        } = &self.shadow_binding;

        for (view, bind) in layer_views.iter().zip(transform_binds.iter()) {
            let mut shadow_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow RenderPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None, // used for multi-sampling
                    ops: wgpu::Operations {
                        // can skip clear if rendering will cover whole surface anyway
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.,
                            g: 1.,
                            b: 1.,
                            a: 1.,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            shadow_render_pass.set_pipeline(&self.shadow_pipeline);
            shadow_render_pass.set_bind_group(0, bind, &[]);
            shadow_render_pass.set_bind_group(1, &self.light_binding.bind_group, &[]);

            for mesh_iter in mtl_iter.clone() {
                for draw_slice in mesh_iter {
                    let DrawSlice {
                        buffer_index,
                        slice,
                        instance_index,
                        ..
                    } = draw_slice;

                    let mesh = self.model_resource_storage.get_mesh_buffer(buffer_index);

                    let instance_slice = if let Some(instance_index) = instance_index {
                        let instances = self
                            .model_resource_storage
                            .get_instance_buffer(instance_index);

                        shadow_render_pass.set_vertex_buffer(1, instances.buffer.slice(..));
                        0..instances.num_instances
                    } else {
                        0..1 // default to use when no instances are set, per wgpu docs
                    };

                    shadow_render_pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                    shadow_render_pass
                        .set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);

                    shadow_render_pass.draw_indexed(slice, 0, instance_slice);
                }
            }
        }

        // render models
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
            render_pass.set_bind_group(2, &material.phong_binding.bind_group, &[]);

            for draw_slice in mesh_iter {
                let DrawSlice {
                    buffer_index,
                    slice,
                    instance_index,
                    pipeline_mode,
                } = draw_slice;

                match pipeline_mode {
                    PipelineMode::Flat => {
                        let pipeline = self.texture_pipelines.get_flat().deref();
                        let texture = &self.shadow_binding.cube_bind;
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_bind_group(3, texture, &[]);
                    }
                    PipelineMode::Color => {
                        let pipeline = self.texture_pipelines.get_color().deref();
                        let color_bind_group = material
                            .color
                            .as_ref()
                            .expect("requested pipeline requires corresponding texture")
                            .deref();
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_bind_group(3, color_bind_group, &[]);
                    }
                    PipelineMode::Normal => {
                        let pipeline = self.texture_pipelines.get_normal().deref();
                        let normal_bind_group = material
                            .normal
                            .as_ref()
                            .expect("requested pipeline requires corresponding texture")
                            .deref();
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_bind_group(3, normal_bind_group, &[]);
                    }
                    PipelineMode::ColorNormal => {
                        let pipeline = self.texture_pipelines.get_color_normal().deref();
                        let color_bind_group = material
                            .color
                            .as_ref()
                            .expect("requested pipeline requires corresponding texture")
                            .deref();
                        let normal_bind_group = material
                            .normal
                            .as_ref()
                            .expect("requested pipeline requires corresponding texture")
                            .deref();
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_bind_group(3, color_bind_group, &[]);
                        render_pass.set_bind_group(4, normal_bind_group, &[]);
                    }
                }

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

        // light pass
        // if let Some(index) = light_mesh_index {
        //     render_pass.set_pipeline(self.light_pipeline.deref());
        //     let buffer = self.model_resource_storage.get_mesh_buffer(index);
        //     render_pass.set_vertex_buffer(0, buffer.vertices.slice(..));
        //     render_pass.draw_indexed(0..buffer.num_indices, 0, 0..1);
        // }

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
            &self.texture_layout,
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
