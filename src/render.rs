use std::{num::NonZeroU32, ops::Deref, sync::Arc};

use anyhow::Result;
use winit::window::Window;

use crate::{
    camera, model,
    render::resource::{ResourceStorage, TextureDims},
};

mod pipeline;
mod raw;
mod resource;

pub use pipeline::PhongCapabilites;
pub use raw::*;
pub use resource::{
    BiTangentBufferIndex,
    DrawCall,
    DrawWorld,
    IndexBufferIndex,
    InstanceBufferIndex,
    MaterialBindingIndex,
    NormalBufferIndex,
    PositionBufferIndex,
    TangentBufferIndex,
    TexCoordBufferIndex, //
};

const SHADER_ROOT: &str = "src/shaders";

#[derive(Copy, Clone, Debug)]
pub enum PipelineMode {
    Flat,
    Color,
    Normal,
    ColorNormal,
}

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

pub struct TargetContext {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
}

impl TargetContext {
    pub fn get_frame_dim(&self) -> camera::FrameDim {
        (self.config.width, self.config.height).into()
    }
}

pub async fn create_render_instance(window: Arc<Window>) -> (GpuContext, TargetContext) {
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
        max_bind_groups: 6,
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

    (
        GpuContext { device, queue },
        TargetContext {
            surface,
            config,
            size,
        },
    )
}

pub struct RenderState {
    ctx: GpuContext,
    target: TargetContext,
    layouts: resource::PhongLayouts,
    phong_pipelines: pipeline::PhongPipelines,
    depth_texture: resource::DepthTexture,
    shadow_binding: resource::ShadowBindings,
    shadow_depth_texture: resource::DepthTexture,
    camera_binding: resource::CameraBinding,
    light_binding: resource::LightBinding,
    model_resource_storage: ResourceStorage,
}

impl RenderState {
    pub fn create(
        ctx: GpuContext,
        target: TargetContext,
        camera: &CameraRaw,
        light: &LightRaw,
    ) -> anyhow::Result<Self> {
        let depth_dims = TextureDims {
            width: NonZeroU32::new(target.config.width.max(1)).unwrap(),
            height: NonZeroU32::new(target.config.height.max(1)).unwrap(),
        };
        let depth_texture = resource::DepthTexture::new(&ctx, depth_dims, Some("depth_texture"));

        let layouts = resource::PhongLayouts::new(&ctx);
        let phong_pipelines = pipeline::PhongPipelines::new(&ctx, &target, &layouts);

        let camera_binding = resource::CameraBinding::new(&ctx, &layouts, camera, Some("Single"));
        let light_binding = resource::LightBinding::new(&ctx, &layouts, light, Some("Single"));

        let dims = TextureDims {
            width: NonZeroU32::new(1024).unwrap(),
            height: NonZeroU32::new(1024).unwrap(),
        };
        let shadow_binding =
            resource::ShadowBindings::new(&ctx, &layouts, dims, light, Some("Shadow"));
        let shadow_depth_texture =
            resource::DepthTexture::new(&ctx, dims, Some("shadow_depth_texture"));

        let model_resource_storage = ResourceStorage::new();

        Ok(Self {
            ctx,
            target,
            layouts,
            phong_pipelines,
            depth_texture,
            shadow_binding,
            shadow_depth_texture,
            camera_binding,
            light_binding,
            model_resource_storage,
        })
    }

    pub fn resize(&mut self, new_size: Option<winit::dpi::PhysicalSize<u32>>) {
        let new_size = new_size.unwrap_or(self.target.size);
        if new_size.width > 0 && new_size.height > 0 {
            self.target.size = new_size;
            self.target.config.width = new_size.width;
            self.target.config.height = new_size.height;
            self.target
                .surface
                .configure(&self.ctx.device, &self.target.config);
            // resize depth texture
            let dims = TextureDims {
                width: NonZeroU32::new(new_size.width).unwrap(),
                height: NonZeroU32::new(new_size.height).unwrap(),
            };
            self.depth_texture =
                resource::DepthTexture::new(&self.ctx, dims, Some("depth_texture"));
        }
    }

    pub fn update_light(&self, data: &LightRaw) {
        self.light_binding.update(&self.ctx, data);
        self.shadow_binding.update(&self.ctx, data);
    }

    pub fn update_camera(&self, data: &CameraRaw) {
        self.camera_binding.update(&self.ctx, data);
    }

    pub fn draw(&mut self, draw_world: resource::DrawWorld) -> Result<(), wgpu::SurfaceError> {
        let sorted_draws = draw_world.sort(&self.model_resource_storage);

        // blocks until surface provides render target
        let output = self.target.surface.get_current_texture()?;
        let output_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // command buffer for GPU
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RenderCommandEncoder"),
            });

        // create shadow map
        let resource::ShadowBindings {
            transform_binds,
            cube_views,
            ..
        } = &self.shadow_binding;

        for (cube_view, transform_bind) in cube_views.iter().zip(transform_binds.iter()) {
            let mut shadow_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow RenderPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: cube_view,
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

            let pipeline = self.phong_pipelines.get_shadow();

            shadow_render_pass.set_pipeline(pipeline);
            shadow_render_pass.set_bind_group(0, transform_bind, &[]);
            shadow_render_pass.set_bind_group(1, &self.light_binding.bind_group, &[]);

            for &index in sorted_draws.iter() {
                let draw = &draw_world.draw_calls[index as usize];

                let index = self.model_resource_storage.index_buffer(draw.index);
                let position = self.model_resource_storage.position_buffer(draw.position);
                let instance = self
                    .model_resource_storage
                    .get_instance_buffer(draw.instance);

                shadow_render_pass
                    .set_index_buffer(index.buffer.slice(..), wgpu::IndexFormat::Uint32);
                shadow_render_pass.set_vertex_buffer(0, position.buffer.slice(..));
                shadow_render_pass.set_vertex_buffer(1, instance.buffer.slice(..));
                shadow_render_pass.draw_indexed(draw.slice.clone(), 0, 0..instance.num_instances);
            }
        }

        // render models
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("RenderPass"),
            color_attachments: &[
                // target for fragment shader @location(0)
                Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
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
        let shadow_texture = &self.shadow_binding.cube_bind;

        // constant bindings
        render_pass.set_bind_group(0, &self.camera_binding.bind_group, &[]);
        render_pass.set_bind_group(1, &self.light_binding.bind_group, &[]);
        render_pass.set_bind_group(3, shadow_texture, &[]);

        #[derive(Debug, Copy, Clone)]
        struct DrawState {
            material: MaterialBindingIndex,
            requirements: pipeline::PhongResourceRequirements,
        }
        let mut draw_state: Option<DrawState> = None;
        for &index in sorted_draws.iter() {
            let draw = &draw_world.draw_calls[index as usize];

            if draw_state.is_none_or(|state| state.material != draw.material) {
                let material = self
                    .model_resource_storage
                    .get_material_binding(draw.material);
                let caps = PhongCapabilites {
                    color_map: material.color.is_some() && draw.caps_filter.color_map,
                    normal_map: material.normal.is_some() && draw.caps_filter.normal_map,
                };
                let pipeline = self.phong_pipelines.get_render(caps);
                let reqs = pipeline.requirements();

                render_pass.set_pipeline(pipeline);

                if let Some(color_index) = reqs.bindings.color_texture {
                    let color_bind = material.color.as_ref().expect("checked above").deref();
                    render_pass.set_bind_group(color_index, color_bind, &[]);
                }
                if let Some(normal_index) = reqs.bindings.normal_texture {
                    let normal_bind = material.normal.as_ref().expect("checked above").deref();
                    render_pass.set_bind_group(normal_index, normal_bind, &[]);
                }

                render_pass.set_bind_group(2, &material.phong_binding.bind_group, &[]);

                draw_state = Some(DrawState {
                    material: draw.material,
                    requirements: reqs,
                });
            }
            let reqs = draw_state.expect("set above").requirements;

            let index = self.model_resource_storage.index_buffer(draw.index);

            let mut vertex_slot = 0;
            let mut next_slot = || {
                vertex_slot += 1;
                vertex_slot - 1
            };

            if reqs.vertex.position.filled() {
                let position = self.model_resource_storage.position_buffer(draw.position);
                render_pass.set_vertex_buffer(next_slot(), position.buffer.slice(..));
            }

            if reqs.vertex.tex_coord.filled() {
                let tex_coord = self.model_resource_storage.tex_coord_buffer(draw.tex_coord);
                render_pass.set_vertex_buffer(next_slot(), tex_coord.buffer.slice(..));
            }

            if reqs.vertex.normal.filled() {
                let normal = self.model_resource_storage.normal_buffer(draw.normal);
                render_pass.set_vertex_buffer(next_slot(), normal.buffer.slice(..));
            }

            if reqs.vertex.tangent.filled() {
                let tangent = self.model_resource_storage.tangent_buffer(draw.tangent);
                render_pass.set_vertex_buffer(next_slot(), tangent.buffer.slice(..));
            }

            if reqs.vertex.bi_tangent.filled() {
                let bi_tangent = self
                    .model_resource_storage
                    .bi_tangent_buffer(draw.bi_tangent);
                render_pass.set_vertex_buffer(next_slot(), bi_tangent.buffer.slice(..));
            }

            let instances = if reqs.vertex.instance.filled() {
                let instance = self
                    .model_resource_storage
                    .get_instance_buffer(draw.instance);
                render_pass.set_vertex_buffer(next_slot(), instance.buffer.slice(..));
                instance.num_instances
            } else {
                1
            };

            render_pass.set_index_buffer(index.buffer.slice(..), wgpu::IndexFormat::Uint32);

            render_pass.draw_indexed(draw.slice.clone(), 0, 0..instances);
        }

        // for mesh_iter in mtl_iter {
        //     let material = self
        //         .model_resource_storage
        //         .get_material_binding(mesh_iter.get_index());
        //     render_pass.set_bind_group(2, &material.phong_binding.bind_group, &[]);
        //
        //     for draw_slice in mesh_iter {
        //         let DrawSlice {
        //             buffer_index,
        //             slice,
        //             instance_index,
        //             pipeline_mode,
        //         } = draw_slice;
        //
        //         match pipeline_mode {
        //             PipelineMode::ColorNormal
        //                 if material.color.is_some() && material.normal.is_some() =>
        //             {
        //                 let pipeline = self.texture_pipelines.get_color_normal().deref();
        //                 let color_bind_group = material
        //                     .color
        //                     .as_ref()
        //                     .expect("requested pipeline requires corresponding texture")
        //                     .deref();
        //                 let normal_bind_group = material
        //                     .normal
        //                     .as_ref()
        //                     .expect("requested pipeline requires corresponding texture")
        //                     .deref();
        //                 render_pass.set_pipeline(pipeline);
        //                 render_pass.set_bind_group(3, color_bind_group, &[]);
        //                 render_pass.set_bind_group(4, normal_bind_group, &[]);
        //             }
        //             PipelineMode::Color if material.color.is_some() => {
        //                 let pipeline = self.texture_pipelines.get_color().deref();
        //                 let color_bind_group = material
        //                     .color
        //                     .as_ref()
        //                     .expect("requested pipeline requires corresponding texture")
        //                     .deref();
        //                 render_pass.set_pipeline(pipeline);
        //                 render_pass.set_bind_group(3, color_bind_group, &[]);
        //             }
        //             PipelineMode::Normal if material.normal.is_some() => {
        //                 let pipeline = self.texture_pipelines.get_normal().deref();
        //                 let normal_bind_group = material
        //                     .normal
        //                     .as_ref()
        //                     .expect("requested pipeline requires corresponding texture")
        //                     .deref();
        //                 render_pass.set_pipeline(pipeline);
        //                 render_pass.set_bind_group(3, normal_bind_group, &[]);
        //             }
        //             _ => {
        //                 let pipeline = self.texture_pipelines.get_flat().deref();
        //                 let texture = &self.shadow_binding.cube_bind;
        //                 render_pass.set_pipeline(pipeline);
        //                 render_pass.set_bind_group(3, texture, &[]);
        //             }
        //         }
        //
        //         let mesh = self.model_resource_storage.get_mesh_buffer(buffer_index);
        //
        //         let instance_slice = if let Some(instance_index) = instance_index {
        //             let instances = self
        //                 .model_resource_storage
        //                 .get_instance_buffer(instance_index);
        //
        //             render_pass.set_vertex_buffer(1, instances.buffer.slice(..));
        //             0..instances.num_instances
        //         } else {
        //             0..1 // default to use when no instances are set, per wgpu docs
        //         };
        //
        //         render_pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        //         render_pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        //
        //         render_pass.draw_indexed(slice, 0, instance_slice);
        //     }
        // }

        // // light pass
        // if let Some(index) = light_mesh_index {
        //     render_pass.set_pipeline(self.light_pipeline.deref());
        //     let buffer = self.model_resource_storage.get_mesh_buffer(index);
        //     render_pass.set_vertex_buffer(0, buffer.vertices.slice(..));
        //     render_pass.draw_indexed(0..buffer.num_indices, 0, 0..1);
        // }

        // render pass recording ends when dropped
        drop(render_pass);

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub fn upload_index_buffer(
        &mut self,
        data: &[u32],
        label: Option<&str>,
    ) -> resource::IndexBufferIndex {
        self.model_resource_storage
            .upload_index_buffer(&self.ctx, data, label)
    }

    pub fn upload_position_buffer(
        &mut self,
        data: &[f32],
        label: Option<&str>,
    ) -> resource::PositionBufferIndex {
        self.model_resource_storage
            .upload_position_buffer(&self.ctx, data, label)
    }

    pub fn upload_tex_coord_buffer(
        &mut self,
        data: &[f32],
        label: Option<&str>,
    ) -> resource::TexCoordBufferIndex {
        self.model_resource_storage
            .upload_tex_coord_buffer(&self.ctx, data, label)
    }

    pub fn upload_normal_buffer(
        &mut self,
        data: &[f32],
        label: Option<&str>,
    ) -> resource::NormalBufferIndex {
        self.model_resource_storage
            .upload_normal_buffer(&self.ctx, data, label)
    }

    pub fn upload_tangent_buffer(
        &mut self,
        data: &[f32],
        label: Option<&str>,
    ) -> resource::TangentBufferIndex {
        self.model_resource_storage
            .upload_tangent_buffer(&self.ctx, data, label)
    }

    pub fn upload_bi_tangent_buffer(
        &mut self,
        data: &[f32],
        label: Option<&str>,
    ) -> resource::BiTangentBufferIndex {
        self.model_resource_storage
            .upload_bi_tangent_buffer(&self.ctx, data, label)
    }

    pub fn add_material(
        &mut self,
        phong: &PhongRaw,
        color: Option<&model::Image>,
        normal: Option<&model::Image>,
        label: Option<&str>,
    ) -> MaterialBindingIndex {
        self.model_resource_storage.upload_material(
            &self.ctx,
            &self.layouts,
            phong,
            color,
            normal,
            label,
        )
    }

    // pub fn add_mesh_buffer(
    //     &mut self,
    //     mesh: &model::MeshCombined,
    //     label: Option<&str>,
    // ) -> MeshBufferIndex {
    //     self.model_resource_storage
    //         .upload_mesh(&self.device, mesh, label)
    // }

    pub fn add_instance_buffer(
        &mut self,
        instances: &[model::Instance],
        label: Option<&str>,
    ) -> InstanceBufferIndex {
        self.model_resource_storage
            .upload_instance(&self.ctx, &instances, label)
    }
}
