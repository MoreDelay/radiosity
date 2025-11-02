use std::{
    num::{NonZeroU32, NonZeroU64},
    ops::Deref,
    sync::Arc,
};

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
    DrawType,
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

    pub fn update_instance(&self, index: InstanceBufferIndex, data: &[model::Instance]) {
        let instance = &self.model_resource_storage.get_instance_buffer(index);
        assert_eq!(data.len(), instance.num_instances as usize);
        let size = NonZeroU64::try_from(std::mem::size_of::<InstanceRaw>() as u64).unwrap();
        let mut view = self
            .ctx
            .queue
            .write_buffer_with(&instance.buffer, 0, size)
            .unwrap();

        let data = data.iter().map(|i| i.to_raw()).collect::<Vec<_>>();
        view.copy_from_slice(bytemuck::cast_slice(&data));
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
            draw_type: DrawType,
            material: MaterialBindingIndex,
            requirements: pipeline::PhongResourceRequirements,
        }
        let mut draw_state: Option<DrawState> = None;
        for &index in sorted_draws.iter() {
            let draw = &draw_world.draw_calls[index as usize];

            let pipeline = match draw.draw_type {
                DrawType::Light => self.phong_pipelines.get_light(),
                DrawType::Render(caps_filter) => {
                    // get actual pipeline
                    let material = self
                        .model_resource_storage
                        .get_material_binding(draw.material);
                    let caps = PhongCapabilites {
                        color_map: material.color.is_some() && caps_filter.color_map,
                        normal_map: material.normal.is_some() && caps_filter.normal_map,
                    };
                    self.phong_pipelines.get_render(caps)
                }
            };
            if draw_state.is_none_or(|state| {
                state.material != draw.material || state.draw_type != draw.draw_type
            }) {
                render_pass.set_pipeline(pipeline);

                let material = self
                    .model_resource_storage
                    .get_material_binding(draw.material);

                let reqs = pipeline.requirements();
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
                    draw_type: draw.draw_type,
                    material: draw.material,
                    requirements: reqs,
                });
            }

            let reqs = draw_state.expect("set above").requirements;

            let index = self.model_resource_storage.index_buffer(draw.index);

            let mut slot_iter = (0..).into_iter();
            if reqs.vertex.position.filled() {
                let slot = slot_iter.next().unwrap();
                let position = self.model_resource_storage.position_buffer(draw.position);
                render_pass.set_vertex_buffer(slot, position.buffer.slice(..));
            }
            if reqs.vertex.tex_coord.filled() {
                let slot = slot_iter.next().unwrap();
                let tex_coord = draw.tex_coord.expect("required by chosen pipeline");
                let tex_coord = self.model_resource_storage.tex_coord_buffer(tex_coord);
                render_pass.set_vertex_buffer(slot, tex_coord.buffer.slice(..));
            }
            if reqs.vertex.normal.filled() {
                let slot = slot_iter.next().unwrap();
                let normal = draw.normal.expect("required by chosen pipeline");
                let normal = self.model_resource_storage.normal_buffer(normal);
                render_pass.set_vertex_buffer(slot, normal.buffer.slice(..));
            }
            if reqs.vertex.tangent.filled() {
                let slot = slot_iter.next().unwrap();
                let tangent = draw.tangent.expect("required by chosen pipeline");
                let tangent = self.model_resource_storage.tangent_buffer(tangent);
                render_pass.set_vertex_buffer(slot, tangent.buffer.slice(..));
            }
            if reqs.vertex.bi_tangent.filled() {
                let slot = slot_iter.next().unwrap();
                let bi_tangent = draw.bi_tangent.expect("required by chosen pipeline");
                let bi_tangent = self.model_resource_storage.bi_tangent_buffer(bi_tangent);
                render_pass.set_vertex_buffer(slot, bi_tangent.buffer.slice(..));
            }

            let instances = if reqs.vertex.instance.filled() {
                let slot = slot_iter.next().unwrap();
                let instance = self
                    .model_resource_storage
                    .get_instance_buffer(draw.instance);
                render_pass.set_vertex_buffer(slot, instance.buffer.slice(..));
                instance.num_instances
            } else {
                1
            };

            render_pass.set_index_buffer(index.buffer.slice(..), wgpu::IndexFormat::Uint32);

            render_pass.draw_indexed(draw.slice.clone(), 0, 0..instances);
        }

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
        color: Option<&TextureRaw>,
        normal: Option<&TextureRaw>,
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

    pub fn add_instance_buffer(
        &mut self,
        instances: &[model::Instance],
        label: Option<&str>,
    ) -> InstanceBufferIndex {
        self.model_resource_storage
            .upload_instance(&self.ctx, instances, label)
    }
}
