use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use pipeline::{LightPipeline, ScenePipeline};
use resource::{InstanceBuffer, MaterialBindingCN, MeshBuffer};
use winit::{event::WindowEvent, window::Window};

use cgmath::{EuclideanSpace, InnerSpace, Rotation3, Zero};

use crate::{camera, light, model, texture};

mod layout;
mod pipeline;
mod resource;

pub use layout::*;

pub struct RenderState {
    pub paused: bool,
    last_time: std::time::Instant,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub window: Arc<Window>,
    scene_pipeline: ScenePipeline,
    #[allow(unused)]
    obj_model: model::Model,
    mesh_buffer: resource::MeshBuffer,
    material_binding: resource::MaterialBindingCN,
    #[allow(unused)]
    instances: Vec<model::Instance>,
    instance_buffer: resource::InstanceBuffer,
    depth_texture: texture::Texture,
    camera: camera::Camera,
    camera_binding: resource::CameraBinding,
    light: light::Light,
    light_binding: resource::LightBinding,
    light_pipeline: LightPipeline,
}

impl RenderState {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        // handle to create adapters and surfaces
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
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
            .request_device(
                &wgpu::DeviceDescriptor {
                    // list available features with adapter.features() or device.features()
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: Default::default(),
                },
                None, // trace path
            )
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
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let texture_layout = resource::TextureBindGroupLayout::new(&device);

        let eye: cgmath::Point3<f32> = (0.0, 4.0, 7.0).into();
        let target: cgmath::Point3<f32> = (0.0, 0.0, 2.0).into();
        let up = cgmath::Vector3::unit_y();
        let fovy = 45.0;
        let znear = 0.1;
        let zfar = 100.0;
        let frame_size = (config.width, config.height).into();
        let dir = target - eye;
        let camera = camera::Camera::new(eye, dir, up, fovy, znear, zfar, frame_size);

        let camera_layout = resource::CameraBindGroupLayout::new(&device);
        let camera_binding =
            resource::CameraBinding::new(&device, &camera_layout, &camera, Some("Single"));

        let position = [2., 2., 2.].into();
        let color = light::Color {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        };
        let light = light::Light::new(position, color);

        let light_layout = resource::LightBindGroupLayout::new(&device);
        let light_binding =
            resource::LightBinding::new(&device, &light_layout, &light, Some("Single"));

        let scene_pipeline = ScenePipeline::new(
            &device,
            config.format,
            &texture_layout,
            &camera_layout,
            &light_layout,
        )?;

        let light_pipeline =
            LightPipeline::new(&device, config.format, &camera_layout, &light_layout)?;

        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let model_path = root.join("resources/cube/cube.obj");
        let obj_model = model::Model::load(&model_path).unwrap();
        let single_model = &obj_model.meshes[0];
        let mesh_buffer = MeshBuffer::new(
            &device,
            &single_model.vertices,
            &single_model.triangles,
            Some("Cube"),
        );
        let single_material = &obj_model.materials[0];
        let color_texture = &single_material.color_texture;
        let normal_texture = &single_material.normal_texture;

        let texture_layout = resource::TextureBindGroupLayout::new(&device);
        let material_binding = MaterialBindingCN::new(
            &device,
            &queue,
            &texture_layout,
            color_texture,
            normal_texture,
            Some("Cube"),
        );

        const SPACE_BETWEEN: f32 = 3.0;
        let instances = (0..model::NUM_INSTANCES_PER_ROW)
            .flat_map(|z| {
                (0..model::NUM_INSTANCES_PER_ROW).map(move |x| {
                    let x = SPACE_BETWEEN * (x as f32 - model::NUM_INSTANCES_PER_ROW as f32 / 2.);
                    let z = SPACE_BETWEEN * (z as f32 - model::NUM_INSTANCES_PER_ROW as f32 / 2.);

                    let position = cgmath::Vector3::new(x, 0.0, z);
                    let rotation = if position.is_zero() {
                        cgmath::Quaternion::from_axis_angle(
                            cgmath::Vector3::unit_z(),
                            cgmath::Deg(0.0),
                        )
                    } else {
                        cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                    };
                    model::Instance { position, rotation }
                })
            })
            .collect::<Vec<_>>();

        let instance_buffer = InstanceBuffer::new(&device, &instances, Some("Grid"));

        let paused = false;
        let last_time = std::time::Instant::now();

        Ok(Self {
            paused,
            last_time,
            surface,
            device,
            queue,
            config,
            size,
            window,
            obj_model,
            instances,
            instance_buffer,
            depth_texture,
            camera_binding,
            light_binding,
            scene_pipeline,
            light_pipeline,
            camera,
            light,
            mesh_buffer,
            material_binding,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            // resize depth texture
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
            let frame_size = (new_size.width, new_size.height).into();
            self.camera.update_frame(frame_size);
            self.camera_binding.update(&self.queue, &self.camera);
        }
    }

    pub fn check_completed(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    pub fn update(&mut self) {
        let elapsed = self.last_time.elapsed();
        let elapsed = elapsed.as_secs_f64();
        let fps = 1. / elapsed;
        self.last_time = std::time::Instant::now();
        println!("FPS: {fps}");

        if self.paused {
            return;
        }
        let old_pos = self.light.pos.to_vec();
        let new_pos =
            cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_y(), cgmath::Deg(0.1))
                * old_pos;
        self.light.pos = cgmath::Point3::from_vec(new_pos);
        self.light_binding.update(&self.queue, self.light);
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // blocks until surface provides render target
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // command buffer for GPU
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        // clear out window by writing a color

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
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
        render_pass.set_vertex_buffer(1, self.instance_buffer.buffer.slice(..));
        render_pass.set_pipeline(&self.light_pipeline.0);
        render_pass.set_vertex_buffer(0, self.mesh_buffer.vertices.slice(..));
        render_pass.set_index_buffer(
            self.mesh_buffer.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.set_bind_group(0, &self.camera_binding.bind_group, &[]);
        render_pass.set_bind_group(1, &self.light_binding.bind_group, &[]);
        render_pass.draw_indexed(0..self.mesh_buffer.num_indices, 0, 0..1);

        // render_pass.draw_light_model(
        //     &self.obj_model,
        //     &self.camera_bind_group,
        //     &self.light_bind_group,
        // );

        render_pass.set_pipeline(&self.scene_pipeline.0);
        render_pass.set_vertex_buffer(0, self.mesh_buffer.vertices.slice(..));
        render_pass.set_index_buffer(
            self.mesh_buffer.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.set_bind_group(0, &self.material_binding.bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_binding.bind_group, &[]);
        render_pass.set_bind_group(2, &self.light_binding.bind_group, &[]);
        render_pass.draw_indexed(
            0..self.mesh_buffer.num_indices,
            0,
            0..self.instance_buffer.num_instances,
        );
        // render_pass.draw_model_instanced(
        //     &self.obj_model,
        //     0..self.instances.len() as u32,
        //     &self.camera_bind_group,
        //     &self.light_bind_group,
        // );

        // render pass recording ends when dropped
        drop(render_pass);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
