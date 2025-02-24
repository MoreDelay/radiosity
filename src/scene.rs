use std::{path::PathBuf, sync::Arc};

use cgmath::{EuclideanSpace, InnerSpace, Rotation3, Zero};
use winit::window::Window;

use crate::{camera, light, model, render};

pub struct SceneState {
    render_state: render::SceneRenderState,
    paused: bool,
    simple: bool,
    last_time: std::time::Instant,
    #[expect(unused)]
    model: model::Model,
    #[expect(unused)]
    instances: Vec<model::Instance>,
    camera: camera::Camera,
    light: light::Light,
}

impl SceneState {
    pub fn new(window: Arc<Window>) -> Self {
        let render_init = pollster::block_on(render::RenderStateInit::new(window));

        let eye: cgmath::Point3<f32> = (0.0, 4.0, 7.0).into();
        let target: cgmath::Point3<f32> = (0.0, 0.0, 2.0).into();
        let up = cgmath::Vector3::unit_y();
        let fovy = 45.0;
        let znear = 0.1;
        let zfar = 100.0;
        let dir = target - eye;
        let frame = render_init.get_frame_dim();
        let camera = camera::Camera::new(eye, dir, up, fovy, znear, zfar, frame);

        let position = [2., 2., 2.].into();
        let color = light::Color {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        };
        let light = light::Light::new(position, color);
        let paused = false;
        let last_time = std::time::Instant::now();

        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let model_path = root.join("resources/cube/cube.obj");
        let model = model::Model::load(&model_path).unwrap();

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

        let render_state = render::SceneRenderState::create(
            render_init,
            &camera,
            &light,
            &model.mesh,
            &instances,
            &model.material.color_texture,
            &model.material.normal_texture,
        )
        .unwrap();

        let simple = false;

        SceneState {
            simple,
            render_state,
            paused,
            last_time,
            model,
            instances,
            camera,
            light,
        }
    }

    pub fn resize_window(&mut self, new_size: Option<winit::dpi::PhysicalSize<u32>>) {
        if let Some(winit::dpi::PhysicalSize { width, height }) = new_size {
            if width > 0 && height > 0 {
                let frame_size = camera::FrameDim(width, height);
                self.camera.update_frame(frame_size);
                self.render_state.update_camera(&self.camera);
            }
        }

        self.render_state.resize(new_size);
    }

    pub fn step(&mut self) {
        const TARGET_FPS: f64 = 150.;
        let minimum_elapsed = std::time::Duration::from_secs_f64(1. / TARGET_FPS);

        let elapsed = self.last_time.elapsed();
        if elapsed < minimum_elapsed {
            let sleep_duration = minimum_elapsed - elapsed;
            std::thread::sleep(sleep_duration);
        }
        let elapsed = self.last_time.elapsed();
        let _fps = 1. / elapsed.as_secs_f64();
        self.last_time = std::time::Instant::now();
        // println!("FPS: {_fps}");

        if self.paused {
            return;
        }
        let old_pos = self.light.pos.to_vec();
        let new_pos =
            cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_y(), cgmath::Deg(1.))
                * old_pos;
        self.light.pos = cgmath::Point3::from_vec(new_pos);

        self.render_state.update_light(&self.light);
        // self.render_state.update_camera(&self.camera);
    }

    pub fn draw(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.render_state.draw(self.simple)
    }

    /// returns true if paused after toggling
    pub fn toggle_pause(&mut self) -> bool {
        self.paused ^= true;
        self.paused
    }

    /// returns true if simple on after toggling
    pub fn toggle_simple(&mut self) -> bool {
        self.simple ^= true;
        self.simple
    }
}
