use std::{path::PathBuf, sync::Arc};

use cgmath::{EuclideanSpace, InnerSpace, Rotation3, Zero};
use winit::window::Window;

use crate::{camera, light, model, render};

pub struct SceneState {
    render_state: render::SceneRenderState,
    paused: bool,
    pipeline_mode: render::PipelineMode,
    use_first_person_camera: bool,
    last_time: std::time::Instant,
    #[expect(unused)]
    model: model::Model,
    #[expect(unused)]
    instances: Vec<model::Instance>,
    target_camera: camera::TargetCamera,
    first_person_camera: camera::FirstPersonCamera,
    light: light::Light,
}

impl SceneState {
    pub fn new(window: Arc<Window>) -> Self {
        let render_init = pollster::block_on(render::RenderStateInit::new(window));

        let pos: cgmath::Point3<f32> = (0.0, 4.0, 7.0).into();
        let target: cgmath::Point3<f32> = (0.0, 0.0, 2.0).into();
        let distance = 10.;
        let frame = render_init.get_frame_dim();
        let target_camera = camera::TargetCamera::new(pos, target, distance, frame);
        let first_person_camera = camera::FirstPersonCamera::new(pos, target - pos, frame);

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
        let model = model::Model::alternative(&model_path).unwrap();

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
            &target_camera,
            &light,
            &model.mesh,
            &instances,
            // model.material.color_texture.as_ref(),
            // model.material.normal_texture.as_ref(),
            Option::<&model::ColorTexture>::None,
            Option::<&model::NormalTexture>::None,
        )
        .unwrap();

        let pipeline_mode = render::PipelineMode::Flat;
        let use_first_person_camera = false;

        SceneState {
            pipeline_mode,
            render_state,
            paused,
            use_first_person_camera,
            last_time,
            model,
            instances,
            target_camera,
            first_person_camera,
            light,
        }
    }

    pub fn resize_window(&mut self, new_size: Option<winit::dpi::PhysicalSize<u32>>) {
        if let Some(winit::dpi::PhysicalSize { width, height }) = new_size {
            if width > 0 && height > 0 {
                let frame_size = camera::FrameDim(width, height);
                self.target_camera.update_frame(frame_size);
                self.first_person_camera.update_frame(frame_size);
                if self.use_first_person_camera {
                    self.render_state.update_camera(&self.first_person_camera);
                } else {
                    self.render_state.update_camera(&self.target_camera);
                }
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

    pub fn drag_camera(&mut self, movement: cgmath::Vector2<f32>) {
        if self.use_first_person_camera {
            self.first_person_camera.rotate(movement);
            self.render_state.update_camera(&self.first_person_camera);
        } else {
            self.target_camera.rotate(movement);
            self.render_state.update_camera(&self.target_camera);
        }
    }

    pub fn go_near(&mut self) {
        if self.use_first_person_camera {
            return;
        }
        self.target_camera.go_near();
        self.render_state.update_camera(&self.target_camera);
    }

    pub fn go_away(&mut self) {
        if self.use_first_person_camera {
            return;
        }
        self.target_camera.go_away();
        self.render_state.update_camera(&self.target_camera);
    }

    pub fn move_camera(&mut self, dir: camera::Direction) {
        if !self.use_first_person_camera {
            return;
        }
        self.first_person_camera.go(dir);
        self.render_state.update_camera(&self.first_person_camera);
    }

    pub fn draw(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.render_state.draw(self.pipeline_mode)
    }

    /// returns true if paused after toggling
    pub fn toggle_pause(&mut self) -> bool {
        self.paused ^= true;
        self.paused
    }

    /// toggles the pipeline mode and returns the mode used after toggling
    pub fn toggle_pipeline(&mut self) -> render::PipelineMode {
        use render::PipelineMode::*;
        let next_mode = match self.pipeline_mode {
            Flat => Color,
            Color => Normal,
            Normal => Flat,
        };
        self.pipeline_mode = next_mode;
        self.pipeline_mode
    }

    pub fn toggle_camera(&mut self) -> bool {
        self.use_first_person_camera ^= true;
        if self.use_first_person_camera {
            self.render_state.update_camera(&self.first_person_camera);
        } else {
            self.render_state.update_camera(&self.target_camera);
        }
        self.use_first_person_camera
    }
}
