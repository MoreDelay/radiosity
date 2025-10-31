use std::{cell::RefCell, path::PathBuf, rc::Rc, sync::Arc};

use nalgebra as na;
use winit::window::Window;

use crate::{camera, light, model, render};

pub mod manager;

pub struct SceneState {
    render_state: Rc<RefCell<render::RenderState>>,
    use_color_map: bool,
    use_normal_map: bool,
    use_first_person_camera: bool,
    last_time: std::time::Instant,
    _storage: Rc<RefCell<model::Storage>>,
    draw_manager: manager::DrawManager,
    _instances: Vec<model::Instance>,
    target_camera: camera::TargetCamera,
    first_person_camera: camera::FirstPersonCamera,
    light: light::Light,
}

impl SceneState {
    pub fn new(window: Arc<Window>) -> Self {
        let (ctx, target) = pollster::block_on(render::create_render_instance(window));

        let pos = 1f32 * na::Vector3::new(0.0, 4.0, 7.0);
        let focus = na::Vector3::new(0.0, 0.0, 2.0);
        let distance = 1.;
        let frame = target.get_frame_dim();
        let target_camera = camera::TargetCamera::new(pos, focus, distance, frame);
        let direction = na::Unit::new_normalize(focus - pos);
        let first_person_camera = camera::FirstPersonCamera::new(pos, direction, frame);

        let light_pos = pos;
        let color = model::Color {
            r: 1.,
            g: 1.,
            b: 1.,
        };
        let light = light::Light::new(light_pos, color);

        let render_state = Rc::new(RefCell::new(
            render::RenderState::create(ctx, target, &target_camera.to_raw(), &light.to_raw())
                .unwrap(),
        ));

        let mut draw_manager = manager::DrawManager::new(Rc::clone(&render_state));

        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // let model_path = root.join("resources/sibenik/sibenik.obj");
        let model_path = root.join("resources/cube/cube.obj");

        let storage = Rc::new(RefCell::new(model::Storage::new()));
        let model_root = model_path.parent().expect("model file lies in a directory");
        let mut mtl_manager = model::parser::SimpleMtlManager::new(model_root.to_path_buf());
        let parsed_obj =
            model::parser::obj::load_obj(&model_path, &mut mtl_manager).expect("worked above");
        let model_index = storage
            .borrow_mut()
            .store_obj(parsed_obj, mtl_manager.extract_list());

        const NUM_INSTANCES_PER_ROW: usize = 5;
        const SPACE_BETWEEN: f32 = 3.;
        let instances = (0..NUM_INSTANCES_PER_ROW)
            .flat_map(|z| {
                (0..NUM_INSTANCES_PER_ROW).map(move |x| {
                    let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.);
                    let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.);

                    let position = na::Vector3::new(x, 0.0, z);
                    let rotation = if position == na::Vector3::zeros() {
                        na::UnitQuaternion::from_axis_angle(&na::Vector3::z_axis(), 0.)
                    } else {
                        na::UnitQuaternion::from_axis_angle(
                            &na::Unit::new_normalize(position),
                            std::f32::consts::FRAC_PI_8,
                        )
                    };
                    model::Instance { position, rotation }
                })
            })
            .collect::<Vec<_>>();
        // let instances = vec![model::Instance {
        //     position: na::Vector3::zeros(),
        //     rotation: na::UnitQuaternion::from_axis_angle(&na::Vector3::z_axis(), 0.),
        // }];

        let instance_index = render_state
            .borrow_mut()
            .add_instance_buffer(&instances, Some("Grid"));

        draw_manager.add_mesh(&storage.borrow(), model_index, instance_index, Some("Cube"));

        SceneState {
            render_state,
            use_color_map: false,
            use_normal_map: false,
            use_first_person_camera: false,
            last_time: std::time::Instant::now(),
            _storage: storage,
            draw_manager,
            _instances: instances,
            target_camera,
            first_person_camera,
            light,
        }
    }

    pub fn resize_window(&mut self, new_size: Option<winit::dpi::PhysicalSize<u32>>) {
        if let Some(winit::dpi::PhysicalSize { width, height }) = new_size
            && (width > 0 && height > 0)
        {
            let frame_size = camera::FrameDim(width, height);
            self.target_camera.update_frame(frame_size);
            self.first_person_camera.update_frame(frame_size);
            if self.use_first_person_camera {
                self.render_state
                    .borrow_mut()
                    .update_camera(&self.first_person_camera.to_raw());
            } else {
                self.render_state
                    .borrow_mut()
                    .update_camera(&self.target_camera.to_raw());
            }
        }

        self.render_state.borrow_mut().resize(new_size);
    }

    pub fn step(&mut self) {
        const TARGET_FPS: f64 = 150.;
        let minimum_elapsed = std::time::Duration::from_secs_f64(1. / TARGET_FPS);

        let elapsed = self.last_time.elapsed();
        if elapsed < minimum_elapsed {
            let sleep_duration = minimum_elapsed - elapsed;
            std::thread::sleep(sleep_duration);
        }
        let epsilon = self.last_time.elapsed().as_secs_f32();
        // let fps = 1. / epsilon.as_secs_f32();
        // println!("FPS: {fps}");
        self.last_time = std::time::Instant::now();
        let moved = self.light.step(epsilon);
        if moved {
            self.render_state
                .borrow_mut()
                .update_light(&self.light.to_raw());
        }

        if self.use_first_person_camera {
            let moved = self.first_person_camera.step(epsilon);
            if moved {
                self.render_state
                    .borrow_mut()
                    .update_camera(&self.first_person_camera.to_raw());
            }
        }
    }

    pub fn drag_camera(&mut self, movement: na::Vector2<f32>) {
        if self.use_first_person_camera {
            self.first_person_camera.rotate(movement);
            self.render_state
                .borrow_mut()
                .update_camera(&self.first_person_camera.to_raw());
        } else {
            self.target_camera.rotate(movement);
            self.render_state
                .borrow_mut()
                .update_camera(&self.target_camera.to_raw());
        }
    }

    pub fn go_near(&mut self) {
        if self.use_first_person_camera {
            return;
        }
        self.target_camera.go_near();
        self.render_state
            .borrow_mut()
            .update_camera(&self.target_camera.to_raw());
    }

    pub fn go_away(&mut self) {
        if self.use_first_person_camera {
            return;
        }
        self.target_camera.go_away();
        self.render_state
            .borrow_mut()
            .update_camera(&self.target_camera.to_raw());
    }

    pub fn set_movement(&mut self, dir: camera::DirectionKey, active: bool) {
        self.first_person_camera.set_movement(dir, active);
    }

    pub fn draw(&mut self) -> Result<(), wgpu::SurfaceError> {
        let caps = render::PhongCapabilites {
            color_map: self.use_color_map,
            normal_map: self.use_normal_map,
        };
        let draw_world = self.draw_manager.create_draw(caps);
        self.render_state.borrow_mut().draw(draw_world)
    }

    /// returns true if paused after toggling
    pub fn toggle_pause(&mut self) -> bool {
        self.light.toggle_pause()
    }

    pub fn toggle_color_map(&mut self) -> bool {
        self.use_color_map = !self.use_color_map;
        self.use_color_map
    }

    pub fn toggle_normal_map(&mut self) -> bool {
        self.use_normal_map = !self.use_normal_map;
        self.use_normal_map
    }

    pub fn toggle_camera(&mut self) -> bool {
        self.use_first_person_camera ^= true;
        if self.use_first_person_camera {
            self.render_state
                .borrow_mut()
                .update_camera(&self.first_person_camera.to_raw());
        } else {
            self.render_state
                .borrow_mut()
                .update_camera(&self.target_camera.to_raw());
        }
        self.use_first_person_camera
    }
}
