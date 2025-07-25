use cgmath::{ElementWise, EuclideanSpace, InnerSpace, Rotation, Rotation2, Rotation3};

use crate::render::{CameraRaw, GpuTransfer};

#[derive(Debug, Copy, Clone)]
pub struct FrameDim(pub u32, pub u32);

pub struct TargetCamera {
    pos: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    distance: f32,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
    frame: FrameDim,
}

#[derive(Debug, Default)]
struct MovementState {
    forward: bool,
    backward: bool,
    right: bool,
    left: bool,
}

pub struct FirstPersonCamera {
    pos: cgmath::Point3<f32>,
    dir: cgmath::Vector3<f32>,
    speed: f32,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
    frame: FrameDim,
    movement_state: MovementState,
}

#[derive(Copy, Clone, Debug)]
pub enum DirectionKey {
    W,
    A,
    S,
    D,
}

impl TargetCamera {
    pub fn new(
        pos: cgmath::Point3<f32>,
        target: cgmath::Point3<f32>,
        distance: f32,
        frame: FrameDim,
    ) -> Self {
        let up = cgmath::Vector3::unit_y();

        let from_target = pos - target;
        let pos = cgmath::Point3::from_vec(from_target.normalize() * distance);
        let pos = pos.add_element_wise(target);

        let FrameDim(width, height) = frame;
        let aspect = width as f32 / height as f32;
        let fovy = 45.0;
        let znear = 0.1;
        let zfar = 100.0;

        Self {
            pos,
            target,
            distance,
            up,
            aspect,
            fovy,
            znear,
            zfar,
            frame,
        }
    }

    pub fn update_frame(&mut self, frame: FrameDim) {
        self.frame = frame;
        let FrameDim(width, height) = frame;
        self.aspect = width as f32 / height as f32;
    }

    pub fn rotate(&mut self, movement: cgmath::Vector2<f32>) {
        let rot90 = cgmath::Rad(0.5 * std::f32::consts::PI);
        let rot90 = cgmath::Basis2::from_angle(rot90);
        let cgmath::Vector2 { x, y } = rot90.rotate_vector(movement);

        let forward = (self.target - self.pos).normalize();
        let right = forward.cross(self.up).normalize();
        let up = right.cross(forward);

        // image coordinate system starts at top left corner
        // and y gets bigger when going down, so we need to invert y
        let rotation_axis = (x * right - y * up).normalize();

        let angle = cgmath::Rad(movement.magnitude()) / 100.;
        let rotation = cgmath::Basis3::from_axis_angle(rotation_axis, angle);

        // rotate around target, so move target to origin and back again
        let pos = self.pos.sub_element_wise(self.target);
        let pos = rotation.rotate_point(pos);
        let pos = pos.add_element_wise(self.target);

        // correct for expected distance (precision errors, overshoot correction)
        let from_target = (pos - self.target).normalize();
        let pos = cgmath::Point3::from_vec(from_target * self.distance);
        let pos = pos.add_element_wise(self.target);

        self.pos = pos;
    }

    pub fn go_near(&mut self) {
        const MIN_DISTANCE: f32 = 0.1;
        // use exponential / logarithmic scale for scolling
        self.distance = MIN_DISTANCE.max(self.distance - self.distance / 10.);

        let from_target = self.pos - self.target;
        let pos = cgmath::Point3::from_vec(from_target.normalize() * self.distance);
        self.pos = pos.add_element_wise(self.target);
    }

    pub fn go_away(&mut self) {
        const MAX_DISTANCE: f32 = 1000.;
        // use exponential / logarithmic scale for scolling
        self.distance = MAX_DISTANCE.min(self.distance + self.distance / 10.);

        let from_target = self.pos - self.target;
        let pos = cgmath::Point3::from_vec(from_target.normalize() * self.distance);
        self.pos = pos.add_element_wise(self.target);
    }

    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let dir = self.target - self.pos;
        let view = cgmath::Matrix4::look_to_rh(self.pos, dir, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        // wgpu uses DirectX / Metal coordinates
        // there it is assumed that x,y are in range [-1., 1.] and z is in range of [0., 1.]
        // cgmath uses OpenGL coordinates that assumes [-1., 1.] for all axes
        // that means we need an affine transform to fix the z-axis
        #[rustfmt::skip]
        const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.5,
            0.0, 0.0, 0.0, 1.0,
        );
        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}

impl GpuTransfer for TargetCamera {
    type Raw = CameraRaw;
    fn to_raw(&self) -> Self::Raw {
        let view_pos = self.pos.to_homogeneous().into();
        let view_proj = self.build_view_projection_matrix().into();

        Self::Raw {
            view_pos,
            view_proj,
        }
    }
}

impl FirstPersonCamera {
    pub fn new(pos: cgmath::Point3<f32>, dir: cgmath::Vector3<f32>, frame: FrameDim) -> Self {
        let dir = dir.normalize();
        let up = cgmath::Vector3::unit_y();
        let speed = 20.;

        let FrameDim(width, height) = frame;
        let aspect = width as f32 / height as f32;
        let fovy = 45.0;
        let znear = 0.1;
        let zfar = 100.0;

        let movement_state = MovementState::default();

        Self {
            pos,
            dir,
            speed,
            up,
            aspect,
            fovy,
            znear,
            zfar,
            frame,
            movement_state,
        }
    }

    pub fn update_frame(&mut self, frame: FrameDim) {
        self.frame = frame;
        let FrameDim(width, height) = frame;
        self.aspect = width as f32 / height as f32;
    }

    pub fn step(&mut self, epsilon: f32) -> bool {
        let distance = self.speed * epsilon;
        let moved_ws = match (self.movement_state.forward, self.movement_state.backward) {
            (true, false) => {
                self.do_move(DirectionKey::W, distance);
                true
            }
            (false, true) => {
                self.do_move(DirectionKey::S, distance);
                true
            }
            _ => false,
        };
        let moved_ad = match (self.movement_state.left, self.movement_state.right) {
            (true, false) => {
                self.do_move(DirectionKey::A, distance);
                true
            }
            (false, true) => {
                self.do_move(DirectionKey::D, distance);
                true
            }
            _ => false,
        };
        moved_ws | moved_ad
    }

    pub fn set_movement(&mut self, dir: DirectionKey, active: bool) {
        match dir {
            DirectionKey::W => self.movement_state.forward = active,
            DirectionKey::A => self.movement_state.left = active,
            DirectionKey::S => self.movement_state.backward = active,
            DirectionKey::D => self.movement_state.right = active,
        };
    }

    fn do_move(&mut self, direction: DirectionKey, distance: f32) {
        let right = self.dir.cross(self.up).normalize();

        let delta = match direction {
            DirectionKey::W => self.dir * distance,
            DirectionKey::A => -right * distance,
            DirectionKey::S => -self.dir * distance,
            DirectionKey::D => right * distance,
        };

        self.pos += delta;
    }

    pub fn rotate(&mut self, movement: cgmath::Vector2<f32>) {
        let rot90 = cgmath::Rad(0.5 * std::f32::consts::PI);
        let rot90 = cgmath::Basis2::from_angle(rot90);
        let cgmath::Vector2 { x, y } = rot90.rotate_vector(movement);

        let right = self.dir.cross(self.up).normalize();
        let up = right.cross(self.dir);

        // image coordinate system starts at top left corner
        // and y gets bigger when going down, so we need to invert y
        let rotation_axis = (x * right - y * up).normalize();

        let angle = cgmath::Rad(movement.magnitude()) / 100.;
        let rotation = cgmath::Basis3::from_axis_angle(rotation_axis, angle);

        self.dir = rotation.rotate_vector(self.dir).normalize();
    }

    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_to_rh(self.pos, self.dir, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        // wgpu uses DirectX / Metal coordinates
        // there it is assumed that x,y are in range [-1., 1.] and z is in range of [0., 1.]
        // cgmath uses OpenGL coordinates that assumes [-1., 1.] for all axes
        // that means we need an affine transform to fix the z-axis
        #[rustfmt::skip]
        const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.5,
            0.0, 0.0, 0.0, 1.0,
        );
        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}

impl GpuTransfer for FirstPersonCamera {
    type Raw = CameraRaw;
    fn to_raw(&self) -> Self::Raw {
        let view_pos = self.pos.to_homogeneous().into();
        let view_proj = self.build_view_projection_matrix().into();

        Self::Raw {
            view_pos,
            view_proj,
        }
    }
}

impl From<(u32, u32)> for FrameDim {
    fn from((width, height): (u32, u32)) -> Self {
        Self(width, height)
    }
}
