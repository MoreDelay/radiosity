use cgmath::{ElementWise, EuclideanSpace, InnerSpace, Rotation, Rotation2, Rotation3};

use crate::render::layout::{CameraRaw, GpuTransfer};

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
        let forward = forward.normalize();
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

        // correct for expected distance (precision errors)
        let from_target = pos - self.target;
        let pos = cgmath::Point3::from_vec(from_target.normalize() * self.distance);
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

// Safety: CameraRaw restricts view_pos to last value != 0
// and view_proj to row major matrix with bottom right != 0
unsafe impl GpuTransfer for TargetCamera {
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
