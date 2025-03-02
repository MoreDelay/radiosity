use crate::render::layout::{CameraRaw, GpuTransfer};

#[derive(Debug, Copy, Clone)]
pub struct FrameDim(pub u32, pub u32);

pub struct Camera {
    pos: cgmath::Point3<f32>,
    dir: cgmath::Vector3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
    frame: FrameDim,
}

impl Camera {
    pub fn new(pos: cgmath::Point3<f32>, target: cgmath::Point3<f32>, frame: FrameDim) -> Self {
        let dir = target - pos;
        let up = cgmath::Vector3::unit_y();

        let aspect = 16. / 9.;
        let fovy = 45.0;
        let znear = 0.1;
        let zfar = 100.0;

        Self {
            pos,
            dir,
            up,
            aspect,
            fovy,
            znear,
            zfar,
            frame,
        }
    }

    pub fn update_frame(&mut self, frame_size: FrameDim) {
        self.frame = frame_size;
        let FrameDim(width, height) = frame_size;
        self.aspect = width as f32 / height as f32;
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

// Safety: CameraRaw restricts view_pos to last value != 0
// and view_proj to row major matrix with bottom right != 0
unsafe impl GpuTransfer for Camera {
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
