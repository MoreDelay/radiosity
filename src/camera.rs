use wgpu::util::DeviceExt;

#[derive(Debug, Copy, Clone)]
pub struct FrameDim(u32, u32);

pub struct CameraParams {
    eye: cgmath::Point3<f32>,
    dir: cgmath::Vector3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
    frame: FrameDim,
}

pub struct Camera {
    pub params: CameraParams,
    buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_pos: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

impl CameraParams {
    pub fn new(
        eye: cgmath::Point3<f32>,
        dir: cgmath::Vector3<f32>,
        up: cgmath::Vector3<f32>,
        fovy: f32,
        znear: f32,
        zfar: f32,
        frame: FrameDim,
    ) -> Self {
        let aspect = 16. / 9.;

        Self {
            eye,
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
        let view = cgmath::Matrix4::look_to_rh(self.eye, self.dir, self.up);
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

impl Camera {
    pub fn new(device: &wgpu::Device, params: CameraParams) -> Self {
        let camera_uniform = CameraUniform::new(&params);

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self { params, buffer }
    }

    pub fn get_binding_resource(&self) -> wgpu::BindingResource<'_> {
        self.buffer.as_entire_binding()
    }

    pub fn update_frame(&mut self, queue: &wgpu::Queue, frame_size: FrameDim) {
        self.params.update_frame(frame_size);
        let camera_uniform = CameraUniform::new(&self.params);
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[camera_uniform]));
    }
}

impl CameraUniform {
    pub fn new(params: &CameraParams) -> Self {
        let view_pos = params.eye.to_homogeneous().into();
        let view_proj = params.build_view_projection_matrix().into();

        Self {
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
