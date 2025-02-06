use wgpu::util::DeviceExt;

#[derive(Debug, Copy, Clone)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[derive(Debug, Copy, Clone)]
pub struct LightParams {
    pub pos: cgmath::Point3<f32>,
    pub color: Color,
}

pub struct Light {
    pub params: LightParams,
    buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    pub position: [f32; 3],
    // WGSL uniforms expect 4 float / 16 bytes alignment
    // more info at: https://www.w3.org/TR/WGSL/#alignment-and-size
    _padding: u32,
    pub color: [f32; 3],
    _padding2: u32,
}

impl Light {
    pub fn new(device: &wgpu::Device, params: LightParams) -> Self {
        let light_uniform = LightUniform::new(params);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light VB"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self { params, buffer }
    }

    pub fn update(&mut self, queue: &wgpu::Queue, params: LightParams) {
        self.params = params;
        let light_uniform = LightUniform::new(params);
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[light_uniform]));
    }

    pub fn get_binding_resource(&self) -> wgpu::BindingResource<'_> {
        self.buffer.as_entire_binding()
    }
}

impl LightParams {
    pub fn new(pos: cgmath::Point3<f32>, color: Color) -> Self {
        Self { pos, color }
    }
}

impl LightUniform {
    pub fn new(params: LightParams) -> Self {
        let position = params.pos.into();
        let color = params.color.into();
        Self {
            position,
            _padding: 0,
            color,
            _padding2: 0,
        }
    }
}

impl From<Color> for [f32; 4] {
    fn from(val: Color) -> Self {
        let Color { r, g, b, a } = val;
        [
            r as f32 / 255.,
            g as f32 / 255.,
            b as f32 / 255.,
            a as f32 / 255.,
        ]
    }
}

impl From<Color> for [f32; 3] {
    fn from(val: Color) -> Self {
        let Color { r, g, b, .. } = val;
        [r as f32 / 255., g as f32 / 255., b as f32 / 255.]
    }
}
