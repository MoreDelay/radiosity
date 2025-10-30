use std::fmt::Debug;

use crate::render::GpuContext;

use super::resource::Texture;

/// Data format used to transfer camera information to the GPU.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraRaw {
    /// finite homogeneous position in right-handed coordinates (w != 0)
    pub view_pos: [f32; 3],
    // WGSL uniforms expect 4 float / 16 bytes alignment
    // more info at: https://www.w3.org/TR/WGSL/#alignment-and-size
    pub _padding: u32,
    /// column-major homogeneous transformation matrix
    pub view_proj: [[f32; 4]; 4],
}

/// Data format used to transfer light information to the GPU.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightRaw {
    /// position in right-handed coordinates
    pub position: [f32; 3],
    /// far plane of light
    pub max_dist: f32,
    /// rgb values in interval [0, 1]
    pub color: [f32; 3],
    // WGSL uniforms expect 4 float / 16 bytes alignment
    // more info at: https://www.w3.org/TR/WGSL/#alignment-and-size
    pub _padding: u32,
}

/// Data format used to transfer light information to the GPU.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PhongRaw {
    /// rgb values in interval [0, 1]
    pub specular_color: [f32; 3],
    pub specular_exponent: f32,
    /// rgb values in interval [0, 1]
    pub diffuse_color: [f32; 3],
    // WGSL uniforms expect 4 float / 16 bytes alignment
    // more info at: https://www.w3.org/TR/WGSL/#alignment-and-size
    pub _padding1: u32,
    /// rgb values in interval [0, 1]
    pub ambient_color: [f32; 3],
    pub _padding2: u32,
}

/// Data format used to transfer information for a model vertex to the GPU.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexRaw {
    /// position in right-handed coordinates
    pub position: [f32; 3],
    /// uv coordinates into texture in interval [0, 1]
    pub tex_coords: [f32; 2],
    /// surface normal of vector with unit length
    pub normal: [f32; 3],
    /// vector tangent to surface with unit length
    pub tangent: [f32; 3],
    /// vector tangent to surface with unit length and orthogonal to tangent
    pub bitangent: [f32; 3],
}

/// Data format used to transfer information for a model instance to the GPU.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    /// column-major homogeneous transformation matrix in world space
    pub model: [[f32; 4]; 4],
    /// column-major rotation transformation matrix, 3x3 submatrix of `model`
    pub normal: [[f32; 3]; 3],
}

/// Data format used to transfer textures to the GPU.
pub struct TextureRaw {
    /// Slice size depending on `size` and `format`.
    /// Multiply all values from `size` together, and multiply into it the byte size of the `format`.
    pub data: Box<[u8]>,
    pub format: wgpu::TextureFormat,
    pub size: wgpu::Extent3d,
}

impl TextureRaw {
    pub fn create_texture(&self, ctx: &GpuContext, label: Option<&str>) -> Texture {
        let &TextureRaw {
            ref data,
            format,
            size,
        } = self;

        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            // specified format above supported by default, only additional view formats here
            view_formats: &[],
        });

        // load image (on CPU) into texture (on GPU) by issuing command over queue
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * size.width),
                rows_per_image: Some(size.height),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Texture {
            _texture: texture,
            view,
            sampler,
        }
    }
}

pub fn position_desc() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 3 * std::mem::size_of::<f32>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        }],
    }
}

pub fn tex_coord_desc() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 2 * std::mem::size_of::<f32>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 0,
            shader_location: 1,
        }],
    }
}

pub fn normal_desc() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 3 * std::mem::size_of::<f32>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 2,
        }],
    }
}

pub fn tangent_desc() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 3 * std::mem::size_of::<f32>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 3,
        }],
    }
}

pub fn bitangent_desc() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 3 * std::mem::size_of::<f32>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 4,
        }],
    }
}

impl VertexRaw {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VertexRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: std::mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: std::mem::size_of::<[f32; 11]>() as wgpu::BufferAddress,
                    shader_location: 4,
                },
            ],
        }
    }
}

impl InstanceRaw {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // model
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}
