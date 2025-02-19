use std::fmt::Debug;

use super::resource::Texture;

/// Data format used to transfer camera information to the GPU.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraRaw {
    /// EXPECTS: any 4 values with last value != 0
    pub view_pos: [f32; 4],
    /// EXPECTS: any matrix in row-major with bottom right value != 0
    pub view_proj: [[f32; 4]; 4],
}
impl private::RawLayout for CameraRaw {}

/// Data format used to transfer light information to the GPU.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightRaw {
    /// EXPECTS: any 3 values
    pub position: [f32; 3],
    // WGSL uniforms expect 4 float / 16 bytes alignment
    // more info at: https://www.w3.org/TR/WGSL/#alignment-and-size
    pub _padding: u32,
    /// EXPECTS: any 3 values in interval [0, 1]
    pub color: [f32; 3],
    pub _padding2: u32,
}
impl private::RawLayout for LightRaw {}

/// Data format used to transfer information for a model vertex to the GPU.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexRaw {
    pub position: [f32; 3],
    /// EXPECTS: any 2 values within interval [0, 1]
    pub tex_coords: [f32; 2],
    /// EXPECTS: any 3-component vector v with |v| = 1
    pub normal: [f32; 3],
    /// EXPECTS: any 3-component vector v with |v| = 1 orthogonal to normal
    pub tangent: [f32; 3],
    /// EXPECTS: any 3-component vector v with |v| = 1 orthogonal to normal and tangent
    pub bitangent: [f32; 3],
}
impl private::RawLayout for VertexRaw {}

/// Data format used to transfer information for a model instance to the GPU.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    /// EXPECTS: any row-major matrix with bottom right != 0
    pub model: [[f32; 4]; 4],
    /// EXPECTS: same matrix as top left 3x3 matrix of `model`
    pub normal: [[f32; 3]; 3],
}
impl private::RawLayout for InstanceRaw {}

/// Data format used to transfer an vertex buffer for triangles and its triangle indices to the GPU.
pub struct TriangleBufferRaw {
    pub vertices: Box<[VertexRaw]>,
    /// EXPECTS: `indices.len()` is a multiple of 3 and any element is smaller than `data.len()`.
    pub indices: Box<[u32]>,
}
impl private::RawLayout for TriangleBufferRaw {}

/// Data format used to transfer an instance buffer to the GPU.
pub struct InstanceBufferRaw {
    pub instances: Box<[InstanceRaw]>,
}
impl private::RawLayout for InstanceBufferRaw {}

/// Data format used to transfer textures to the GPU.
pub struct TextureRaw<'a> {
    /// EXPECTS: Slice size depending on `size` and `format`.
    /// Multiply all values from `size` together, and multiply into it the byte size of the `format`.
    pub data: &'a [u8],
    pub format: wgpu::TextureFormat,
    pub size: wgpu::Extent3d,
}
impl<'a> private::RawLayoutRef<'a> for TextureRaw<'a> {}

/// Trait to enable generating data in a format that can be transfered to the GPU.
///
/// The allowed data formats are sealed so that only implemented formats are used which have
/// pre-defined associated uses.
///
/// This trait specifically is mainly used for data that is destined for Uniform buffers.
///
/// # Safety
/// Values copied over into raw format must match the expectation of the raw format
pub unsafe trait GpuTransfer {
    type Raw: private::RawLayout;
    fn to_raw(&self) -> Self::Raw;
}

/// Trait to enable generating data in a format that can be transfered to the GPU.
///
/// The allowed data formats are sealed so that only implemented formats are used which have
/// pre-defined associated uses.
///
/// This trait specifically is mainly used for data that is destined for variable length buffers
/// such as vertex buffers or texture buffers.
///
/// # Safety
/// Slice must match with resulting TextureFormat and sizes in Extend3d
pub unsafe trait GpuTransferRef<'a> {
    type Raw: private::RawLayoutRef<'a>;
    fn to_raw(&'a self) -> Self::Raw;
}

/// Helper trait to enable blanket implementation for creating textures
pub trait GpuTransferTexture {
    fn create_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: Option<&str>,
    ) -> Texture;
}

impl<T> GpuTransferTexture for T
where
    T: for<'a> GpuTransferRef<'a, Raw = TextureRaw<'a>>,
{
    fn create_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: Option<&str>,
    ) -> Texture {
        let TextureRaw { data, format, size } = self.to_raw();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
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
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * size.width),
                rows_per_image: Some(size.height),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Texture {
            texture,
            view,
            sampler,
        }
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

mod private {
    pub trait RawLayout {}
    // Perhaps because only a single layout (`TextureRaw`) uses this constraint trait, it is
    // getting marked as unused. Keep it for future additions.
    #[expect(unused)]
    pub trait RawLayoutRef<'a> {}
}
