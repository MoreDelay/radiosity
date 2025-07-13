use std::{num::NonZeroU64, ops::Deref};

use static_assertions::const_assert_ne;
use wgpu::util::DeviceExt;

use crate::render::layout::PhongRaw;

use super::{
    CameraRaw, GpuTransfer, GpuTransferTexture, InstanceBufferRaw, LightRaw, TriangleBufferRaw,
};

pub struct NormalTextureBindGroupLayout(pub wgpu::BindGroupLayout);
pub struct ColorTextureBindGroupLayout(pub wgpu::BindGroupLayout);
pub struct FlatTextureBindGroupLayout(pub wgpu::BindGroupLayout);

pub enum TextureBindGroupLayout {
    Flat(FlatTextureBindGroupLayout),
    Color(ColorTextureBindGroupLayout),
    Normal(NormalTextureBindGroupLayout),
}

pub struct CameraBindGroupLayout(pub wgpu::BindGroupLayout);
pub struct LightBindGroupLayout(pub wgpu::BindGroupLayout);
pub struct PhongBindGroupLayout(pub wgpu::BindGroupLayout);

pub struct Texture {
    #[expect(unused)]
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

pub struct MaterialBinding {
    pub bind_group: wgpu::BindGroup,
    #[expect(unused)]
    pub color: Option<Texture>,
    #[expect(unused)]
    pub normal: Option<Texture>,
}
pub struct CameraBinding {
    pub bind_group: wgpu::BindGroup,
    pub buffer: wgpu::Buffer,
}
pub struct LightBinding {
    pub bind_group: wgpu::BindGroup,
    pub buffer: wgpu::Buffer,
}
pub struct PhongBinding {
    pub bind_group: wgpu::BindGroup,
    #[expect(dead_code)]
    pub buffer: wgpu::Buffer,
}

pub struct MeshBuffer {
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub num_indices: u32,
}
pub struct InstanceBuffer {
    pub buffer: wgpu::Buffer,
    pub num_instances: u32,
}

pub trait HasColor {}
pub trait HasNormal {}

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // technically, we do not need a sampler, but required by our current struct
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            // compare function would be used as a filter / modulation for some other operation
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }
}

impl Deref for FlatTextureBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Deref for ColorTextureBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Deref for NormalTextureBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for TextureBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        match self {
            TextureBindGroupLayout::Flat(layout) => layout,
            TextureBindGroupLayout::Color(layout) => layout,
            TextureBindGroupLayout::Normal(layout) => layout,
        }
    }
}

impl HasColor for ColorTextureBindGroupLayout {}
impl HasColor for NormalTextureBindGroupLayout {}

impl HasNormal for NormalTextureBindGroupLayout {}

#[derive(Copy, Clone, Debug)]
pub enum TextureAvailability {
    None,
    Color,
    NormalAndColor,
}

impl TextureBindGroupLayout {
    pub fn new(device: &wgpu::Device, availability: TextureAvailability) -> Self {
        match availability {
            TextureAvailability::None => Self::Flat(FlatTextureBindGroupLayout::new(device)),
            TextureAvailability::Color => Self::Color(ColorTextureBindGroupLayout::new(device)),
            TextureAvailability::NormalAndColor => {
                Self::Normal(NormalTextureBindGroupLayout::new(device))
            }
        }
    }
}

impl NormalTextureBindGroupLayout {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        Self(layout)
    }
}

impl ColorTextureBindGroupLayout {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        Self(layout)
    }
}

impl FlatTextureBindGroupLayout {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_bind_group_layout"),
            entries: &[],
        });
        Self(layout)
    }
}

impl CameraBindGroupLayout {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, // for multiple datasets varying in size
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        Self(layout)
    }
}

impl LightBindGroupLayout {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        Self(layout)
    }
}

impl PhongBindGroupLayout {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        Self(layout)
    }
}

impl MaterialBinding {
    pub fn new<T, N>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &TextureBindGroupLayout,
        color_texture: Option<&T>,
        normal_texture: Option<&N>,
        label: Option<&str>,
    ) -> Self
    where
        T: GpuTransferTexture,
        N: GpuTransferTexture,
    {
        let color = color_texture.map(|t| t.create_texture(device, queue, label));
        let color_entries = color.as_ref().map(|t| {
            [
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&t.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&t.sampler),
                },
            ]
        });

        let normal = normal_texture.map(|t| t.create_texture(device, queue, label));
        let normal_entries = normal.as_ref().map(|t| {
            [
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&t.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&t.sampler),
                },
            ]
        });

        let entries: Vec<_> = [color_entries, normal_entries]
            .into_iter()
            .flatten() // filter missing materials
            .flatten() // flatten [view, sampler] slices to single level
            .collect();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout,
            entries: &entries,
        });

        Self {
            bind_group,
            color,
            normal,
        }
    }
}

impl CameraBinding {
    pub fn new<C>(
        device: &wgpu::Device,
        layout: &CameraBindGroupLayout,
        data: &C,
        label: Option<&str>,
    ) -> Self
    where
        C: GpuTransfer<Raw = CameraRaw>,
    {
        let raw_data = data.to_raw();

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label.map(|s| format!("{s} Camera Buffer")).as_deref(),
            contents: bytemuck::cast_slice(&[raw_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: label.map(|s| format!("{s} Camera Bind Group")).as_deref(),
            layout: &layout.0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self { bind_group, buffer }
    }

    pub fn update<C>(&self, queue: &wgpu::Queue, data: &C)
    where
        C: GpuTransfer<Raw = CameraRaw>,
    {
        // make sure size unwrap never panics
        const_assert_ne!(std::mem::size_of::<CameraRaw>(), 0);
        let size = NonZeroU64::try_from(std::mem::size_of::<CameraRaw>() as u64).unwrap();

        let mut view = queue.write_buffer_with(&self.buffer, 0, size).unwrap();
        let raw_data = data.to_raw();
        view.copy_from_slice(bytemuck::cast_slice(&[raw_data]));
    }
}

impl LightBinding {
    pub fn new<L>(
        device: &wgpu::Device,
        layout: &LightBindGroupLayout,
        data: &L,
        label: Option<&str>,
    ) -> Self
    where
        L: GpuTransfer<Raw = LightRaw>,
    {
        let raw_data = data.to_raw();

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label.map(|s| format!("{s} Light Buffer")).as_deref(),
            contents: bytemuck::cast_slice(&[raw_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: label.map(|s| format!("{s} Light Bind Group")).as_deref(),
            layout: &layout.0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self { bind_group, buffer }
    }

    pub fn update<L>(&self, queue: &wgpu::Queue, data: &L)
    where
        L: GpuTransfer<Raw = LightRaw>,
    {
        // make sure size unwrap never panics
        const_assert_ne!(std::mem::size_of::<LightRaw>(), 0);
        let size = NonZeroU64::try_from(std::mem::size_of::<LightRaw>() as u64).unwrap();

        let mut view = queue.write_buffer_with(&self.buffer, 0, size).unwrap();
        let raw_data = data.to_raw();
        view.copy_from_slice(bytemuck::cast_slice(&[raw_data]));
    }
}

impl PhongBinding {
    pub fn new<P>(
        device: &wgpu::Device,
        layout: &PhongBindGroupLayout,
        data: &P,
        label: Option<&str>,
    ) -> Self
    where
        P: GpuTransfer<Raw = PhongRaw>,
    {
        let raw_data = data.to_raw();

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label.map(|s| format!("{s} Phong Buffer")).as_deref(),
            contents: bytemuck::cast_slice(&[raw_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: label.map(|s| format!("{s} Phong Bind Group")).as_deref(),
            layout: &layout.0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self { bind_group, buffer }
    }

    // pub fn update<P>(&self, queue: &wgpu::Queue, data: &P)
    // where
    //     P: GpuTransfer<Raw = PhongRaw>,
    // {
    //     // make sure size unwrap never panics
    //     const_assert_ne!(std::mem::size_of::<LightRaw>(), 0);
    //     let size = NonZeroU64::try_from(std::mem::size_of::<LightRaw>() as u64).unwrap();
    //
    //     let mut view = queue.write_buffer_with(&self.buffer, 0, size).unwrap();
    //     let raw_data = data.to_raw();
    //     view.copy_from_slice(bytemuck::cast_slice(&[raw_data]));
    // }
}

impl MeshBuffer {
    pub fn new<M>(device: &wgpu::Device, mesh: &M, label: Option<&str>) -> Self
    where
        M: GpuTransfer<Raw = TriangleBufferRaw>,
    {
        let TriangleBufferRaw { vertices, indices } = mesh.to_raw();
        // let raw_vertices = vertices.iter().map(|v| v.to_raw()).collect::<Vec<_>>();

        let num_indices = indices.len() as u32;
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label.map(|s| format!("{s:?} Vertex Buffer")).as_deref(),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label.map(|s| format!("{s:?} Index Buffer")).as_deref(),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertices,
            indices,
            num_indices,
        }
    }
}

impl InstanceBuffer {
    pub fn new<I>(device: &wgpu::Device, instances: &I, label: Option<&str>) -> Self
    where
        I: GpuTransfer<Raw = InstanceBufferRaw>,
    {
        let InstanceBufferRaw { instances } = instances.to_raw();
        let num_instances = instances.len() as u32;

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label.map(|s| format!("{s:?} Instance Buffer")).as_deref(),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self {
            buffer,
            num_instances,
        }
    }
}
