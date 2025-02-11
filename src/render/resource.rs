use wgpu::util::DeviceExt;

use super::{CameraRaw, GPUTransfer, GPUTransferIndexed, InstanceRaw, LightRaw, VertexRaw};

pub struct TextureBindGroupLayout(pub wgpu::BindGroupLayout);
pub struct CameraBindGroupLayout(pub wgpu::BindGroupLayout);
pub struct LightBindGroupLayout(pub wgpu::BindGroupLayout);

pub struct Texture {
    #[allow(unused)]
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

pub struct MaterialBindingCN {
    pub bind_group: wgpu::BindGroup,
    #[allow(unused)]
    pub color: Texture,
    #[allow(unused)]
    pub normal: Texture,
}
pub struct CameraBinding {
    pub bind_group: wgpu::BindGroup,
    pub buffer: wgpu::Buffer,
}
pub struct LightBinding {
    pub bind_group: wgpu::BindGroup,
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

impl TextureBindGroupLayout {
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

impl MaterialBindingCN {
    pub fn new<C, N>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &TextureBindGroupLayout,
        color_texture: &C,
        normal_texture: &N,
        label: Option<&str>,
    ) -> Self
    where
        C: GPUTransferIndexed,
        N: GPUTransferIndexed,
    {
        let color = color_texture.create_texture(device, queue, label);
        let normal = normal_texture.create_texture(device, queue, label);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout: &layout.0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&color.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&normal.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&normal.sampler),
                },
            ],
        });

        Self {
            bind_group,
            color,
            normal,
        }
    }
}

impl CameraBinding {
    pub fn new<T: GPUTransfer<Raw = CameraRaw>>(
        device: &wgpu::Device,
        layout: &CameraBindGroupLayout,
        data: &T,
        label: Option<&str>,
    ) -> Self {
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

    pub fn update<T: GPUTransfer>(&self, queue: &wgpu::Queue, data: &T) {
        let raw_data = data.to_raw();
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[raw_data]));
    }
}

impl LightBinding {
    pub fn new<T: GPUTransfer<Raw = LightRaw>>(
        device: &wgpu::Device,
        layout: &LightBindGroupLayout,
        data: &T,
        label: Option<&str>,
    ) -> Self {
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

    pub fn update<T: GPUTransfer>(&self, queue: &wgpu::Queue, data: T) {
        let raw_data = data.to_raw();
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[raw_data]));
    }
}

impl MeshBuffer {
    pub fn new<V, T>(
        device: &wgpu::Device,
        vertices: &[V],
        indices: &[T],
        label: Option<&str>,
    ) -> Self
    where
        V: GPUTransfer<Raw = VertexRaw>,
        T: Copy + Into<(u32, u32, u32)>,
    {
        let raw_vertices = vertices
            .iter()
            .map(<V as GPUTransfer>::to_raw)
            .collect::<Vec<_>>();

        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label.map(|s| format!("{s:?} Vertex Buffer")).as_deref(),
            contents: bytemuck::cast_slice(&raw_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = indices
            .iter()
            .flat_map(|&t| {
                let (a, b, c) = t.into();
                [a, b, c]
            })
            .collect::<Vec<_>>();
        let num_indices = indices.len() as u32;
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
    pub fn new<T: GPUTransfer<Raw = InstanceRaw>>(
        device: &wgpu::Device,
        instances: &[T],
        label: Option<&str>,
    ) -> Self {
        let num_instances = instances.len() as u32;
        let raw_instances = instances
            .iter()
            .map(<T as GPUTransfer>::to_raw)
            .collect::<Vec<_>>();

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label.map(|s| format!("{s:?} Instance Buffer")).as_deref(),
            contents: bytemuck::cast_slice(&raw_instances),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self {
            buffer,
            num_instances,
        }
    }
}
