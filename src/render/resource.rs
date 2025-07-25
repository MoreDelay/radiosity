use std::{
    num::{NonZeroU32, NonZeroU64},
    ops::Deref,
};

use static_assertions::const_assert_ne;
use wgpu::util::DeviceExt;

use crate::render::layout::PhongRaw;

use super::{
    CameraRaw, GpuTransfer, GpuTransferTexture, InstanceBufferRaw, LightRaw, TriangleBufferRaw,
};

#[derive(Debug, Clone, Copy)]
pub struct TextureDims {
    pub width: NonZeroU32,
    pub height: NonZeroU32,
}

pub struct TextureBindGroupLayout(wgpu::BindGroupLayout);

pub struct CameraBindGroupLayout(pub wgpu::BindGroupLayout);
pub struct LightBindGroupLayout(pub wgpu::BindGroupLayout);
pub struct ShadowUniformBindGroupLayout(pub wgpu::BindGroupLayout);
pub struct ShadowTextureBindGroupLayout(pub wgpu::BindGroupLayout);
pub struct PhongBindGroupLayout(pub wgpu::BindGroupLayout);

pub struct Texture {
    #[expect(unused)]
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

pub struct MaterialBindings {
    pub phong_binding: PhongBinding,
    pub color: Option<(wgpu::BindGroup, Texture)>,
    pub normal: Option<(wgpu::BindGroup, Texture)>,
}
pub struct CameraBinding {
    pub bind_group: wgpu::BindGroup,
    pub buffer: wgpu::Buffer,
}
pub struct LightBinding {
    pub bind_group: wgpu::BindGroup,
    pub buffer: wgpu::Buffer,
}
pub struct ShadowBinding {
    pub uniform_bind_group: wgpu::BindGroup,
    pub texture_bind_group: wgpu::BindGroup,
    pub buffer: wgpu::Buffer,
    pub texture: Texture,
}
pub struct PhongBinding {
    pub bind_group: wgpu::BindGroup,
    #[expect(unused)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshBufferIndex {
    index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstanceBufferIndex {
    index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialBindingIndex {
    index: usize,
}

pub struct ModelResourceStorage {
    mesh_buffers: Vec<MeshBuffer>,
    instance_buffers: Vec<InstanceBuffer>,
    material_bindings: Vec<MaterialBindings>,
}

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn create_depth(device: &wgpu::Device, dims: TextureDims, label: Option<&str>) -> Self {
        let TextureDims { width, height } = dims;
        let size = wgpu::Extent3d {
            width: width.get(),
            height: height.get(),
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            texture,
            view,
            sampler,
        }
    }
}

impl TextureBindGroupLayout {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("NormalTextureBindGroupLayout"),
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

impl Deref for TextureBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl CameraBindGroupLayout {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CameraBindGroupLayout"),
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
            label: Some("LightBindGroupLayout"),
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

impl ShadowUniformBindGroupLayout {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShadowBindGroupLayout"),
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

impl ShadowTextureBindGroupLayout {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShadowBindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        Self(layout)
    }
}

impl PhongBindGroupLayout {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PhongBindGroupLayout"),
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

impl MaterialBindings {
    #[expect(clippy::too_many_arguments)]
    pub fn new<P, T, N>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        phong_layout: &PhongBindGroupLayout,
        texture_layout: &TextureBindGroupLayout,
        phong: &P,
        color_texture: Option<&T>,
        normal_texture: Option<&N>,
        label: Option<&str>,
    ) -> Self
    where
        P: GpuTransfer<Raw = PhongRaw>,
        T: GpuTransferTexture,
        N: GpuTransferTexture,
    {
        let color = color_texture.map(|t| {
            let texture_label = label.map(|s| format!("{s}-ColorTexture"));
            let texture = t.create_texture(device, queue, texture_label.as_deref());
            let entries = [
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ];
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label,
                layout: texture_layout.deref(),
                entries: &entries,
            });
            (bind_group, texture)
        });

        let normal = normal_texture.map(|t| {
            let texture_label = label.map(|s| format!("{s}-NormalTexture"));
            let texture = t.create_texture(device, queue, texture_label.as_deref());
            let entries = [
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ];
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label,
                layout: texture_layout.deref(),
                entries: &entries,
            });
            (bind_group, texture)
        });

        let phong_label = label.map(|s| format!("{s}-PhongTexture"));
        let phong_binding = PhongBinding::new(device, phong_layout, phong, phong_label.as_deref());

        Self {
            phong_binding,
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
            label: label.map(|s| format!("{s}-CameraBuffer")).as_deref(),
            contents: bytemuck::cast_slice(&[raw_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: label.map(|s| format!("{s}-CameraBindGroup")).as_deref(),
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
            label: label.map(|s| format!("{s}-LightBuffer")).as_deref(),
            contents: bytemuck::cast_slice(&[raw_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: label.map(|s| format!("{s}-LightBindGroup")).as_deref(),
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

impl ShadowBinding {
    pub fn new<C>(
        device: &wgpu::Device,
        uniform_layout: &ShadowUniformBindGroupLayout,
        texture_layout: &ShadowTextureBindGroupLayout,
        light_camera: &C,
        label: Option<&str>,
    ) -> Self
    where
        C: GpuTransfer<Raw = CameraRaw>,
    {
        let raw_data = light_camera.to_raw();

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label.map(|s| format!("{s}-ShadowBuffer")).as_deref(),
            contents: bytemuck::cast_slice(&[raw_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let dims = TextureDims {
            width: NonZeroU32::new(1024).unwrap(),
            height: NonZeroU32::new(1024).unwrap(),
        };
        let texture = Texture::create_depth(
            device,
            dims,
            label.map(|s| format!("{s}-ShadowTexture")).as_deref(),
        );
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: label
                .map(|s| format!("{s}-ShadowUniformBindGroup"))
                .as_deref(),
            layout: &uniform_layout.0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: label
                .map(|s| format!("{s}-ShadowTextureBindGroup"))
                .as_deref(),
            layout: &texture_layout.0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ],
        });
        Self {
            uniform_bind_group,
            texture_bind_group,
            buffer,
            texture,
        }
    }

    #[expect(unused)]
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
            label: label.map(|s| format!("{s}-PhongBuffer")).as_deref(),
            contents: bytemuck::cast_slice(&[raw_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: label.map(|s| format!("{s}-PhongBindGroup")).as_deref(),
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
            label: label.map(|s| format!("{s}-VertexBuffer")).as_deref(),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label.map(|s| format!("{s}-IndexBuffer")).as_deref(),
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
            label: label.map(|s| format!("{s}-InstanceBuffer")).as_deref(),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self {
            buffer,
            num_instances,
        }
    }
}

impl ModelResourceStorage {
    pub fn new() -> Self {
        Self {
            mesh_buffers: Vec::new(),
            instance_buffers: Vec::new(),
            material_bindings: Vec::new(),
        }
    }

    pub fn get_mesh_buffer(&self, index: MeshBufferIndex) -> &MeshBuffer {
        let MeshBufferIndex { index } = index;
        &self.mesh_buffers[index]
    }

    pub fn get_instance_buffer(&self, index: InstanceBufferIndex) -> &InstanceBuffer {
        let InstanceBufferIndex { index } = index;
        &self.instance_buffers[index]
    }

    pub fn get_material_binding(&self, index: MaterialBindingIndex) -> &MaterialBindings {
        let MaterialBindingIndex { index } = index;
        &self.material_bindings[index]
    }

    pub fn upload_mesh<M>(
        &mut self,
        device: &wgpu::Device,
        mesh: &M,
        label: Option<&str>,
    ) -> MeshBufferIndex
    where
        M: GpuTransfer<Raw = TriangleBufferRaw>,
    {
        let index = self.mesh_buffers.len();
        self.mesh_buffers.push(MeshBuffer::new(device, mesh, label));
        MeshBufferIndex { index }
    }

    pub fn upload_instance<I>(
        &mut self,
        device: &wgpu::Device,
        instances: &I,
        label: Option<&str>,
    ) -> InstanceBufferIndex
    where
        I: GpuTransfer<Raw = InstanceBufferRaw>,
    {
        let index = self.instance_buffers.len();
        self.instance_buffers
            .push(InstanceBuffer::new(device, instances, label));
        InstanceBufferIndex { index }
    }

    #[expect(clippy::too_many_arguments)]
    pub fn upload_material<P, T, N>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        phong_layout: &PhongBindGroupLayout,
        texture_layout: &TextureBindGroupLayout,
        phong: &P,
        color_texture: Option<&T>,
        normal_texture: Option<&N>,
        label: Option<&str>,
    ) -> MaterialBindingIndex
    where
        P: GpuTransfer<Raw = PhongRaw>,
        T: GpuTransferTexture,
        N: GpuTransferTexture,
    {
        let index = self.material_bindings.len();
        self.material_bindings.push(MaterialBindings::new(
            device,
            queue,
            phong_layout,
            texture_layout,
            phong,
            color_texture,
            normal_texture,
            label,
        ));
        MaterialBindingIndex { index }
    }
}
