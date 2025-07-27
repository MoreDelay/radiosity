use std::{
    f32,
    num::{NonZeroU32, NonZeroU64},
    ops::Deref,
};

use static_assertions::const_assert_ne;
use wgpu::util::DeviceExt;

use crate::render::OPENGL_TO_WGPU_MATRIX;

use super::{
    CameraRaw, GpuTransfer, GpuTransferTexture, InstanceBufferRaw, LightRaw, PhongRaw,
    TriangleBufferRaw,
};

#[derive(Debug, Clone, Copy)]
pub struct TextureDims {
    pub width: NonZeroU32,
    pub height: NonZeroU32,
}

pub struct TextureBindGroupLayout(wgpu::BindGroupLayout);
pub struct DebugTextureBindGroupLayout(wgpu::BindGroupLayout);

pub struct CameraBindGroupLayout(wgpu::BindGroupLayout);
pub struct LightBindGroupLayout(wgpu::BindGroupLayout);
pub struct PhongBindGroupLayout(wgpu::BindGroupLayout);

pub struct Texture {
    #[expect(unused)]
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

pub struct DepthTexture(Texture);

pub struct ColorBinding {
    bind_group: wgpu::BindGroup,
    #[expect(unused)]
    texture: Texture,
}
pub struct NormalBinding {
    bind_group: wgpu::BindGroup,
    #[expect(unused)]
    texture: Texture,
}

pub struct MaterialBindings {
    pub phong_binding: PhongBinding,
    pub color: Option<ColorBinding>,
    pub normal: Option<NormalBinding>,
}
pub struct CameraBinding {
    pub bind_group: wgpu::BindGroup,
    pub buffer: wgpu::Buffer,
}
pub struct PhongBinding {
    pub bind_group: wgpu::BindGroup,
    #[expect(unused)]
    buffer: wgpu::Buffer,
}

// Shadow resources
pub struct LightBinding {
    pub bind_group: wgpu::BindGroup,
    pub buffer: wgpu::Buffer,
}

pub struct ShadowTransformBindGroupLayout(wgpu::BindGroupLayout);
pub struct ShadowTextureBindGroupLayout(wgpu::BindGroupLayout);
pub struct ShadowLayouts {
    pub transform: ShadowTransformBindGroupLayout,
    pub texture: ShadowTextureBindGroupLayout,
}

pub struct ShadowBindings {
    #[expect(unused)]
    texture: wgpu::Texture,
    #[expect(unused)]
    sampler: wgpu::Sampler,
    view_buffers: [wgpu::Buffer; 6],
    #[expect(unused)]
    cube_view: wgpu::TextureView,

    pub cube_bind: wgpu::BindGroup,
    pub transform_binds: [wgpu::BindGroup; 6],
    pub layer_views: [wgpu::TextureView; 6],
}

// 3D model resources
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

impl DepthTexture {
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new(device: &wgpu::Device, dims: TextureDims, label: Option<&str>) -> Self {
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
            format: Self::FORMAT,
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

        let texture = Texture {
            texture,
            view,
            sampler,
        };
        Self(texture)
    }
}

impl TextureBindGroupLayout {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TextureBindGroupLayout"),
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

impl ShadowTransformBindGroupLayout {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShadowTransformBindGroupLayout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
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
            label: Some("ShadowTextureBindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        Self(layout)
    }
}

impl ShadowLayouts {
    pub fn new(device: &wgpu::Device) -> Self {
        let transform = ShadowTransformBindGroupLayout::new(device);
        let texture = ShadowTextureBindGroupLayout::new(device);
        Self { transform, texture }
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
            ColorBinding {
                bind_group,
                texture,
            }
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
            NormalBinding {
                bind_group,
                texture,
            }
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

impl ShadowBindings {
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

    pub fn new(
        device: &wgpu::Device,
        layouts: &ShadowLayouts,
        dims: TextureDims,
        light: &LightRaw,
        label: Option<&str>,
    ) -> Self {
        let TextureDims { width, height } = dims;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: label.map(|s| format!("{s}-ShadowCubeTexture")).as_deref(),
            size: wgpu::Extent3d {
                width: width.get(),
                height: height.get(),
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: None,
            ..Default::default()
        });

        let cube_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: label.map(|s| format!("{s}-ShadowCubeView")).as_deref(),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let cube_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: label.map(|s| format!("{s}-ShadowCubeBind")).as_deref(),
            layout: &layouts.texture,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&cube_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let view_buffers: [wgpu::Buffer; 6] = Self::create_projs(light)
            .into_iter()
            .map(|proj| {
                let proj: [[f32; 4]; 4] = proj.into();
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: label.map(|s| format!("{s}-ShadowViewTransform")).as_deref(),
                    contents: bytemuck::cast_slice(&[proj]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let transform_binds = view_buffers
            .iter()
            .map(|b| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: label.map(|s| format!("{s}-ShadowViewTransform")).as_deref(),
                    layout: &layouts.transform,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: b.as_entire_binding(),
                    }],
                })
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let layer_views = [
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: label.map(|s| format!("{s}-CubeView-0")).as_deref(),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: 0,
                array_layer_count: Some(1),
                ..Default::default()
            }),
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: label.map(|s| format!("{s}-CubeView-1")).as_deref(),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: 1,
                array_layer_count: Some(1),
                ..Default::default()
            }),
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: label.map(|s| format!("{s}-CubeView-2")).as_deref(),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: 2,
                array_layer_count: Some(1),
                ..Default::default()
            }),
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: label.map(|s| format!("{s}-CubeView-3")).as_deref(),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: 3,
                array_layer_count: Some(1),
                ..Default::default()
            }),
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: label.map(|s| format!("{s}-CubeView-4")).as_deref(),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: 4,
                array_layer_count: Some(1),
                ..Default::default()
            }),
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: label.map(|s| format!("{s}-CubeView-5")).as_deref(),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: 5,
                array_layer_count: Some(1),
                ..Default::default()
            }),
        ];

        Self {
            texture,
            sampler,
            cube_bind,
            cube_view,
            view_buffers,
            transform_binds,
            layer_views,
        }
    }

    pub fn update(&self, queue: &wgpu::Queue, light: &LightRaw) {
        let size = 4 * 4 * std::mem::size_of::<f32>();
        let size = NonZeroU64::new(size as u64).unwrap();

        for (proj, buffer) in Self::create_projs(light)
            .iter()
            .zip(self.view_buffers.iter())
        {
            let mut view = queue.write_buffer_with(buffer, 0, size).unwrap();
            let proj: [[f32; 4]; 4] = (*proj).into();
            view.copy_from_slice(bytemuck::cast_slice(&[proj]));
        }
    }

    fn create_projs(light: &LightRaw) -> [cgmath::Matrix4<f32>; 6] {
        let dir_pos_x = cgmath::Vector3::unit_x();
        let dir_neg_x = -cgmath::Vector3::unit_x();
        let dir_pos_y = cgmath::Vector3::unit_y();
        let dir_neg_y = -cgmath::Vector3::unit_y();
        let dir_pos_z = -cgmath::Vector3::unit_z();
        let dir_neg_z = cgmath::Vector3::unit_z();

        let up_pos_x = cgmath::Vector3::unit_y();
        let up_neg_x = cgmath::Vector3::unit_y();
        let up_pos_y = cgmath::Vector3::unit_z();
        let up_neg_y = -cgmath::Vector3::unit_z();
        let up_pos_z = cgmath::Vector3::unit_y();
        let up_neg_z = cgmath::Vector3::unit_y();

        // TODO: What is going on??? Why not 90 but 68???
        let view = cgmath::perspective(cgmath::Deg(68.), 1., 0.1, light.max_dist);
        let pos = light.position.into();
        [
            OPENGL_TO_WGPU_MATRIX * view * cgmath::Matrix4::look_to_rh(pos, dir_pos_x, up_pos_x),
            OPENGL_TO_WGPU_MATRIX * view * cgmath::Matrix4::look_to_rh(pos, dir_neg_x, up_neg_x),
            OPENGL_TO_WGPU_MATRIX * view * cgmath::Matrix4::look_to_rh(pos, dir_pos_y, up_pos_y),
            OPENGL_TO_WGPU_MATRIX * view * cgmath::Matrix4::look_to_rh(pos, dir_neg_y, up_neg_y),
            OPENGL_TO_WGPU_MATRIX * view * cgmath::Matrix4::look_to_rh(pos, dir_pos_z, up_pos_z),
            OPENGL_TO_WGPU_MATRIX * view * cgmath::Matrix4::look_to_rh(pos, dir_neg_z, up_neg_z),
        ]
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
}

impl MeshBuffer {
    pub fn new<M>(device: &wgpu::Device, mesh: &M, label: Option<&str>) -> Self
    where
        M: GpuTransfer<Raw = TriangleBufferRaw>,
    {
        let TriangleBufferRaw { vertices, indices } = mesh.to_raw();

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

impl Deref for TextureBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for DebugTextureBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for CameraBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Deref for LightBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Deref for ShadowTransformBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Deref for ShadowTextureBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Deref for PhongBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for DepthTexture {
    type Target = Texture;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for ColorBinding {
    type Target = wgpu::BindGroup;

    fn deref(&self) -> &Self::Target {
        &self.bind_group
    }
}

impl Deref for NormalBinding {
    type Target = wgpu::BindGroup;

    fn deref(&self) -> &Self::Target {
        &self.bind_group
    }
}
