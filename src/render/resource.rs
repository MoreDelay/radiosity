use std::{
    f32,
    num::{NonZeroU32, NonZeroU64},
    ops::Deref,
};

use nalgebra as na;

use static_assertions::const_assert_ne;
use wgpu::util::DeviceExt;
use zerocopy::IntoBytes;

use crate::{model, render::pipeline::PhongCapabilites};

use super::{CameraRaw, GpuContext, LightRaw, PhongRaw, TextureRaw};

#[derive(Debug, Clone, Copy)]
pub struct TextureDims {
    pub width: NonZeroU32,
    pub height: NonZeroU32,
}

pub struct TextureBindGroupLayout(wgpu::BindGroupLayout);

impl TextureBindGroupLayout {
    pub fn new(ctx: &GpuContext) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

impl Deref for TextureBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct CameraBindGroupLayout(wgpu::BindGroupLayout);

impl CameraBindGroupLayout {
    pub fn new(ctx: &GpuContext) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

impl Deref for CameraBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct LightBindGroupLayout(wgpu::BindGroupLayout);

impl LightBindGroupLayout {
    pub fn new(ctx: &GpuContext) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

impl Deref for LightBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct PhongParamBindGroupLayout(wgpu::BindGroupLayout);

impl PhongParamBindGroupLayout {
    pub fn new(ctx: &GpuContext) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

impl Deref for PhongParamBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct PhongLayouts {
    pub camera: CameraBindGroupLayout,
    pub light: LightBindGroupLayout,
    pub phong: PhongParamBindGroupLayout,
    pub texture: TextureBindGroupLayout,
    pub shadow_transform: ShadowTransformBindGroupLayout,
    pub shadow_texture: ShadowTextureBindGroupLayout,
}

impl PhongLayouts {
    pub fn new(ctx: &GpuContext) -> Self {
        let camera = CameraBindGroupLayout::new(ctx);
        let light = LightBindGroupLayout::new(ctx);
        let phong = PhongParamBindGroupLayout::new(ctx);
        let texture = TextureBindGroupLayout::new(ctx);
        let shadow_transform = ShadowTransformBindGroupLayout::new(ctx);
        let shadow_texture = ShadowTextureBindGroupLayout::new(ctx);
        Self {
            camera,
            light,
            phong,
            texture,
            shadow_transform,
            shadow_texture,
        }
    }
}

pub struct Texture {
    pub _texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

pub struct DepthTexture(Texture);

impl DepthTexture {
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new(ctx: &GpuContext, dims: TextureDims, label: Option<&str>) -> Self {
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
        let texture = ctx.device.create_texture(&desc);

        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
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
            _texture: texture,
            view,
            sampler,
        };
        Self(texture)
    }
}

impl Deref for DepthTexture {
    type Target = Texture;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct ColorBinding {
    bind_group: wgpu::BindGroup,
    _texture: Texture,
}

impl Deref for ColorBinding {
    type Target = wgpu::BindGroup;

    fn deref(&self) -> &Self::Target {
        &self.bind_group
    }
}

pub struct NormalBinding {
    bind_group: wgpu::BindGroup,
    _texture: Texture,
}

impl Deref for NormalBinding {
    type Target = wgpu::BindGroup;

    fn deref(&self) -> &Self::Target {
        &self.bind_group
    }
}

pub struct MaterialBindingGroup {
    pub phong: PhongBinding,
    pub color: Option<ColorBinding>,
    pub normal: Option<NormalBinding>,
}

impl MaterialBindingGroup {
    pub fn new(
        ctx: &GpuContext,
        layouts: &PhongLayouts,
        phong: &PhongRaw,
        color_texture: Option<&TextureRaw>,
        normal_texture: Option<&TextureRaw>,
        label: Option<&str>,
    ) -> Self {
        let color = color_texture.map(|t| {
            let texture_label = label.map(|s| format!("{s}-ColorTexture"));
            let texture = t.create_texture(ctx, texture_label.as_deref());
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
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label,
                layout: &layouts.texture,
                entries: &entries,
            });
            ColorBinding {
                bind_group,
                _texture: texture,
            }
        });

        let normal = normal_texture.map(|t| {
            let texture_label = label.map(|s| format!("{s}-NormalTexture"));
            let texture = t.create_texture(ctx, texture_label.as_deref());
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
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label,
                layout: &layouts.texture,
                entries: &entries,
            });
            NormalBinding {
                bind_group,
                _texture: texture,
            }
        });

        let phong_label = label.map(|s| format!("{s}-PhongMaterial"));
        let phong = PhongBinding::new(ctx, layouts, phong, phong_label.as_deref());

        Self {
            phong,
            color,
            normal,
        }
    }
}

pub struct CameraBinding {
    pub bind_group: wgpu::BindGroup,
    pub buffer: wgpu::Buffer,
}

impl CameraBinding {
    pub fn new(
        ctx: &GpuContext,
        layouts: &PhongLayouts,
        data: &CameraRaw,
        label: Option<&str>,
    ) -> Self {
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: label.map(|s| format!("{s}-CameraBuffer")).as_deref(),
                contents: bytemuck::cast_slice(&[*data]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: label.map(|s| format!("{s}-CameraBindGroup")).as_deref(),
            layout: &layouts.camera,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self { bind_group, buffer }
    }

    pub fn update(&self, ctx: &GpuContext, data: &CameraRaw) {
        // make sure size unwrap never panics
        const_assert_ne!(std::mem::size_of::<CameraRaw>(), 0);
        let size = NonZeroU64::try_from(std::mem::size_of::<CameraRaw>() as u64).unwrap();

        let mut view = ctx.queue.write_buffer_with(&self.buffer, 0, size).unwrap();
        view.copy_from_slice(bytemuck::cast_slice(&[*data]));
    }
}

pub struct PhongBinding {
    pub bind_group: wgpu::BindGroup,
    _buffer: wgpu::Buffer,
}

impl PhongBinding {
    pub fn new(
        ctx: &GpuContext,
        layouts: &PhongLayouts,
        data: &PhongRaw,
        label: Option<&str>,
    ) -> Self {
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: label.map(|s| format!("{s}-PhongBuffer")).as_deref(),
                contents: bytemuck::cast_slice(&[*data]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: label.map(|s| format!("{s}-PhongBindGroup")).as_deref(),
            layout: &layouts.phong,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            bind_group,
            _buffer: buffer,
        }
    }
}

// Shadow resources
pub struct LightBinding {
    pub bind_group: wgpu::BindGroup,
    pub buffer: wgpu::Buffer,
}

impl LightBinding {
    pub fn new(
        ctx: &GpuContext,
        layouts: &PhongLayouts,
        data: &LightRaw,
        label: Option<&str>,
    ) -> Self {
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: label.map(|s| format!("{s}-LightBuffer")).as_deref(),
                contents: bytemuck::cast_slice(&[*data]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: label.map(|s| format!("{s}-LightBindGroup")).as_deref(),
            layout: &layouts.light,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self { bind_group, buffer }
    }

    pub fn update(&self, ctx: &GpuContext, data: &LightRaw) {
        // make sure size unwrap never panics
        const_assert_ne!(std::mem::size_of::<LightRaw>(), 0);
        let size = NonZeroU64::try_from(std::mem::size_of::<LightRaw>() as u64).unwrap();

        let mut view = ctx.queue.write_buffer_with(&self.buffer, 0, size).unwrap();
        view.copy_from_slice(bytemuck::cast_slice(&[*data]));
    }
}

pub struct ShadowTransformBindGroupLayout(wgpu::BindGroupLayout);

impl ShadowTransformBindGroupLayout {
    pub fn new(ctx: &GpuContext) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

impl Deref for ShadowTransformBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct ShadowTextureBindGroupLayout(wgpu::BindGroupLayout);

impl ShadowTextureBindGroupLayout {
    pub fn new(ctx: &GpuContext) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

impl Deref for ShadowTextureBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
pub struct ShadowBindings {
    _texture: wgpu::Texture,
    _sampler: wgpu::Sampler,
    view_buffers: [wgpu::Buffer; 6],
    _cube_view: wgpu::TextureView,

    pub cube_bind: wgpu::BindGroup,
    pub transform_binds: [wgpu::BindGroup; 6],
    pub cube_views: [wgpu::TextureView; 6],
}

impl ShadowBindings {
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

    pub fn new(
        ctx: &GpuContext,
        layouts: &PhongLayouts,
        dims: TextureDims,
        light: &LightRaw,
        label: Option<&str>,
    ) -> Self {
        let TextureDims { width, height } = dims;
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
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

        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
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

        let cube_bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: label.map(|s| format!("{s}-ShadowCubeBind")).as_deref(),
            layout: &layouts.shadow_texture,
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
                ctx.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: label.map(|s| format!("{s}-ShadowViewTransform")).as_deref(),
                    layout: &layouts.shadow_transform,
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
            _texture: texture,
            _sampler: sampler,
            cube_bind,
            _cube_view: cube_view,
            view_buffers,
            transform_binds,
            cube_views: layer_views,
        }
    }

    pub fn update(&self, ctx: &GpuContext, light: &LightRaw) {
        let size = 4 * 4 * std::mem::size_of::<f32>();
        let size = NonZeroU64::new(size as u64).unwrap();

        for (proj, buffer) in Self::create_projs(light)
            .iter()
            .zip(self.view_buffers.iter())
        {
            let mut view = ctx.queue.write_buffer_with(buffer, 0, size).unwrap();
            let proj: [[f32; 4]; 4] = (*proj).into();
            view.copy_from_slice(bytemuck::cast_slice(&[proj]));
        }
    }

    fn create_projs(light: &LightRaw) -> [na::Matrix4<f32>; 6] {
        let dir_pos_x = na::Vector3::new(1., 0., 0.);
        let dir_neg_x = na::Vector3::new(-1., 0., 0.);
        let dir_pos_y = na::Vector3::new(0., 1., 0.);
        let dir_neg_y = na::Vector3::new(0., -1., 0.);
        let dir_pos_z = na::Vector3::new(0., 0., -1.);
        let dir_neg_z = na::Vector3::new(0., 0., 1.);

        let up_pos_x = na::Vector3::new(0., 1., 0.);
        let up_neg_x = na::Vector3::new(0., 1., 0.);
        let up_pos_y = na::Vector3::new(0., 0., 1.);
        let up_neg_y = na::Vector3::new(0., 0., -1.);
        let up_pos_z = na::Vector3::new(0., 1., 0.);
        let up_neg_z = na::Vector3::new(0., 1., 0.);

        let pos = light.position.into();
        let view_pos_x =
            crate::math::view_matrix(pos, crate::math::rotation_towards(dir_pos_x, up_pos_x));
        let view_neg_x =
            crate::math::view_matrix(pos, crate::math::rotation_towards(dir_neg_x, up_neg_x));
        let view_pos_y =
            crate::math::view_matrix(pos, crate::math::rotation_towards(dir_pos_y, up_pos_y));
        let view_neg_y =
            crate::math::view_matrix(pos, crate::math::rotation_towards(dir_neg_y, up_neg_y));
        let view_pos_z =
            crate::math::view_matrix(pos, crate::math::rotation_towards(dir_pos_z, up_pos_z));
        let view_neg_z =
            crate::math::view_matrix(pos, crate::math::rotation_towards(dir_neg_z, up_neg_z));

        let proj = crate::math::perspective_projection(90., 1., 0.1, light.max_dist);
        [
            proj * view_pos_x,
            proj * view_neg_x,
            proj * view_pos_y,
            proj * view_neg_y,
            proj * view_pos_z,
            proj * view_neg_z,
        ]
    }
}

// 3D model resources
pub struct InstanceBuffer {
    pub buffer: wgpu::Buffer,
    pub num_instances: u32,
}

impl InstanceBuffer {
    pub fn new(ctx: &GpuContext, instances: &[model::Instance], label: Option<&str>) -> Self {
        let instances = instances
            .iter()
            .map(|instance| instance.to_raw())
            .collect::<Vec<_>>();
        let num_instances = instances.len() as u32;

        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: label.map(|s| format!("{s}-InstanceBuffer")).as_deref(),
                contents: bytemuck::cast_slice(&instances),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        Self {
            buffer,
            num_instances,
        }
    }
}

pub(super) struct IndexBuffer {
    pub buffer: wgpu::Buffer,
    pub _count: u32,
}

impl IndexBuffer {
    fn new(ctx: &GpuContext, data: &[u32], label: Option<&str>) -> Self {
        let count = data.len() as u32;
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: label.map(|s| format!("{s}-IndexBuffer")).as_deref(),
                contents: data.as_bytes(),
                usage: wgpu::BufferUsages::INDEX,
            });
        Self {
            buffer,
            _count: count,
        }
    }
}

pub(super) struct PositionBuffer {
    pub buffer: wgpu::Buffer,
    pub _count: u32,
}

impl PositionBuffer {
    fn new(ctx: &GpuContext, data: &[f32], label: Option<&str>) -> Self {
        assert!(
            data.len().is_multiple_of(3),
            "position buffer has wrong size"
        );
        let count = (data.len() / 3) as u32;
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: data.as_bytes(),
                usage: wgpu::BufferUsages::VERTEX,
            });
        Self {
            buffer,
            _count: count,
        }
    }
}

pub(super) struct TexCoordsBuffer {
    pub buffer: wgpu::Buffer,
    pub _count: u32,
}

impl TexCoordsBuffer {
    fn new(ctx: &GpuContext, data: &[f32], label: Option<&str>) -> Self {
        assert!(
            data.len().is_multiple_of(2),
            "tex coords buffer has wrong size"
        );
        let count = (data.len() / 2) as u32;
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: data.as_bytes(),
                usage: wgpu::BufferUsages::VERTEX,
            });
        Self {
            buffer,
            _count: count,
        }
    }
}

pub(super) struct NormalBuffer {
    pub buffer: wgpu::Buffer,
    pub _count: u32,
}

impl NormalBuffer {
    fn new(ctx: &GpuContext, data: &[f32], label: Option<&str>) -> Self {
        assert!(data.len().is_multiple_of(3), "normal buffer has wrong size");
        let count = (data.len() / 2) as u32;
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: data.as_bytes(),
                usage: wgpu::BufferUsages::VERTEX,
            });
        Self {
            buffer,
            _count: count,
        }
    }
}

pub(super) struct TangentBuffer {
    pub buffer: wgpu::Buffer,
    pub _count: u32,
}

impl TangentBuffer {
    fn new(ctx: &GpuContext, data: &[f32], label: Option<&str>) -> Self {
        assert!(
            data.len().is_multiple_of(3),
            "tangent buffer has wrong size"
        );
        let count = (data.len() / 2) as u32;
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: data.as_bytes(),
                usage: wgpu::BufferUsages::VERTEX,
            });
        Self {
            buffer,
            _count: count,
        }
    }
}

pub(super) struct BiTangentBuffer {
    pub buffer: wgpu::Buffer,
    pub _count: u32,
}

impl BiTangentBuffer {
    fn new(ctx: &GpuContext, data: &[f32], label: Option<&str>) -> Self {
        assert!(
            data.len().is_multiple_of(3),
            "tangent buffer has wrong size"
        );
        let count = (data.len() / 2) as u32;
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: data.as_bytes(),
                usage: wgpu::BufferUsages::VERTEX,
            });
        Self {
            buffer,
            _count: count,
        }
    }
}

pub(super) struct ResourceStorage {
    index_buffers: Vec<IndexBuffer>,
    position_buffers: Vec<PositionBuffer>,
    tex_coord_buffers: Vec<TexCoordsBuffer>,
    normal_buffers: Vec<NormalBuffer>,
    tangent_buffers: Vec<TangentBuffer>,
    bi_tangent_buffers: Vec<BiTangentBuffer>,
    instance_buffers: Vec<InstanceBuffer>,
    material_bindings: Vec<MaterialBindingGroup>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndexBufferIndex(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PositionBufferIndex(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TexCoordBufferIndex(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NormalBufferIndex(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TangentBufferIndex(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BiTangentBufferIndex(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstanceBufferIndex(u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialBindingIndex(u32);

impl ResourceStorage {
    pub fn new() -> Self {
        Self {
            index_buffers: Vec::new(),
            position_buffers: Vec::new(),
            tex_coord_buffers: Vec::new(),
            normal_buffers: Vec::new(),
            tangent_buffers: Vec::new(),
            bi_tangent_buffers: Vec::new(),
            // mesh_buffers: Vec::new(),
            instance_buffers: Vec::new(),
            material_bindings: Vec::new(),
        }
    }

    pub fn index_buffer(&self, index: IndexBufferIndex) -> &IndexBuffer {
        &self.index_buffers[index.0 as usize]
    }

    pub fn position_buffer(&self, index: PositionBufferIndex) -> &PositionBuffer {
        &self.position_buffers[index.0 as usize]
    }

    pub fn tex_coord_buffer(&self, index: TexCoordBufferIndex) -> &TexCoordsBuffer {
        &self.tex_coord_buffers[index.0 as usize]
    }

    pub fn normal_buffer(&self, index: NormalBufferIndex) -> &NormalBuffer {
        &self.normal_buffers[index.0 as usize]
    }

    pub fn tangent_buffer(&self, index: TangentBufferIndex) -> &TangentBuffer {
        &self.tangent_buffers[index.0 as usize]
    }

    pub fn bi_tangent_buffer(&self, index: BiTangentBufferIndex) -> &BiTangentBuffer {
        &self.bi_tangent_buffers[index.0 as usize]
    }

    pub fn get_instance_buffer(&self, index: InstanceBufferIndex) -> &InstanceBuffer {
        &self.instance_buffers[index.0 as usize]
    }

    pub fn get_material_binding(&self, index: MaterialBindingIndex) -> &MaterialBindingGroup {
        &self.material_bindings[index.0 as usize]
    }

    pub fn upload_index_buffer(
        &mut self,
        ctx: &GpuContext,
        data: &[u32],
        label: Option<&str>,
    ) -> IndexBufferIndex {
        let index = self.index_buffers.len();
        let index = IndexBufferIndex(index as u32);
        self.index_buffers.push(IndexBuffer::new(ctx, data, label));
        index
    }

    pub fn upload_position_buffer(
        &mut self,
        ctx: &GpuContext,
        data: &[f32],
        label: Option<&str>,
    ) -> PositionBufferIndex {
        let index = self.position_buffers.len();
        let index = PositionBufferIndex(index as u32);
        self.position_buffers
            .push(PositionBuffer::new(ctx, data, label));
        index
    }

    pub fn upload_tex_coord_buffer(
        &mut self,
        ctx: &GpuContext,
        data: &[f32],
        label: Option<&str>,
    ) -> TexCoordBufferIndex {
        let index = self.tex_coord_buffers.len();
        let index = TexCoordBufferIndex(index as u32);
        self.tex_coord_buffers
            .push(TexCoordsBuffer::new(ctx, data, label));
        index
    }

    pub fn upload_normal_buffer(
        &mut self,
        ctx: &GpuContext,
        data: &[f32],
        label: Option<&str>,
    ) -> NormalBufferIndex {
        let index = self.normal_buffers.len();
        let index = NormalBufferIndex(index as u32);
        self.normal_buffers
            .push(NormalBuffer::new(ctx, data, label));
        index
    }

    pub fn upload_tangent_buffer(
        &mut self,
        ctx: &GpuContext,
        data: &[f32],
        label: Option<&str>,
    ) -> TangentBufferIndex {
        let index = self.tangent_buffers.len();
        let index = TangentBufferIndex(index as u32);
        self.tangent_buffers
            .push(TangentBuffer::new(ctx, data, label));
        index
    }

    pub fn upload_bi_tangent_buffer(
        &mut self,
        ctx: &GpuContext,
        data: &[f32],
        label: Option<&str>,
    ) -> BiTangentBufferIndex {
        let index = self.bi_tangent_buffers.len();
        let index = BiTangentBufferIndex(index as u32);
        self.bi_tangent_buffers
            .push(BiTangentBuffer::new(ctx, data, label));
        index
    }

    pub fn upload_instance(
        &mut self,
        ctx: &GpuContext,
        instances: &[model::Instance],
        label: Option<&str>,
    ) -> InstanceBufferIndex {
        let index = self.instance_buffers.len();
        let index = InstanceBufferIndex(index as u32);
        self.instance_buffers
            .push(InstanceBuffer::new(ctx, instances, label));
        index
    }

    pub fn upload_material(
        &mut self,
        ctx: &GpuContext,
        layouts: &PhongLayouts,
        phong: &PhongRaw,
        color_texture: Option<&TextureRaw>,
        normal_texture: Option<&TextureRaw>,
        label: Option<&str>,
    ) -> MaterialBindingIndex {
        let index = self.material_bindings.len();
        let index = MaterialBindingIndex(index as u32);
        self.material_bindings.push(MaterialBindingGroup::new(
            ctx,
            layouts,
            phong,
            color_texture,
            normal_texture,
            label,
        ));
        index
    }
}

#[derive(Debug, Copy, Clone)]
pub enum DrawType {
    Render(PhongCapabilites),
    Light,
}

impl PartialEq for DrawType {
    fn eq(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}

#[derive(Debug)]
pub struct DrawCall {
    pub material: MaterialBindingIndex,
    pub draw_type: DrawType,
    pub instance: InstanceBufferIndex,

    // primitive data
    pub index: IndexBufferIndex,
    pub slice: std::ops::Range<u32>,

    // vertex data
    pub position: PositionBufferIndex,
    pub tex_coord: Option<TexCoordBufferIndex>,
    pub normal: Option<NormalBufferIndex>,
    pub tangent: Option<TangentBufferIndex>,
    pub bi_tangent: Option<BiTangentBufferIndex>,
}

pub struct DrawWorld {
    pub draw_calls: Vec<DrawCall>,
}

impl DrawWorld {
    // Use bucket sort / counting sort to issue draw calls with shared resources (currently only
    // material). Relies on resource indices to be unique per resource to act as sort key.
    pub(super) fn sort(&self, storage: &ResourceStorage) -> Vec<u32> {
        let n_elements = self.draw_calls.len();
        let n_buckets = storage.material_bindings.len();
        let mut bucket_sizes = vec![0; n_buckets];

        for draw in self.draw_calls.iter() {
            let bucket = draw.material.0 as usize;
            bucket_sizes[bucket] += 1;
        }

        for i in 1..n_buckets {
            bucket_sizes[i] += bucket_sizes[i - 1];
        }
        let sizes_prefix_sum = bucket_sizes;

        let mut bucket_occupied = vec![0; n_buckets];
        let mut sorted_indices = vec![0; n_elements];

        for (index, draw) in self.draw_calls.iter().enumerate() {
            let bucket = draw.material.0 as usize;
            let offset = match bucket {
                0 => 0,
                1.. => sizes_prefix_sum[bucket - 1],
            };
            let sorted_index = offset + bucket_occupied[bucket];
            sorted_indices[sorted_index] = index as u32;
            bucket_occupied[bucket] += 1;
        }

        sorted_indices
    }
}
