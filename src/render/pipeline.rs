use std::ops::Deref;

use crate::render::{GpuContext, TargetContext, resource::PhongLayouts};

use super::{raw, resource};

pub struct FlatScenePipeline(wgpu::RenderPipeline);
pub struct ColorScenePipeline(wgpu::RenderPipeline);
pub struct NormalScenePipeline(wgpu::RenderPipeline);
pub struct ColorNormalScenePipeline(wgpu::RenderPipeline);
pub struct LightPipeline(wgpu::RenderPipeline);

pub struct TexturePipelines {
    flat: FlatScenePipeline,
    color: ColorScenePipeline,
    normal: NormalScenePipeline,
    color_normal: ColorNormalScenePipeline,
}

pub struct ShadowPipeline(wgpu::RenderPipeline);

impl TexturePipelines {
    pub fn new(
        ctx: &GpuContext,
        target: &TargetContext,
        layouts: &PhongLayouts,
    ) -> anyhow::Result<Self> {
        let flat = FlatScenePipeline::new(ctx, target, layouts)?;
        let color = ColorScenePipeline::new(ctx, target, layouts)?;
        let normal = NormalScenePipeline::new(ctx, target, layouts)?;
        let color_normal = ColorNormalScenePipeline::new(ctx, target, layouts)?;

        Ok(Self {
            flat,
            color,
            normal,
            color_normal,
        })
    }

    pub fn get_flat(&self) -> &FlatScenePipeline {
        &self.flat
    }

    #[expect(unused)]
    pub fn get_color(&self) -> &ColorScenePipeline {
        &self.color
    }

    #[expect(unused)]
    pub fn get_normal(&self) -> &NormalScenePipeline {
        &self.normal
    }

    #[expect(unused)]
    pub fn get_color_normal(&self) -> &ColorNormalScenePipeline {
        &self.color_normal
    }
}

impl FlatScenePipeline {
    const SHADER: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/shaders/model_none.wgsl"
    ));

    pub fn new(
        ctx: &GpuContext,
        target: &TargetContext,
        layouts: &PhongLayouts,
    ) -> anyhow::Result<Self> {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("FlatPipelineLayout"),
                bind_group_layouts: &[
                    &layouts.camera,
                    &layouts.light,
                    &layouts.phong,
                    &layouts.shadow_texture,
                ],
                push_constant_ranges: &[],
            });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("FlatShader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
        };
        let pipeline = create_render_pipeline(
            ctx,
            target,
            &layout,
            Some(resource::DepthTexture::FORMAT),
            // &[raw::VertexRaw::desc(), raw::InstanceRaw::desc()],
            &[
                raw::position_desc(),
                raw::tex_coord_desc(),
                raw::normal_desc(),
                raw::tangent_desc(),
                raw::bitangent_desc(),
                raw::InstanceRaw::desc(),
            ],
            shader,
            Some("FlatPipeline"),
        );
        Ok(Self(pipeline))
    }
}

impl ColorScenePipeline {
    const SHADER: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/shaders/model_c.wgsl"
    ));

    pub fn new(
        ctx: &GpuContext,
        target: &TargetContext,
        layouts: &PhongLayouts,
    ) -> anyhow::Result<Self> {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ColorPipelineLayout"),
                bind_group_layouts: &[
                    &layouts.camera,
                    &layouts.light,
                    &layouts.phong,
                    &layouts.texture,
                ],
                push_constant_ranges: &[],
            });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("ColorShader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
        };
        let pipeline = create_render_pipeline(
            ctx,
            target,
            &layout,
            Some(resource::DepthTexture::FORMAT),
            &[raw::VertexRaw::desc(), raw::InstanceRaw::desc()],
            shader,
            Some("ColorPipeline"),
        );
        Ok(Self(pipeline))
    }
}

impl NormalScenePipeline {
    const SHADER: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/shaders/model_n.wgsl"
    ));

    pub fn new(
        ctx: &GpuContext,
        target: &TargetContext,
        layouts: &PhongLayouts,
    ) -> anyhow::Result<Self> {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("NormalPipelineLayout"),
                bind_group_layouts: &[
                    &layouts.camera,
                    &layouts.light,
                    &layouts.phong,
                    &layouts.texture,
                ],
                push_constant_ranges: &[],
            });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("NormalShader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
        };
        let pipeline = create_render_pipeline(
            ctx,
            target,
            &layout,
            Some(resource::DepthTexture::FORMAT),
            &[raw::VertexRaw::desc(), raw::InstanceRaw::desc()],
            shader,
            Some("NormalPipeline"),
        );
        Ok(Self(pipeline))
    }
}

impl ColorNormalScenePipeline {
    const SHADER: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/shaders/model_cn.wgsl"
    ));

    pub fn new(
        ctx: &GpuContext,
        target: &TargetContext,
        layouts: &PhongLayouts,
    ) -> anyhow::Result<Self> {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ColorNormalPipelineLayout"),
                bind_group_layouts: &[
                    &layouts.camera,
                    &layouts.light,
                    &layouts.phong,
                    &layouts.texture, // color texture
                    &layouts.texture, // normal texture
                ],
                push_constant_ranges: &[],
            });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("ColorNormalShader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
        };
        let pipeline = create_render_pipeline(
            ctx,
            target,
            &layout,
            Some(resource::DepthTexture::FORMAT),
            &[raw::VertexRaw::desc(), raw::InstanceRaw::desc()],
            shader,
            Some("ColorNormalPipeline"),
        );
        Ok(Self(pipeline))
    }
}

impl LightPipeline {
    const SHADER: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/shaders/light.wgsl"
    ));

    pub fn new(
        ctx: &GpuContext,
        target: &TargetContext,
        layouts: &PhongLayouts,
    ) -> anyhow::Result<Self> {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&layouts.camera, &layouts.light],
                push_constant_ranges: &[],
            });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("Light Shader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
        };
        let pipeline = create_render_pipeline(
            ctx,
            target,
            &layout,
            Some(resource::DepthTexture::FORMAT),
            &[raw::VertexRaw::desc(), raw::InstanceRaw::desc()],
            shader,
            Some("Light Pipeline"),
        );
        Ok(Self(pipeline))
    }
}

impl ShadowPipeline {
    const SHADER: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/shaders/shadow.wgsl"
    ));

    pub fn new(ctx: &GpuContext, layouts: &PhongLayouts) -> anyhow::Result<Self> {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ShadowPipelineLayout"),
                bind_group_layouts: &[&layouts.shadow_transform, &layouts.light],
                push_constant_ranges: &[],
            });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("ShadowShader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
        };
        let shader = ctx.device.create_shader_module(shader);

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ShadowPipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    // buffers: &[raw::VertexRaw::desc(), raw::InstanceRaw::desc()],
                    buffers: &[
                        raw::position_desc(),
                        raw::tex_coord_desc(),
                        raw::normal_desc(),
                        raw::tangent_desc(),
                        raw::bitangent_desc(),
                        raw::InstanceRaw::desc(),
                    ],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: resource::ShadowBindings::FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: resource::DepthTexture::FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0, // use all samples
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });
        Ok(Self(pipeline))
    }
}

fn create_render_pipeline(
    ctx: &GpuContext,
    target: &TargetContext,
    layout: &wgpu::PipelineLayout,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layout: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    label: Option<&str>,
) -> wgpu::RenderPipeline {
    let shader = ctx.device.create_shader_module(shader);

    ctx.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label,
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: vertex_layout,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target.config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: depth_format.map(|depth_format| wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, // what pixels to keep
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0, // use all samples
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
}

impl Deref for FlatScenePipeline {
    type Target = wgpu::RenderPipeline;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for ColorScenePipeline {
    type Target = wgpu::RenderPipeline;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for NormalScenePipeline {
    type Target = wgpu::RenderPipeline;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for ColorNormalScenePipeline {
    type Target = wgpu::RenderPipeline;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for LightPipeline {
    type Target = wgpu::RenderPipeline;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for ShadowPipeline {
    type Target = wgpu::RenderPipeline;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
