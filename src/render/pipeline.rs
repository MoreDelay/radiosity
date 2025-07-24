use std::ops::Deref;

use crate::render::resource::DepthTexture;

use super::{layout, resource};

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
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        texture_layout: &resource::TextureBindGroupLayout,
        camera_layout: &resource::CameraBindGroupLayout,
        light_layout: &resource::LightBindGroupLayout,
        phong_layout: &resource::PhongBindGroupLayout,
    ) -> anyhow::Result<Self> {
        let flat = FlatScenePipeline::new(
            device,
            color_format,
            camera_layout,
            light_layout,
            phong_layout,
        )?;
        let color = ColorScenePipeline::new(
            device,
            color_format,
            texture_layout,
            camera_layout,
            light_layout,
            phong_layout,
        )?;
        let normal = NormalScenePipeline::new(
            device,
            color_format,
            texture_layout,
            camera_layout,
            light_layout,
            phong_layout,
        )?;
        let color_normal = ColorNormalScenePipeline::new(
            device,
            color_format,
            texture_layout,
            camera_layout,
            light_layout,
            phong_layout,
        )?;

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

    pub fn get_color(&self) -> &ColorScenePipeline {
        &self.color
    }

    pub fn get_normal(&self) -> &NormalScenePipeline {
        &self.normal
    }

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
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        camera_layout: &resource::CameraBindGroupLayout,
        light_layout: &resource::LightBindGroupLayout,
        phong_layout: &resource::PhongBindGroupLayout,
    ) -> anyhow::Result<Self> {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FlatPipelineLayout"),
            bind_group_layouts: &[&camera_layout.0, &light_layout.0, &phong_layout.0],
            push_constant_ranges: &[],
        });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("FlatShader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
        };
        let pipeline = create_render_pipeline(
            device,
            &layout,
            color_format,
            Some(resource::DepthTexture::DEPTH_FORMAT),
            &[layout::VertexRaw::desc(), layout::InstanceRaw::desc()],
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
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        texture_layout: &resource::TextureBindGroupLayout,
        camera_layout: &resource::CameraBindGroupLayout,
        light_layout: &resource::LightBindGroupLayout,
        phong_layout: &resource::PhongBindGroupLayout,
    ) -> anyhow::Result<Self> {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ColorPipelineLayout"),
            bind_group_layouts: &[
                &camera_layout.0,
                &light_layout.0,
                &phong_layout.0,
                texture_layout,
            ],
            push_constant_ranges: &[],
        });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("ColorShader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
        };
        let pipeline = create_render_pipeline(
            device,
            &layout,
            color_format,
            Some(resource::DepthTexture::DEPTH_FORMAT),
            &[layout::VertexRaw::desc(), layout::InstanceRaw::desc()],
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
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        texture_layout: &resource::TextureBindGroupLayout,
        camera_layout: &resource::CameraBindGroupLayout,
        light_layout: &resource::LightBindGroupLayout,
        phong_layout: &resource::PhongBindGroupLayout,
    ) -> anyhow::Result<Self> {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("NormalPipelineLayout"),
            bind_group_layouts: &[
                &camera_layout.0,
                &light_layout.0,
                &phong_layout.0,
                texture_layout,
            ],
            push_constant_ranges: &[],
        });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("NormalShader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
        };
        let pipeline = create_render_pipeline(
            device,
            &layout,
            color_format,
            Some(resource::DepthTexture::DEPTH_FORMAT),
            &[layout::VertexRaw::desc(), layout::InstanceRaw::desc()],
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
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        texture_layout: &resource::TextureBindGroupLayout,
        camera_layout: &resource::CameraBindGroupLayout,
        light_layout: &resource::LightBindGroupLayout,
        phong_layout: &resource::PhongBindGroupLayout,
    ) -> anyhow::Result<Self> {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ColorNormalPipelineLayout"),
            bind_group_layouts: &[
                &camera_layout.0,
                &light_layout.0,
                &phong_layout.0,
                texture_layout, // color texture
                texture_layout, // normal texture
            ],
            push_constant_ranges: &[],
        });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("ColorNormalShader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
        };
        let pipeline = create_render_pipeline(
            device,
            &layout,
            color_format,
            Some(resource::DepthTexture::DEPTH_FORMAT),
            &[layout::VertexRaw::desc(), layout::InstanceRaw::desc()],
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
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        camera_layout: &resource::CameraBindGroupLayout,
        light_layout: &resource::LightBindGroupLayout,
    ) -> anyhow::Result<Self> {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Light Pipeline Layout"),
            bind_group_layouts: &[&camera_layout.0, &light_layout.0],
            push_constant_ranges: &[],
        });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("Light Shader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
        };
        let pipeline = create_render_pipeline(
            device,
            &layout,
            color_format,
            Some(resource::DepthTexture::DEPTH_FORMAT),
            &[layout::VertexRaw::desc(), layout::InstanceRaw::desc()],
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

    pub fn new(
        device: &wgpu::Device,
        camera_layout: &resource::CameraBindGroupLayout,
    ) -> anyhow::Result<Self> {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[&camera_layout.0],
            push_constant_ranges: &[],
        });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Shader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
        };
        let shader = device.create_shader_module(shader);

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[layout::VertexRaw::desc(), layout::InstanceRaw::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: None,
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
                format: DepthTexture::DEPTH_FORMAT,
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
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layout: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    label: Option<&str>,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                format: color_format,
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
