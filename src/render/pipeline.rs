use super::{layout, resource};

pub struct ScenePipeline(pub wgpu::RenderPipeline);
pub struct LightPipeline(pub wgpu::RenderPipeline);

impl ScenePipeline {
    const SHADER: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/shaders/shader.wgsl"
    ));

    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        texture_layout: &resource::TextureBindGroupLayout,
        camera_layout: &resource::CameraBindGroupLayout,
        light_layout: &resource::LightBindGroupLayout,
    ) -> anyhow::Result<Self> {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&texture_layout.0, &camera_layout.0, &light_layout.0],
            push_constant_ranges: &[],
        });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("Scene Shader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER.into()),
        };
        let pipeline = create_render_pipeline(
            device,
            &layout,
            color_format,
            Some(resource::Texture::DEPTH_FORMAT),
            &[layout::VertexRaw::desc(), layout::InstanceRaw::desc()],
            shader,
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
            label: Some("Render Pipeline Layout"),
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
            Some(resource::Texture::DEPTH_FORMAT),
            &[layout::VertexRaw::desc(), layout::InstanceRaw::desc()],
            shader,
        );
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
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
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
