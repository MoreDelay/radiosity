use std::ops::Deref;

use crate::render::{GpuContext, TargetContext, resource::PhongLayouts};

use super::{raw, resource};

#[derive(Debug, Copy, Clone)]
pub struct PhongCapabilites {
    pub color_map: bool,
    pub normal_map: bool,
}

#[derive(Debug, Copy, Clone)]
pub struct PhongVertexRequirements {
    pub position: Option<u32>,
    pub tex_coord: Option<u32>,
    pub normal: Option<u32>,
    pub tangent: Option<u32>,
    pub instance: Option<u32>,
}

impl PhongVertexRequirements {
    fn create_vertex_layout(&self) -> Vec<wgpu::VertexBufferLayout<'static>> {
        let mut layout: Vec<(u32, wgpu::VertexBufferLayout)> = Vec::with_capacity(6);

        if let Some(position) = self.position {
            layout.push((position, raw::position_desc()));
        }
        if let Some(tex_coord) = self.tex_coord {
            layout.push((tex_coord, raw::tex_coord_desc()));
        }
        if let Some(normal) = self.normal {
            layout.push((normal, raw::normal_desc()));
        }
        if let Some(tangent) = self.tangent {
            layout.push((tangent, raw::tangent_desc()));
        }

        if let Some(instance) = self.instance {
            layout.push((instance, raw::InstanceRaw::desc()));
        }

        layout.sort_by_key(|(index, _)| *index);
        layout.into_iter().map(|(_, layout)| layout).collect()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PhongBindingRequirements {
    pub camera: Option<u32>,
    pub shadow_transform: Option<u32>,
    pub light: Option<u32>,
    pub phong: Option<u32>,
    pub shadow_texture: Option<u32>,
    pub color_texture: Option<u32>,
    pub normal_texture: Option<u32>,
}

impl PhongBindingRequirements {
    fn create_bind_group_layouts<'a>(
        &self,
        layouts: &'a PhongLayouts,
    ) -> Vec<&'a wgpu::BindGroupLayout> {
        assert!(
            !(self.camera.is_some() && self.shadow_transform.is_some()),
            "did not expect both camera and shadow transform to be used in one shader pass"
        );

        let mut layout: Vec<(u32, &'a wgpu::BindGroupLayout)> = Vec::with_capacity(6);

        if let Some(camera) = self.camera {
            layout.push((camera, &layouts.camera));
        }
        if let Some(shadow_transform) = self.shadow_transform {
            layout.push((shadow_transform, &layouts.shadow_transform));
        }
        if let Some(light) = self.light {
            layout.push((light, &layouts.light));
        }
        if let Some(phong) = self.phong {
            layout.push((phong, &layouts.phong));
        }
        if let Some(shadow_texture) = self.shadow_texture {
            layout.push((shadow_texture, &layouts.shadow_texture));
        }
        if let Some(color_texture) = self.color_texture {
            layout.push((color_texture, &layouts.texture));
        }
        if let Some(normal_texture) = self.normal_texture {
            layout.push((normal_texture, &layouts.texture));
        }

        layout.sort_by_key(|(index, _)| *index);
        layout.into_iter().map(|(_, binding)| binding).collect()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PhongResourceRequirements {
    pub vertex: PhongVertexRequirements,
    pub bindings: PhongBindingRequirements,
}

#[derive(Debug)]
pub struct PhongPipeline {
    pipeline: wgpu::RenderPipeline,
    requirements: PhongResourceRequirements,
}

impl PhongPipeline {
    const ENTRY: &str = "package::phong";

    pub fn new(
        ctx: &GpuContext,
        target: &TargetContext,
        layouts: &PhongLayouts,
        capabilities: PhongCapabilites,
    ) -> Self {
        let PhongCapabilites {
            color_map,
            normal_map,
        } = capabilities;

        let bindings = {
            let mut slot_iter = 0..;

            PhongBindingRequirements {
                camera: Some(slot_iter.next().unwrap()),
                shadow_transform: None,
                light: Some(slot_iter.next().unwrap()),
                phong: Some(slot_iter.next().unwrap()),
                shadow_texture: Some(slot_iter.next().unwrap()),
                color_texture: color_map.then(|| slot_iter.next().unwrap()),
                normal_texture: normal_map.then(|| slot_iter.next().unwrap()),
            }
        };

        let vertex = {
            let mut slot_iter = 0..;

            PhongVertexRequirements {
                position: Some(slot_iter.next().unwrap()),
                tex_coord: match color_map || normal_map {
                    true => Some(slot_iter.next().unwrap()),
                    false => None,
                },
                normal: Some(slot_iter.next().unwrap()),
                tangent: Some(slot_iter.next().unwrap()),
                instance: Some(slot_iter.next().unwrap()),
            }
        };
        let requirements = PhongResourceRequirements { vertex, bindings };

        let bind_group_layouts = requirements.bindings.create_bind_group_layouts(layouts);
        let vertex_layout = requirements.vertex.create_vertex_layout();

        let mut features = vec![
            ("HAS_NORMAL", true),
            ("HAS_TANGENT", true),
            ("USE_SHADOW_MAP", true),
        ];
        let mut constants = vec![(
            "SHADOW_MAP_SLOT",
            requirements.bindings.shadow_texture.unwrap() as f64,
        )];

        if requirements.vertex.tex_coord.is_some() {
            features.push(("HAS_TEX_COORD", true));
        }
        if let Some(slot) = requirements.bindings.color_texture {
            features.push(("USE_COLOR_MAP", true));
            constants.push(("COLOR_MAP_SLOT", slot as f64));
        }
        if let Some(slot) = requirements.bindings.normal_texture {
            features.push(("USE_NORMAL_MAP", true));
            constants.push(("NORMAL_MAP_SLOT", slot as f64));
        }

        let shader = wesl::Wesl::new(super::SHADER_ROOT)
            .set_features(features.clone())
            .add_constants(constants.clone())
            .compile(&Self::ENTRY.parse().unwrap())
            .inspect_err(|e| eprintln!("WESL error: {e}"))
            .unwrap()
            .to_string();
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("ColorNormalShader"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        };
        let shader = ctx.device.create_shader_module(shader);

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ColorNormalPipelineLayout"),
                bind_group_layouts: &bind_group_layouts,
                push_constant_ranges: &[],
            });
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Pipeline {capabilities:?}"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_layout,
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
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: resource::DepthTexture::FORMAT,
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
            });
        Self {
            pipeline,
            requirements,
        }
    }

    pub fn requirements(&self) -> PhongResourceRequirements {
        self.requirements
    }
}

impl Deref for PhongPipeline {
    type Target = wgpu::RenderPipeline;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

#[derive(Debug)]
pub struct PhongPipelines {
    render: [PhongPipeline; Self::RENDER_VARIANTS as usize + 1],
    light: LightPipeline,
    shadow: ShadowPipeline,
}

impl PhongPipelines {
    const COLOR_MAP_BIT: u32 = 0;
    const NORMAL_MAP_BIT: u32 = 1;
    const FIRST_UNUSED_BIT: u32 = 2;

    const RENDER_VARIANTS: u32 = (1 << Self::FIRST_UNUSED_BIT) - 1;

    pub fn new(ctx: &GpuContext, target: &TargetContext, layouts: &PhongLayouts) -> Self {
        let render = (0..=Self::RENDER_VARIANTS)
            .map(|index| PhongPipeline::new(ctx, target, layouts, Self::index_to_caps(index)))
            .collect::<Vec<_>>();
        let render = render.try_into().unwrap();
        let light = LightPipeline::new(ctx, target, layouts);
        let shadow = ShadowPipeline::new(ctx, layouts);
        Self {
            render,
            light,
            shadow,
        }
    }

    pub fn get_render(&self, caps: PhongCapabilites) -> &PhongPipeline {
        let index = Self::caps_to_index(caps);
        &self.render[index as usize]
    }

    pub fn get_light(&self) -> &LightPipeline {
        &self.light
    }

    pub fn get_shadow(&self) -> &ShadowPipeline {
        &self.shadow
    }

    fn index_to_caps(index: u32) -> PhongCapabilites {
        let color_map = index & (1 << Self::COLOR_MAP_BIT) != 0;
        let normal_map = index & (1 << Self::NORMAL_MAP_BIT) != 0;
        PhongCapabilites {
            color_map,
            normal_map,
        }
    }

    fn caps_to_index(caps: PhongCapabilites) -> u32 {
        let mut index = 0;
        index |= (caps.color_map as u32) << Self::COLOR_MAP_BIT;
        index |= (caps.normal_map as u32) << Self::NORMAL_MAP_BIT;
        index
    }
}

#[derive(Debug)]
pub struct LightPipeline(pub PhongPipeline);

impl LightPipeline {
    pub fn new(ctx: &GpuContext, target: &TargetContext, layouts: &PhongLayouts) -> Self {
        let vertex = {
            let mut slot_iter = 0..;

            PhongVertexRequirements {
                position: Some(slot_iter.next().unwrap()),
                tex_coord: None,
                normal: None,
                tangent: None,
                instance: Some(slot_iter.next().unwrap()),
            }
        };

        let bindings = {
            let mut slot_iter = 0..;

            PhongBindingRequirements {
                camera: Some(slot_iter.next().unwrap()),
                shadow_transform: None,
                light: None,
                phong: Some(slot_iter.next().unwrap()),
                shadow_texture: None,
                color_texture: None,
                normal_texture: None,
            }
        };
        let requirements = PhongResourceRequirements { vertex, bindings };

        let vertex_layout = requirements.vertex.create_vertex_layout();

        let shader = wesl::Wesl::new(super::SHADER_ROOT)
            .compile(&PhongPipeline::ENTRY.parse().unwrap())
            .inspect_err(|e| eprintln!("WESL error: {e}"))
            .unwrap()
            .to_string();
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("Light Shader"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        };
        let shader = ctx.device.create_shader_module(shader);

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&layouts.camera, &layouts.light, &layouts.phong],
                push_constant_ranges: &[],
            });
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Light Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_layout,
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
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: resource::DepthTexture::FORMAT,
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
            });

        Self(PhongPipeline {
            pipeline,
            requirements,
        })
    }

    #[expect(unused)]
    pub fn requirements(&self) -> PhongResourceRequirements {
        let vertex = PhongVertexRequirements {
            position: Some(0),
            tex_coord: None,
            normal: None,
            tangent: None,
            instance: Some(1),
        };
        let bindings = PhongBindingRequirements {
            camera: Some(0),
            shadow_transform: None,
            light: Some(1),
            phong: None,
            shadow_texture: None,
            color_texture: None,
            normal_texture: None,
        };
        PhongResourceRequirements { vertex, bindings }
    }
}

impl Deref for LightPipeline {
    type Target = PhongPipeline;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug)]
pub struct ShadowPipeline(wgpu::RenderPipeline);

impl ShadowPipeline {
    const SHADER: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/shaders/shadow.wgsl"
    ));

    pub fn new(ctx: &GpuContext, layouts: &PhongLayouts) -> Self {
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
                    buffers: &[raw::position_desc(), raw::InstanceRaw::desc()],
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
        Self(pipeline)
    }

    #[expect(unused)]
    pub fn requirements(&self) -> PhongResourceRequirements {
        let vertex = PhongVertexRequirements {
            position: Some(0),
            tex_coord: None,
            normal: None,
            tangent: None,
            instance: Some(1),
        };
        let bindings = PhongBindingRequirements {
            camera: None,
            shadow_transform: Some(0),
            light: Some(1),
            phong: None,
            shadow_texture: None,
            color_texture: None,
            normal_texture: None,
        };
        PhongResourceRequirements { vertex, bindings }
    }
}

impl Deref for ShadowPipeline {
    type Target = wgpu::RenderPipeline;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
