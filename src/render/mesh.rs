use crate::{model, render};

use wgpu::util::DeviceExt;
use zerocopy::IntoBytes;

pub struct InstanceBuffer {
    pub buffer: wgpu::Buffer,
    pub num_instances: u32,
}

impl InstanceBuffer {
    pub fn new(
        ctx: &render::GpuContext,
        instances: &[model::Instance],
        label: Option<&str>,
    ) -> Self {
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
    pub _range: std::ops::Range<u32>,
}

impl IndexBuffer {
    pub fn new(ctx: &render::GpuContext, data: &[[u32; 3]], label: Option<&str>) -> Self {
        let count = data.len() as u32;
        let min = data.iter().flatten().copied().reduce(|a, b| a.min(b));
        let max = data.iter().flatten().copied().reduce(|a, b| a.max(b));
        let range = match (min, max) {
            (Some(min), Some(max)) => min..max,
            _ => 0..0,
        };
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
            _range: range,
        }
    }
}

pub(super) struct PositionBuffer {
    pub buffer: wgpu::Buffer,
    pub _count: u32,
}

impl PositionBuffer {
    pub fn new(ctx: &render::GpuContext, data: &[f32], label: Option<&str>) -> Self {
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
    pub fn new(ctx: &render::GpuContext, data: &[f32], label: Option<&str>) -> Self {
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
    pub fn new(ctx: &render::GpuContext, data: &[f32], label: Option<&str>) -> Self {
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
    pub fn new(ctx: &render::GpuContext, data: &[f32], label: Option<&str>) -> Self {
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
    pub fn new(ctx: &render::GpuContext, data: &[f32], label: Option<&str>) -> Self {
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
