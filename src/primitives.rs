use crate::{
    model::parser::{self, MtlKa},
    render::layout::{GpuTransfer, VertexRaw},
};

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: cgmath::Point3<f32>,
    pub tex_coords: cgmath::Point2<f32>,
    pub normal: cgmath::Vector3<f32>,
    pub tangent: cgmath::Vector3<f32>,
    pub bitangent: cgmath::Vector3<f32>,
}

#[derive(Debug, Copy, Clone)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl GpuTransfer for Vertex {
    type Raw = VertexRaw;
    fn to_raw(&self) -> Self::Raw {
        VertexRaw {
            position: self.position.into(),
            tex_coords: self.tex_coords.into(),
            normal: self.normal.into(),
            tangent: self.tangent.into(),
            bitangent: self.bitangent.into(),
        }
    }
}

impl From<Color> for [f32; 3] {
    fn from(val: Color) -> Self {
        let Color { r, g, b, .. } = val;
        [r as f32 / 255., g as f32 / 255., b as f32 / 255.]
    }
}

impl From<MtlKa> for Color {
    fn from(MtlKa(r, g, b): MtlKa) -> Self {
        let r = (r * 255.) as u8;
        let g = (g * 255.) as u8;
        let b = (b * 255.) as u8;
        Color { r, g, b }
    }
}

impl From<parser::MtlKd> for Color {
    fn from(parser::MtlKd(r, g, b): parser::MtlKd) -> Self {
        let r = (r * 255.) as u8;
        let g = (g * 255.) as u8;
        let b = (b * 255.) as u8;
        Color { r, g, b }
    }
}

impl From<parser::MtlKs> for Color {
    fn from(parser::MtlKs(r, g, b): parser::MtlKs) -> Self {
        let r = (r * 255.) as u8;
        let g = (g * 255.) as u8;
        let b = (b * 255.) as u8;
        Color { r, g, b }
    }
}
