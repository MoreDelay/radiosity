use crate::{
    model::parser::mtl,
    render::{GpuTransfer, VertexRaw},
};

use nalgebra as na;

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: na::Vector3<f32>,
    pub tex_coords: na::Vector2<f32>,
    pub normal: na::UnitVector3<f32>,
    pub tangent: na::UnitVector3<f32>,
    pub bitangent: na::UnitVector3<f32>,
}

#[derive(Debug, Copy, Clone)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl GpuTransfer for Vertex {
    type Raw = VertexRaw;
    fn to_raw(&self) -> Self::Raw {
        VertexRaw {
            position: self.position.into(),
            tex_coords: self.tex_coords.into(),
            normal: (*self.normal).into(),
            tangent: (*self.tangent).into(),
            bitangent: (*self.bitangent).into(),
        }
    }
}

impl From<Color> for [f32; 3] {
    fn from(val: Color) -> Self {
        let Color { r, g, b, .. } = val;
        [r, g, b]
    }
}

impl From<mtl::MtlKa> for Color {
    fn from(mtl::MtlKa(r, g, b): mtl::MtlKa) -> Self {
        Color { r, g, b }
    }
}

impl From<mtl::MtlKd> for Color {
    fn from(mtl::MtlKd(r, g, b): mtl::MtlKd) -> Self {
        Color { r, g, b }
    }
}

impl From<mtl::MtlKs> for Color {
    fn from(mtl::MtlKs(r, g, b): mtl::MtlKs) -> Self {
        Color { r, g, b }
    }
}
