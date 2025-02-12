use crate::render::{GpuTransfer, LightRaw};

#[derive(Debug, Copy, Clone)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[derive(Debug, Copy, Clone)]
pub struct Light {
    pub pos: cgmath::Point3<f32>,
    pub color: Color,
}

// Safety: LightRaw expects any values for position and any 3 values within [0, 1] for color
unsafe impl GpuTransfer for Light {
    type Raw = LightRaw;
    fn to_raw(&self) -> Self::Raw {
        let Self { pos, color } = *self;
        Self::Raw {
            position: pos.into(),
            _padding: 0,
            color: color.into(),
            _padding2: 0,
        }
    }
}

impl Light {
    pub fn new(pos: cgmath::Point3<f32>, color: Color) -> Self {
        Self { pos, color }
    }
}

impl From<Color> for [f32; 4] {
    fn from(val: Color) -> Self {
        let Color { r, g, b, a } = val;
        [
            r as f32 / 255.,
            g as f32 / 255.,
            b as f32 / 255.,
            a as f32 / 255.,
        ]
    }
}

impl From<Color> for [f32; 3] {
    fn from(val: Color) -> Self {
        let Color { r, g, b, .. } = val;
        [r as f32 / 255., g as f32 / 255., b as f32 / 255.]
    }
}
