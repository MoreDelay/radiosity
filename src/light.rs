use cgmath::{EuclideanSpace, Rotation3};

use crate::render::layout::{GpuTransfer, LightRaw};

#[derive(Debug, Copy, Clone)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[derive(Debug, Copy, Clone)]
pub struct Light {
    pos: cgmath::Point3<f32>,
    color: Color,
    rotational_speed: f32,
    paused: bool,
}

impl Light {
    pub fn new(pos: cgmath::Point3<f32>, color: Color) -> Self {
        let rotational_speed = 90.;
        let paused = false;
        Self {
            pos,
            color,
            rotational_speed,
            paused,
        }
    }

    pub fn step(&mut self, epsilon: f32) -> bool {
        if self.paused {
            return false;
        }
        let rotational_distance = self.rotational_speed * epsilon;
        let old_pos = self.pos.to_vec();
        let new_pos = cgmath::Quaternion::from_axis_angle(
            cgmath::Vector3::unit_y(),
            cgmath::Deg(rotational_distance),
        ) * old_pos;
        self.pos = cgmath::Point3::from_vec(new_pos);
        true
    }

    pub fn toggle_pause(&mut self) -> bool {
        self.paused = !self.paused;
        self.paused
    }
}

impl GpuTransfer for Light {
    type Raw = LightRaw;
    fn to_raw(&self) -> Self::Raw {
        let Self { pos, color, .. } = *self;
        Self::Raw {
            position: pos.into(),
            _padding: 0,
            color: color.into(),
            _padding2: 0,
        }
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
