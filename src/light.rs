use cgmath::{EuclideanSpace, Rotation3};

use crate::{
    primitives,
    render::{GpuTransfer, LightRaw},
};

#[derive(Debug, Copy, Clone)]
pub struct Light {
    pos: cgmath::Point3<f32>,
    color: primitives::Color,
    rotational_speed: f32,
    paused: bool,
}

impl Light {
    pub fn new(pos: cgmath::Point3<f32>, color: primitives::Color) -> Self {
        let rotational_speed = 90.;
        let paused = true;
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
