use nalgebra as na;

use crate::{
    model,
    render::{GpuTransfer, LightRaw},
};

#[derive(Debug, Copy, Clone)]
pub struct Light {
    pos: na::Vector3<f32>,
    max_dist: f32,
    color: model::Color,
    rotational_speed: f32,
    paused: bool,
}

impl Light {
    pub fn new(pos: na::Vector3<f32>, color: model::Color) -> Self {
        let rotational_speed = 90.;
        let max_dist = 100.;
        let paused = true;
        Self {
            pos,
            max_dist,
            color,
            rotational_speed,
            paused,
        }
    }

    pub fn step(&mut self, epsilon: f32) -> bool {
        if self.paused {
            return false;
        }
        let angle = self.rotational_speed * epsilon;
        let angle = angle * std::f32::consts::PI / 180.;
        let axis = na::Vector3::<f32>::y_axis();
        let rotation = na::Rotation3::from_axis_angle(&axis, angle);
        self.pos = rotation * self.pos;
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
        let Self {
            pos,
            color,
            max_dist,
            ..
        } = *self;
        Self::Raw {
            position: pos.into(),
            max_dist,
            color: color.into(),
            _padding: 0,
        }
    }
}
