use crate::{camera, light, model};

#[allow(unused)]
pub struct SceneState {
    paused: bool,
    last_time: std::time::Instant,
    obj_model: model::Model,
    instances: Vec<model::Instance>,
    camera: camera::Camera,
    light: light::Light,
}

#[allow(unused)]
impl SceneState {
    pub fn new() -> Self {
        todo!()
    }
}
