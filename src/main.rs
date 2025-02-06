mod app;
mod camera;
mod light;
mod model;
mod texture;

use anyhow::Result;
use app::App;

fn main() -> Result<()> {
    env_logger::init();

    pollster::block_on(App::run())
}
