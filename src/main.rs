mod app;

use anyhow::Result;
use app::App;

fn main() -> Result<()> {
    env_logger::init();

    pollster::block_on(App::run())
}
