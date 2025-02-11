use anyhow::Result;
use radiosity::app::App;

fn main() -> Result<()> {
    env_logger::init();

    App::run()
}
