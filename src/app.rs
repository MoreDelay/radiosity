use std::sync::Arc;

use anyhow::Result;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use crate::{render::RenderState, scene::SceneState};

#[derive(Default)]
pub struct App {
    window: Option<Arc<Window>>,
    render: Option<RenderState>,
    scene: Option<SceneState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes())
                    .unwrap(),
            );
            self.window = Some(window.clone());
            let state = pollster::block_on(RenderState::new(window.clone()))
                .expect("should create new scene");
            self.render = Some(state);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Self {
            window: Some(window),
            render: Some(state),
            scene: None,
        } = self
        else {
            println!("unreachable? no window or state on window_event");
            return;
        };

        if window_id != state.window.id() || state.check_completed(&event) {
            return;
        }

        use WindowEvent::*;
        match event {
            CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            KeyboardInput { event, .. } => match event {
                KeyEvent {
                    physical_key: PhysicalKey::Code(key @ (KeyCode::Escape | KeyCode::KeyQ)),
                    state: ElementState::Pressed,
                    ..
                } => {
                    println!("pressed {key:?}; closing");
                    event_loop.exit();
                }
                KeyEvent {
                    physical_key: PhysicalKey::Code(KeyCode::Space),
                    state: ElementState::Pressed,
                    ..
                } => {
                    state.paused ^= true;
                    println!(
                        "toggle pause: {}",
                        if state.paused { "paused" } else { "running" }
                    );
                }
                _ => (),
            },
            RedrawRequested => {
                // request another redraw after this
                window.request_redraw();
                state.update();

                use wgpu::SurfaceError::*;
                match state.render() {
                    Ok(()) => {}
                    Err(Lost | Outdated) => state.resize(state.size),
                    Err(OutOfMemory) => {
                        log::error!("OutOfMemory");
                        event_loop.exit();
                    }
                    Err(Timeout) => {
                        log::warn!("Surface Timeout")
                    }
                }
            }
            Resized(size) => state.resize(size),
            _ => (),
        }
    }
}

impl App {
    pub fn run() -> Result<()> {
        let event_loop = EventLoop::new()?;
        // use Poll when you want to redraw continuously
        event_loop.set_control_flow(ControlFlow::Wait);

        let mut app = App::default();
        event_loop.run_app(&mut app)?;
        Ok(())
    }
}
