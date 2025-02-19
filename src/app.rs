use std::sync::Arc;

use anyhow::Result;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use crate::scene::SceneState;

#[derive(Default)]
pub struct App {
    window: Option<Arc<Window>>,
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
            let scene = SceneState::new(window.clone());
            self.scene = Some(scene);
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
            scene: Some(scene),
        } = self
        else {
            unreachable!("window and scene state should always be available on window_event")
        };

        if window_id != window.id() {
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
                    let paused = scene.toggle_pause();
                    println!(
                        "toggle pause: {}",
                        if paused { "paused" } else { "running" }
                    );
                }
                _ => (),
            },
            RedrawRequested => {
                // request another redraw after this
                window.request_redraw();
                scene.step();

                use wgpu::SurfaceError::*;
                match scene.draw() {
                    Ok(()) => {}
                    Err(Lost | Outdated) => scene.resize_window(None),
                    Err(OutOfMemory) => {
                        log::error!("OutOfMemory");
                        event_loop.exit();
                    }
                    Err(Timeout) => {
                        log::warn!("Surface Timeout")
                    }
                }
            }
            Resized(size) => {
                scene.resize_window(Some(size));
            }
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
