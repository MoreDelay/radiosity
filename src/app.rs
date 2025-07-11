use std::sync::Arc;

use anyhow::Result;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use crate::{camera, scene::SceneState};

#[derive(Default)]
pub struct App {
    window: Option<Arc<Window>>,
    scene: Option<SceneState>,
    mouse: Option<MouseState>,
}

#[derive(Copy, Clone, Debug)]
struct ScreenPos {
    x: f64,
    y: f64,
}

enum MouseState {
    OnScreen { pos: ScreenPos, drag: bool },
    OutsideScreen { last_pos: ScreenPos },
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
            mouse,
        } = self
        else {
            unreachable!("window and scene state should always be available on window_event")
        };

        if window_id != window.id() {
            return;
        }

        match event {
            // interactions with window
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
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
                    Err(Timeout) => log::warn!("Surface Timeout"),
                    Err(Other) => log::warn!("Got a generic error"),
                }
            }
            WindowEvent::Resized(size) => {
                scene.resize_window(Some(size));
            }

            // interactions with mouse
            WindowEvent::CursorLeft { .. } => {
                let Some(MouseState::OnScreen { pos: last_pos, .. }) = mouse.take() else {
                    unreachable!("last known state should have mouse on screen")
                };
                *mouse = Some(MouseState::OutsideScreen { last_pos });
            }
            WindowEvent::CursorEntered { .. } => {
                let pos = match mouse.take() {
                    None => ScreenPos { x: 0., y: 0. },
                    Some(MouseState::OutsideScreen { last_pos }) => last_pos,
                    Some(MouseState::OnScreen { .. }) => {
                        unreachable!("mouse just entered and should not be already on screen")
                    }
                };
                *mouse = Some(MouseState::OnScreen { pos, drag: false });
            }
            WindowEvent::CursorMoved {
                position: winit::dpi::PhysicalPosition { x, y },
                ..
            } => {
                let pos = ScreenPos { x, y };
                let (last_pos, drag) = match mouse.take() {
                    Some(MouseState::OnScreen { pos, drag }) => (pos, drag),
                    None | Some(MouseState::OutsideScreen { .. }) => {
                        unreachable!("should have mouse on screen to get move event")
                    }
                };
                if drag {
                    let from = cgmath::Point2 {
                        x: last_pos.x as f32,
                        y: last_pos.y as f32,
                    };
                    let to = cgmath::Point2 {
                        x: pos.x as f32,
                        y: pos.y as f32,
                    };
                    let vec = to - from;
                    scene.drag_camera(vec);
                }
                *mouse = Some(MouseState::OnScreen { pos, drag });
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                let Some(MouseState::OnScreen { pos, .. }) = mouse.take() else {
                    unreachable!("should have mouse on screen to get click event");
                };
                *mouse = Some(MouseState::OnScreen { pos, drag: true });
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                let Some(MouseState::OnScreen { pos, .. }) = mouse.take() else {
                    unreachable!("should have mouse on screen to get click event");
                };
                *mouse = Some(MouseState::OnScreen { pos, drag: false });
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta {
                    MouseScrollDelta::LineDelta(_x, y) => y as f64,
                    MouseScrollDelta::PixelDelta(winit::dpi::PhysicalPosition { y, .. }) => y,
                };
                if delta > 0. {
                    scene.go_near()
                } else if delta < 0. {
                    scene.go_away()
                }
            }

            // interactions with keyboard
            WindowEvent::KeyboardInput { event, .. } => match event {
                KeyEvent {
                    physical_key: PhysicalKey::Code(key @ (KeyCode::Escape | KeyCode::KeyQ)),
                    state: ElementState::Pressed,
                    ..
                } => {
                    println!("pressed {key:?}; closing");
                    event_loop.exit();
                }
                KeyEvent {
                    physical_key: PhysicalKey::Code(KeyCode::KeyP),
                    state: ElementState::Pressed,
                    ..
                } => {
                    let paused = scene.toggle_pause();
                    println!(
                        "toggle pause: {}",
                        if paused { "paused" } else { "running" }
                    );
                }
                KeyEvent {
                    physical_key: PhysicalKey::Code(KeyCode::KeyT),
                    state: ElementState::Pressed,
                    ..
                } => {
                    use crate::render::PipelineMode::*;
                    let mode = scene.toggle_pipeline();
                    println!(
                        "toggle pipeline: {}",
                        match mode {
                            Flat => "flat",
                            Color => "color",
                            Normal => "normal",
                        }
                    );
                }
                KeyEvent {
                    physical_key: PhysicalKey::Code(KeyCode::KeyC),
                    state: ElementState::Pressed,
                    ..
                } => {
                    let first_person = scene.toggle_camera();
                    println!(
                        "toggle camera: {}",
                        if first_person {
                            "first person"
                        } else {
                            "target"
                        }
                    );
                }
                KeyEvent {
                    physical_key:
                        PhysicalKey::Code(
                            key @ (KeyCode::KeyW | KeyCode::KeyA | KeyCode::KeyS | KeyCode::KeyD),
                        ),
                    state: state @ (ElementState::Pressed | ElementState::Released),
                    ..
                } => {
                    let active = matches!(state, ElementState::Pressed);
                    match key {
                        KeyCode::KeyW => scene.set_movement(camera::DirectionKey::W, active),
                        KeyCode::KeyA => scene.set_movement(camera::DirectionKey::A, active),
                        KeyCode::KeyS => scene.set_movement(camera::DirectionKey::S, active),
                        KeyCode::KeyD => scene.set_movement(camera::DirectionKey::D, active),
                        _ => unreachable!(),
                    }
                }
                KeyEvent {
                    physical_key: PhysicalKey::Code(KeyCode::KeyH),
                    state: ElementState::Pressed,
                    ..
                } => println!("{}", App::help()),
                _ => (),
            },

            _ => (),
        }
    }
}

impl App {
    pub fn run() -> Result<()> {
        println!("{}", App::help());

        let event_loop = EventLoop::new()?;
        // use Poll when you want to redraw continuously
        event_loop.set_control_flow(ControlFlow::Wait);

        let mut app = App::default();
        event_loop.run_app(&mut app)?;
        Ok(())
    }

    pub fn help() -> String {
        r#"
Controls:
q: quit
h: show this message again
t: cycle through textures
c: toggle between satelite and first-person camera
p: toggle light movement
wasd: move in first-person view
"#
        .to_string()
    }
}
