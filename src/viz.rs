#[macro_use] extern crate conrod;
extern crate tinn;

fn main() {
    feature::main();
}

mod feature {
    extern crate find_folder;
    extern crate image;
    use conrod::{self, widget, Colorable, Positionable, Sizeable, Widget, color};
    use conrod::backend::glium::glium;
    use conrod::backend::glium::glium::Surface;
    use std;
    use tinn::*;


    pub struct EventLoop {
        ui_needs_update: bool,
        last_update: std::time::Instant,
    }

    impl EventLoop {

        pub fn new() -> Self {
            EventLoop {
                last_update: std::time::Instant::now(),
                ui_needs_update: true,
            }
        }

        /// Produce an iterator yielding all available events.
        pub fn next(&mut self, events_loop: &mut glium::glutin::EventsLoop) -> Vec<glium::glutin::Event> {
            // We don't want to loop any faster than 60 FPS, so wait until it has been at least 16ms
            // since the last yield.
            let last_update = self.last_update;
            let sixteen_ms = std::time::Duration::from_millis(16);
            let duration_since_last_update = std::time::Instant::now().duration_since(last_update);
            if duration_since_last_update < sixteen_ms {
                std::thread::sleep(sixteen_ms - duration_since_last_update);
            }

            // Collect all pending events.
            let mut events = Vec::new();
            events_loop.poll_events(|event| events.push(event));

            // If there are no events and the `Ui` does not need updating, wait for the next event.
            if events.is_empty() && !self.ui_needs_update {
                events_loop.run_forever(|event| {
                    events.push(event);
                    glium::glutin::ControlFlow::Break
                });
            }

            self.ui_needs_update = false;
            self.last_update = std::time::Instant::now();

            events
        }

        /// Notifies the event loop that the `Ui` requires another update whether or not there are any
        /// pending events.
        ///
        /// This is primarily used on the occasion that some part of the `Ui` is still animating and
        /// requires further updates to do so.
        pub fn needs_update(&mut self) {
            self.ui_needs_update = true;
        }

    }

    pub fn main() {
        const WIDTH: u32 = 150;
        const HEIGHT: u32 = 150;

        // Build the window.
        let mut events_loop = glium::glutin::EventsLoop::new();
        let window = glium::glutin::WindowBuilder::new()
            .with_title("Image Widget Demonstration")
            .with_dimensions(WIDTH, HEIGHT);
        let context = glium::glutin::ContextBuilder::new()
            .with_vsync(true)
            .with_multisampling(4);
        let display = glium::Display::new(window, context, &events_loop).unwrap();

        // construct our `Ui`.
        let mut ui = conrod::UiBuilder::new([WIDTH as f64, HEIGHT as f64]).build();

        // A type used for converting `conrod::render::Primitives` into `Command`s that can be used
        // for drawing to the glium `Surface`.
        let mut renderer = conrod::backend::glium::Renderer::new(&display).unwrap();

        // The `WidgetId` for our background and `Image` widgets.
        widget_ids!(struct Ids { background, digit_image });
        let ids = Ids::new(ui.widget_id_generator());

        // Create our `conrod::image::Map` which describes each of our widget->image mappings.
        // In our case we only have one image, however the macro may be used to list multiple.
        let data = Data::build("semeion.data").unwrap();
        let mut img_idx = 0;

        let digit_image = load_semeion_image(&data.inp[img_idx], &display);
        let (w, h) = (digit_image.get_width(), digit_image.get_height().unwrap());
        let mut image_map = conrod::image::Map::new();
        let digit_image_id = image_map.insert(digit_image);

        // Poll events from the window.
        let mut event_loop = EventLoop::new();
        'main: loop {

            // Handle all events.
            for event in event_loop.next(&mut events_loop) {

                // Use the `winit` backend feature to convert the winit event to a conrod one.
                if let Some(event) = conrod::backend::winit::convert_event(event.clone(), &display) {
                    ui.handle_event(event);
                }

                match event {
                    glium::glutin::Event::WindowEvent { event, .. } => match event {
                        // Break from the loop upon `Escape`.
                        glium::glutin::WindowEvent::Closed | glium::glutin::WindowEvent::KeyboardInput {
                            input: glium::glutin::KeyboardInput {
                                virtual_keycode: Some(glium::glutin::VirtualKeyCode::Escape),
                                ..
                            },
                            ..
                        } => break 'main,
                        glium::glutin::WindowEvent::KeyboardInput {
                            input: glium::glutin::KeyboardInput {
                                state: glium::glutin::ElementState::Released,
                                virtual_keycode: Some(glium::glutin::VirtualKeyCode::Right), ..
                            }, ..
                        } => {
                            println!("RIGHT");
                            if img_idx < data.inp.len() {
                                img_idx += 1;
                                let digit_image = load_semeion_image(&data.inp[img_idx], &display);
                                let digit_image_id = image_map.replace(digit_image_id, digit_image).unwrap();
                                ui.needs_redraw();
                            }
                        },
                        glium::glutin::WindowEvent::KeyboardInput {
                            input: glium::glutin::KeyboardInput {
                                state: glium::glutin::ElementState::Released,
                                virtual_keycode: Some(glium::glutin::VirtualKeyCode::Left), ..
                            }, ..
                        } => {
                            println!("LEFT");
                            if img_idx > 0 {
                                img_idx -= 1;
                                let digit_image = load_semeion_image(&data.inp[img_idx], &display);
                                let digit_image_id = image_map.replace(digit_image_id, digit_image).unwrap();
                                ui.needs_redraw();
                            }
                        },
                        _ => (),
                    },
                    _ => (),
                }
            }

            // Instantiate the widgets.
            {
                let ui = &mut ui.set_widgets();
                // Draw a light blue background.
                widget::Canvas::new().color(color::BLACK).set(ids.background, ui);
                // Instantiate the `Image` at its full size in the middle of the window.
                widget::Image::new(digit_image_id).w_h(w as f64, h as f64).middle().set(ids.digit_image, ui);
            }

            // Render the `Ui` and then display it on the screen.
            if let Some(primitives) = ui.draw_if_changed() {
                renderer.fill(&display, primitives, &image_map);
                let mut target = display.draw();
                target.clear_color(0.0, 0.0, 0.0, 1.0);
                renderer.draw(&display, &mut target, &image_map).unwrap();
                target.finish().unwrap();
            }
        }
    }

    // Convert a row from the Semeion set into a texture
    fn load_semeion_image(row: &Vec<f64>, display: &glium::Display) -> glium::texture::Texture2d {
        let img = image::ImageBuffer::from_fn(16_u32, 16_u32, |x, y| {
            let px = row[(x+16*y) as usize];
            if px == 1.0 {
                image::Rgb([255u8, 255u8, 255u8])
            } else {
                image::Rgb([0u8, 0u8, 0u8])
            }
        });
//        println!("[load img] have an RGB image with dims: {:?}", img.dimensions());
        let raw_image = glium::texture::RawImage2d::from_raw_rgb_reversed(&img.into_raw(), (16,16));
        let texture = glium::texture::Texture2d::new(display, raw_image).unwrap();
        texture
    }
}
