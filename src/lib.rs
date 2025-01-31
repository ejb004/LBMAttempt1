use std::f32::consts::LN_10;
use std::time::{Duration, Instant};
use std::{fs, iter, thread};

mod boundary;
mod node;
mod saving_data;

use boundary::{is_point_in_letter, BoundaryNode};
use image::ImageReader;
use node::Node;
use rand::Rng;
use wgpu::{util::DeviceExt, BindGroupLayoutDescriptor};
use wgpu::{BindGroupLayoutEntry, TextureViewDescriptor};
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{Key, NamedKey},
    window::{Window, WindowBuilder},
};

const SCALE: u32 = 6; // 2 is default
const NX: u32 = 64 * SCALE * 2;
const NY: u32 = 64 * SCALE;
const NZ: u32 = 1;
const NODES: u32 = NX * NY * NZ;
const WORKGROUP_SIZE: u32 = 16;
const WINDOW_SIZE: u32 = 2048;

const SUBSTEPS: u32 = 2_u32.pow(5);

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [0.5 * NX as f32 / NY as f32, 0.5, 0.0],
        uv: [1.0, 1.0],
    },
    Vertex {
        position: [-0.5 * NX as f32 / NY as f32, 0.5, 0.0],
        uv: [0.0, 1.0],
    },
    Vertex {
        position: [-0.5 * NX as f32 / NY as f32, -0.5, 0.0],
        uv: [0.0, 0.0],
    },
    Vertex {
        position: [0.5 * NX as f32 / NY as f32, -0.5, 0.0],
        uv: [1.0, 0.0],
    },
];

const INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];

struct FrameRateCounter {
    last_frame_time: Instant,
    last_print_time: Instant,
    frame_count: u32,
    fps: f32,
    iterations: u64,
}

impl FrameRateCounter {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            last_frame_time: now,
            last_print_time: now,
            frame_count: 0,
            fps: 0.0,
            iterations: 0,
        }
    }

    fn update(&mut self) {
        let now = Instant::now();
        let duration = now.duration_since(self.last_frame_time);
        self.last_frame_time = now;

        // Increment frame count for the FPS calculation
        self.frame_count += 1;

        // Update and print FPS every 1 second
        if now.duration_since(self.last_print_time).as_secs_f32() >= 1.0 {
            self.fps = self.frame_count as f32;
            let glups = (NX * NY) as f32 * self.fps / 1000000000.0;
            println!(
                "FPS: {:.2}; Iterations: {:}; GLUPS: {:.3}",
                self.fps, self.iterations, glups
            );
            self.frame_count = 0;
            self.last_print_time = now;
        }
    }

    fn get_fps(&self) -> f32 {
        self.fps
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    nodes_x: u32,
    nodes_y: u32,
    inlet_vel: [f32; 2],
    boundary_nodes: u32,
    walls: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VisualisationUniforms {
    nodes_x: u32,
    nodes_y: u32,
    mode: u32,
    min_value: f32,
    max_value: f32,
    boundary_nodes: u32,
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    window: Window,

    compute_pipeline: wgpu::ComputePipeline,
    compute_bg0: wgpu::BindGroup,
    compute_bg1: wgpu::BindGroup,
    compute_toggle: bool,

    render_bg: wgpu::BindGroup,
    frame_rate_counter: FrameRateCounter,

    count: u32,

    visualise_buffer: wgpu::Buffer,
    visualise_uniform: VisualisationUniforms,

    compute_buffer0: wgpu::Buffer,
}

impl State {
    async fn new(window: Window) -> Self {
        let mut frame_rate_counter = FrameRateCounter::new();

        let size = window.inner_size();
        window.focus_window();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an Srgb surface texture. Using a different
        // one will result all the colors comming out darker. If you want to support non
        // Srgb surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // ------------------- UNIFORM =---------------------------------------------------- ----------------------------  //
        // UNIFORM -------------

        let nodes: Vec<Node> = (0..NODES)
            .map(|i| {
                let mut rng = rand::thread_rng();
                let f: f32 = rng.gen();

                // Node::with_density(1.0 + y / 20.0)

                let x = i % NX;
                let y = i / NX;
                if x <= 1 || x >= NX - 2 || y <= 1 || y >= NY - 2 {
                    Node::with_density(1.0)
                } else {
                    Node::with_density(1.0 + f / 10.0)
                }
            })
            .collect();

        // boundaries
        let mut boundaries: Vec<BoundaryNode> = Vec::new();

        let mut boundary_array = [false; (NX * NY) as usize];

        // -================== AIRFOIL =======================
        // for x_i in 0..NX {
        //     for y_i in 0..NY {
        //         if boundary::sd_airfoil([x_i as f32, y_i as f32], center_x, center_y, 00.0, 40.0)
        //             <= 0.0
        //         {
        //             boundary_array[(x_i as u32 + y_i as u32 * NX) as usize] = true
        //         }
        //     }
        // }

        // -================== LBM =======================
        // for x_i in 0..NX {
        //     for y_i in 0..NY {
        //         let letter_spacing = 75.0; // Adjust spacing between letters
        //         let base_x = 256.0 - 1.5 * letter_spacing; // Center the text at x=256

        //         // Check if point is in any letter
        //         if is_point_in_letter(x_i as f32, y_i as f32, 'L', base_x, letter_spacing)
        //             || is_point_in_letter(x_i as f32, y_i as f32, 'B', base_x, letter_spacing)
        //             || is_point_in_letter(x_i as f32, y_i as f32, 'M', base_x, letter_spacing)
        //         {
        //             boundary_array[(x_i as u32 + y_i as u32 * NX) as usize] = true;
        //         }
        //     }
        // }

        // -================== CIRCLE =======================
        for x_i in 0..NX {
            for y_i in 0..NY {
                let center_x = NX as f32 / 4.0; // 1/4 of width
                let center_y = NY as f32 / 2.0; // 1/2 of height
                let radius = 20.0;

                // Calculate distance from point to center
                let dx = x_i as f32 - center_x;
                let dy = y_i as f32 - center_y;
                let distance = (dx * dx + dy * dy).sqrt();

                // If point is inside or on the circle (distance <= radius), set boundary to true
                if distance <= radius {
                    boundary_array[(x_i as u32 + y_i as u32 * NX) as usize] = true;
                }
            }
        }

        // -================== BOLTZMANN =======================
        // let img = ImageReader::open("src/boltzmann.png")
        //     .expect("Failed to open image")
        //     .decode()
        //     .expect("Failed to decode image")
        //     .into_luma8();

        // let scale_down = 1_f32; // Changed to float for scaling calculations
        // let (orig_width, orig_height) = img.dimensions();
        // let img_width = (orig_width as f32 / scale_down) as u32;
        // let img_height = (orig_height as f32 / scale_down) as u32;

        // for x_i in 0..NX {
        //     for y_i in 0..NY {
        //         if x_i + img_width / 2 >= NX / 4
        //             && x_i <= img_width / 2 + NX / 4
        //             && y_i + img_height / 2 >= NY / 2
        //             && y_i <= img_height / 2 + NY / 2
        //         {
        //             // Calculate position in the centered, scaled region
        //             let px = x_i - (NX / 4 - img_width / 2);
        //             let py = y_i - (NY / 2 - img_height / 2);

        //             // Scale back up to original image coordinates
        //             let orig_x = ((px as f32) * scale_down) as u32;
        //             let scaled_y = ((py as f32) * scale_down) as u32;

        //             // Safe vertical flip with overflow check
        //             let orig_y = if scaled_y < orig_height {
        //                 orig_height - 1 - scaled_y
        //             } else {
        //                 0 // or handle this case as needed
        //             };

        //             // Check if we're within bounds of original image
        //             if orig_x < orig_width && orig_y < orig_height {
        //                 let pixel = img.get_pixel(orig_x, orig_y);
        //                 let brightness = pixel[0];
        //                 if brightness > 128 {
        //                     boundary_array[(x_i as u32 + y_i as u32 * NX) as usize] = true;
        //                 }
        //             }
        //         }
        //     }
        // }

        let packed_boundaries: Vec<u32> = boundary_array
            .chunks(32)
            .map(|chunk| {
                let mut packed = 0u32;
                for (i, &b) in chunk.iter().enumerate() {
                    if b {
                        packed |= 1 << i;
                    }
                }
                packed
            })
            .collect();

        let uniform = Params {
            nodes_x: NX,
            nodes_y: NY,
            inlet_vel: [0.1, 0.0],
            boundary_nodes: boundaries.len() as u32,
            walls: crate::boundary::pack_LDC(),
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let visualise_uniform = VisualisationUniforms {
            nodes_x: NX,
            nodes_y: NY,
            mode: 0, // 0: velocity magnitude, 1: vorticity, 2: density set mode
            min_value: 0.0,
            max_value: 0.75,
            boundary_nodes: boundary_array.len() as u32,
        };

        let visualise_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Visual Buffer"),
            contents: bytemuck::cast_slice(&[visualise_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // COMPUTTATIONAL SETUP ======================================================================= //

        // COMPUTE SHADER ----------------------------------------------------------------------------------------------------

        let utils_shader = include_str!("zou_he.wgsl");
        let compute_shader_string = include_str!("compute3.wgsl");
        let combined_shader = format!("{}\n{}", compute_shader_string, utils_shader);

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(combined_shader.into()),
        });

        let compute_buffer0 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Compute Buffer 00"),
            contents: bytemuck::cast_slice(&nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let compute_buffer1 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Compute Buffer 01"),
            contents: bytemuck::cast_slice(&nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let boundary_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Boundary Buffer"),
            contents: bytemuck::cast_slice(&packed_boundaries),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let compute_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("compute BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let compute_bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform BG"),
            layout: &compute_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: compute_buffer0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: compute_buffer1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: boundary_buffer.as_entire_binding(),
                },
            ],
        });

        // CHECK IF YOU NEED THE UNIFORM HERE OR CAN HAVE IT SEPERATELY
        let compute_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform BG"),
            layout: &compute_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: compute_buffer1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: compute_buffer0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: boundary_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&compute_bgl],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "cs_main",
        });

        // --------------------------- SHADER ---------------------------------------------------------------------  //

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("render BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render BG"),
            layout: &render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: visualise_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: compute_buffer0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: boundary_buffer.as_entire_binding(),
                },
            ],
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&render_bgl],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                // or Features::POLYGON_MODE_POINT
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = INDICES.len() as u32;

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            window,
            compute_pipeline,
            compute_bg0,
            compute_bg1,
            compute_toggle: true,

            render_bg,
            frame_rate_counter,
            count: 0,

            visualise_buffer,
            visualise_uniform,

            compute_buffer0,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn toggle_mode(&mut self) {
        // Cycle through modes: 0 -> 1 -> 2 -> 0
        self.visualise_uniform.mode = (self.visualise_uniform.mode + 1) % 3;

        // Update the buffer with new uniform values
        self.queue.write_buffer(
            &self.visualise_buffer,
            0,
            bytemuck::cast_slice(&[self.visualise_uniform]),
        );
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &WindowEvent) -> bool {
        self.window().request_redraw();
        false
    }

    fn update(&mut self) {}

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&TextureViewDescriptor {
            label: None,
            format: None,
            dimension: None,
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(
                0,
                if self.compute_toggle == true {
                    &self.compute_bg0
                } else {
                    &self.compute_bg1
                },
                &[],
            );

            let dispatch_x = (NX + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let dispatch_y = (NY + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            self.frame_rate_counter.iterations += 1;
        }

        if self.count % SUBSTEPS == 0 {
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });

                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass.set_bind_group(0, &self.render_bg, &[]);
                render_pass
                    .set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
            }
        }

        self.queue.submit(iter::once(encoder.finish()));
        if self.count % SUBSTEPS == 0 {
            output.present();
            self.count = 0;
        }

        // thread::sleep(Duration::from_millis(200));

        if self.compute_toggle {
            self.compute_toggle = false
        } else {
            self.compute_toggle = true
        }

        self.frame_rate_counter.update();

        self.count += 1;

        Ok(())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Could't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize {
            width: WINDOW_SIZE,
            height: WINDOW_SIZE,
        })
        .build(&event_loop)
        .unwrap();

    // State::new uses async code, so we're going to wait for it to finish
    let mut state = State::new(window).await;

    let _ = event_loop.run(move |event, ewlt| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window().id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                logical_key: Key::Named(NamedKey::Escape),
                                ..
                            },
                        ..
                    } => {
                        println!("{}", state.count);
                        ewlt.exit()
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                logical_key: Key::Named(NamedKey::Space),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        // Cycle through modes: 0 -> 1 -> 2 -> 0
                        state.visualise_uniform.mode = (state.visualise_uniform.mode + 1) % 3;

                        // Update the buffer with new uniform values
                        state.queue.write_buffer(
                            &state.visualise_buffer,
                            0,
                            bytemuck::cast_slice(&[state.visualise_uniform]),
                        );
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                logical_key: Key::Named(NamedKey::ArrowUp),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        pollster::block_on(State::save_simulation_state(
                            &state.device,
                            &state.queue,
                            &state.compute_buffer0,
                            NX,
                            NY,
                        ));
                    }

                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::RedrawRequested => {
                        state.update();

                        match state.render() {
                            Ok(_) => {}
                            // Reconfigure the surface if it's lost or outdated
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                state.resize(state.size)
                            }
                            // The system is out of memory, we should probably quit
                            Err(wgpu::SurfaceError::OutOfMemory) => ewlt.exit(),
                            // We're ignoring timeouts
                            Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                        }
                    }

                    _ => {}
                };
            }
        }
        _ => {}
    });
}
