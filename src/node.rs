pub const WEIGHTS: [f32; 9] = [
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Node {
    distribution: [f32; 9],
}

impl Node {
    pub fn to_raw(&self) -> Node {
        Node {
            distribution: self.distribution,
        }
    }

    pub fn default() -> Node {
        Node {
            distribution: [
                4.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 36.0,
                1.0 / 36.0,
                1.0 / 36.0,
                1.0 / 36.0,
            ],
        }
    }

    pub fn with_density(phi: f32) -> Node {
        Node {
            distribution: WEIGHTS.map(|w| w * phi),
        }
    }

    pub fn with_density2(phi: f32) -> Node {
        Node {
            distribution: {
                let mut result = [0.0_f32; 9]; // Assuming WEIGHTS has 9 elements
                for (i, &w) in WEIGHTS.iter().enumerate() {
                    result[i] = if i == 1 {
                        w * phi * 2.0 // Multiply second element by 2
                    } else {
                        w * phi
                    };
                }
                result
            },
        }
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Node>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}
