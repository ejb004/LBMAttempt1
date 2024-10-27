#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BoundaryNode {
    pub index: u32,
}

impl BoundaryNode {
    pub fn new(x: u32, y: u32, nx: u32) -> BoundaryNode {
        BoundaryNode { index: y * nx + x }
    }
}
