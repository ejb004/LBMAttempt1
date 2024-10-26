#[repr(C)]
pub struct BoundaryNode {
    index: u32,
}

impl BoundaryNode {
    pub fn new(x: u32, y: u32, nx: u32) -> BoundaryNode {
        BoundaryNode { index: y * nx + x }
    }
}
