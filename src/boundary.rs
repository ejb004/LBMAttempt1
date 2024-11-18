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

// Rust code for packing boundary conditions
const BOUNDARY_MOVING_LID: u32 = 1;
const BOUNDARY_NO_SLIP: u32 = 2;
const BOUNDARY_ZOUHE_INFLOW: u32 = 3;

// Define bit positions for each wall
const NORTH_WALL_SHIFT: u32 = 0;
const SOUTH_WALL_SHIFT: u32 = 8;
const EAST_WALL_SHIFT: u32 = 16;
const WEST_WALL_SHIFT: u32 = 24;

fn pack_boundary_conditions(north: u32, south: u32, east: u32, west: u32) -> u32 {
    // Mask each value to ensure only using 8 bits
    let north_masked = (north & 0xFF) << NORTH_WALL_SHIFT;
    let south_masked = (south & 0xFF) << SOUTH_WALL_SHIFT;
    let east_masked = (east & 0xFF) << EAST_WALL_SHIFT;
    let west_masked = (west & 0xFF) << WEST_WALL_SHIFT;

    // Combine all walls into single u32
    north_masked | south_masked | east_masked | west_masked
}

pub fn pack_all_no_slip() -> u32 {
    pack_boundary_conditions(
        BOUNDARY_NO_SLIP,
        BOUNDARY_NO_SLIP,
        BOUNDARY_NO_SLIP,
        BOUNDARY_NO_SLIP,
    )
}

pub fn pack_LDC() -> u32 {
    pack_boundary_conditions(
        BOUNDARY_NO_SLIP,
        BOUNDARY_NO_SLIP,
        BOUNDARY_NO_SLIP,
        BOUNDARY_ZOUHE_INFLOW,
    )
}
