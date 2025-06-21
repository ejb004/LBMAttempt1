struct Uniforms {
    nodes_x: u32,
    nodes_y: u32,
    inlet_velocity: vec2<f32>, // Inlet flow velocity
    boundary_nodes: u32,
    walls: u32,
};


@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Define storage buffer 1 (input)
@group(0) @binding(1) var<storage, read> inputBuffer: array<f32>;

// Define storage buffer 2 (output)
@group(0) @binding(2) var<storage, read_write> outputBuffer: array<f32>;

@group(0) @binding(3) var<storage, read> boundaryBuffer: array<u32>;

// wall types
const BOUNDARY_MOVING_LID = 1u;
const BOUNDARY_NO_SLIP = 2u;
const BOUNDARY_ZOUHE_INFLOW = 3u;
const BOUNDARY_ZOUHE_OUTLFOW = 4u;
// Define bit positions for each wall
const NORTH_WALL_SHIFT: u32 = 0u;
const SOUTH_WALL_SHIFT: u32 = 8u;
const EAST_WALL_SHIFT: u32 = 16u;
const WEST_WALL_SHIFT: u32 = 24u;

// Wall type getters
fn get_north_boundary() -> u32 {
    return (uniforms.walls >> NORTH_WALL_SHIFT) & 0xFFu;
}

fn get_south_boundary() -> u32 {
    return (uniforms.walls >> SOUTH_WALL_SHIFT) & 0xFFu;
}

fn get_east_boundary() -> u32 {
    return (uniforms.walls >> EAST_WALL_SHIFT) & 0xFFu;
}

fn get_west_boundary() -> u32 {
    return (uniforms.walls >> WEST_WALL_SHIFT) & 0xFFu;
}


fn get_bool(index: u32) -> bool {
    let array_index = index >> 5u; // Divide by 32 (index / 32)
    let bit_index = index & 31u; // Modulo 32 (index % 32)
    return (boundaryBuffer[array_index] & (1u << bit_index)) != 0u;
}

fn get_bool_2d(x: u32, y: u32) -> bool {
    let index = y * uniforms.nodes_x + x;
    return get_bool(index);
}

// D2Q9 velocity vectors stored as var arrays for dynamic access
var<private> c_x: array<f32, 9> = array<f32, 9>(0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0);
var<private> c_y: array<f32, 9> = array<f32, 9>(0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0);

// 6 2 5
// 3 0 1
// 7 4 8

// D2Q9 weights as var array
var<private> w: array<f32, 9> = array<f32, 9>(4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0);

// Simulation parameters
const tau = 0.55; // Relaxation time
const omega = 1.0 / tau; // Relaxation frequency

// Helper function to get flattened array index
fn getIndex(x: u32, y: u32, direction: u32) -> u32 {
    return (y * uniforms.nodes_x + x) * 9u + direction;
}

// Compute equilibrium distribution
fn computeEquilibrium(density: f32, ux: f32, uy: f32, direction: u32) -> f32 {
    let cu = c_x[direction] * ux + c_y[direction] * uy;
    let usqr = ux * ux + uy * uy;
    return w[direction] * density * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr);
}

// Modified boundary condition check
fn applyBoundaryConditions(x: u32, y: u32) -> bool {
    // Check walls (top and bottom only, leaving sides for inlet/outlet)
    if (y == 0u || x == 0u || y == uniforms.nodes_y - 1u || x == uniforms.nodes_x - 1u) {
        return true;
    }

    if (get_bool_2d(x, y)) {
        return true;
    }

    return false;
}

fn getBoundaryType(x: u32, y: u32) -> u32 {
    var wx = x;
    var wy = y;

    // Handle coordinate adjustments for moving walls independently
    if get_north_boundary() == BOUNDARY_MOVING_LID && y == uniforms.nodes_y - 2u {
        wy = uniforms.nodes_y - 1u;
    }
    if get_south_boundary() == BOUNDARY_MOVING_LID && y == 1u {
        wy = 0u;
    }
    if get_east_boundary() == BOUNDARY_MOVING_LID && x == uniforms.nodes_x - 2u {
        wx = uniforms.nodes_x - 1u;
    }
    if (get_west_boundary() == BOUNDARY_MOVING_LID || get_west_boundary() == BOUNDARY_ZOUHE_INFLOW) && x == 1u {
        wx = 0u;
    }

    // Check boundaries independently
    if (wy == uniforms.nodes_y - 1u) {
        return get_north_boundary();
    } 
    if (wy == 0u) {
        return get_south_boundary();
    } 
    if (wx == 0u) {
        return get_west_boundary();
    } 
    if (wx == uniforms.nodes_x - 1u) {
        return get_east_boundary();
    }
    
    return 0u;
}

// fn getBoundaryType(x: u32, y: u32) -> u32 {
//     if (y == uniforms.nodes_y - 2u) {
//         if get_north_boundary() == BOUNDARY_MOVING_LID {
//             return BOUNDARY_MOVING_LID ;
//         }else {
//             return BOUNDARY_NO_SLIP;
//         }
        
//     } else if (y == 0u || x == 0u || x == uniforms.nodes_x - 1u) {
//         return BOUNDARY_NO_SLIP;
//     }
//     return 0u;
// }


fn handleMovingLid(x: u32, y: u32, i: u32) -> f32 {
    var density = 0.0;
    var momentum_x = 0.0;
    var momentum_y = 0.0;

    // Calculate local density and momentum
    for (var j = 0u; j < 9u; j++) {
        let fj = inputBuffer[getIndex(x, y, j)];
        density += fj;
        momentum_x += f32(c_x[j]) * fj;
        momentum_y += f32(c_y[j]) * fj;
    }

    // Set lid velocity
    let ux = uniforms.inlet_velocity.x;
    let uy = uniforms.inlet_velocity.y;

    // Compute equilibrium with lid velocity
    return computeEquilibrium(density, ux, uy, i);
}

fn handleNoSlip(x: u32, y: u32, i: u32) -> f32 {
    // Simple bounce-back
    if (i == 0u) {
        return inputBuffer[getIndex(x, y, i)];
    }
    let opposite = (i + 4u) % 8u;
    return inputBuffer[getIndex(x, y, opposite)];
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= uniforms.nodes_x || y >= uniforms.nodes_y) {
        return;
    }

    // // Handle inlet (left boundary)
    // if (x == 1u) {
    //     let u_inlet = getInletVelocity(f32(y));
    //     // Set equilibrium distribution for inlet velocity
    //     for (var i = 0u; i < 9u; i++) {
    //         outputBuffer[getIndex(x, y, i)] = computeEquilibrium(1.0, u_inlet, 0.0, i);
    //     }
    //     return;
    // }

    // For each direction, look at where distributions would have streamed FROM
    for (var i = 0u; i < 9u; i++) {
        let src_x = i32(x) - i32(c_x[i]);
        let src_y = i32(y) - i32(c_y[i]);

        var f: f32;

        if (applyBoundaryConditions(u32(src_x), u32(src_y))) {

            // if (y == 0u || y == uniforms.nodes_y - 1u) {
            //     // Full bounce-back for sphere and walls
            //     let opposite = (i + 4u) % 8u;
            //     if (i == 0u) {
            //         f = inputBuffer[getIndex(x, y, i)];
            //     } else {
            //         f = inputBuffer[getIndex(x, y, opposite)];
            //     }
            // } else {
            //     // Handle other boundaries (should not reach here due to inlet/outlet handling)
            //     f = inputBuffer[getIndex(x, y, i)];
            // }

            // Handle different types of boundaries
            switch (getBoundaryType(x, y)) {
            case BOUNDARY_MOVING_LID: {
                f = handleMovingLid(x, y, i);
            }case BOUNDARY_NO_SLIP: {
                f = handleNoSlip(x, y, i);
            }case BOUNDARY_ZOUHE_INFLOW: {
                f = handleWestInflow(x,y,i);
            }case BOUNDARY_ZOUHE_OUTLFOW: {
                f = handleEastOutflow(x,y,i);
            }default: {
                f = inputBuffer[getIndex(x, y, i)];
            }}
        } else {
            // Regular fluid cell handling
            let src_idx = getIndex(u32(src_x), u32(src_y), i);
            f = inputBuffer[src_idx];

            var density = 0.0;
            var momentum_x = 0.0;
            var momentum_y = 0.0;

            for (var j = 0u; j < 9u; j++) {
                let fj = inputBuffer[getIndex(u32(src_x), u32(src_y), j)];
                density += fj;
                momentum_x += c_x[j] * fj;
                momentum_y += c_y[j] * fj;
            }

            let ux = momentum_x / density;
            let uy = momentum_y / density;

            let feq = computeEquilibrium(density, ux, uy, i);
            f = f - omega * (f - feq);
        }

        outputBuffer[getIndex(x, y, i)] = f;
    }
}