struct Uniforms {
    nodes_x: u32,
    nodes_y: u32,
    inlet_velocity: vec2<f32>, // Inlet flow velocity
    boundary_nodes: u32,
    walls: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> inputBuffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputBuffer: array<f32>;
@group(0) @binding(3) var<storage, read> boundaryBuffer: array<u32>;

// Boundary types
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
    let array_index = index >> 5u;
    let bit_index = index & 31u;
    return (boundaryBuffer[array_index] & (1u << bit_index)) != 0u;
}

fn get_bool_2d(x: u32, y: u32) -> bool {
    let index = y * uniforms.nodes_x + x;
    return get_bool(index);
}

// D2Q9 velocity vectors
var<private> c_x: array<f32, 9> = array<f32, 9>(0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0);
var<private> c_y: array<f32, 9> = array<f32, 9>(0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0);

// D2Q9 weights
var<private> w: array<f32, 9> = array<f32, 9>(4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0);

// Lower viscosity for better vortex shedding
const tau: f32 = 0.52;  // Lower for higher Reynolds number
const omega: f32 = 1.0 / tau;

// Helper function to get flattened array index
fn getIndex(x: u32, y: u32, direction: u32) -> u32 {
    return (y * uniforms.nodes_x + x) * 9u + direction;
}

// Compute equilibrium distribution with higher velocity limits
fn computeEquilibrium(density: f32, ux: f32, uy: f32, direction: u32) -> f32 {
    // Clamp density to prevent negative values
    let safe_density = max(density, 0.1);
    
    // Higher velocity limits for F1 speeds
    let safe_ux = clamp(ux, -0.2, 0.2);
    let safe_uy = clamp(uy, -0.2, 0.2);
    
    let cu = c_x[direction] * safe_ux + c_y[direction] * safe_uy;
    let usqr = safe_ux * safe_ux + safe_uy * safe_uy;
    
    return w[direction] * safe_density * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr);
}

// Modified boundary condition check
fn applyBoundaryConditions(x: u32, y: u32) -> bool {
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

fn handleMovingLid(x: u32, y: u32, i: u32) -> f32 {
    var density = 0.0;

    // Calculate local density
    for (var j = 0u; j < 9u; j++) {
        let fj = inputBuffer[getIndex(x, y, j)];
        density += fj;
    }

    // Ensure positive density
    density = max(density, 1.0);

    // Set lid velocity (higher for realistic F1 flow)
    let ux = clamp(uniforms.inlet_velocity.x, -0.15, 0.15);
    let uy = clamp(uniforms.inlet_velocity.y, -0.15, 0.15);

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

fn handleWestInflow(x: u32, y: u32, i: u32) -> f32 {
    // Clean inlet velocity without perturbations
    let ux = clamp(uniforms.inlet_velocity.x, 0.0, 0.2);
    let uy = clamp(uniforms.inlet_velocity.y, -0.05, 0.05);
    
    // Calculate density from known distributions
    let f0 = inputBuffer[getIndex(x, y, 0u)];
    let f2 = inputBuffer[getIndex(x, y, 2u)];
    let f3 = inputBuffer[getIndex(x, y, 3u)];
    let f4 = inputBuffer[getIndex(x, y, 4u)];
    let f6 = inputBuffer[getIndex(x, y, 6u)];
    let f7 = inputBuffer[getIndex(x, y, 7u)];
    
    let density = max(1.0, (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux));

    // Calculate unknown distributions
    switch(i) {
        case 1u: {  // right
            return f3 + (2.0/3.0) * density * ux;
        }
        case 5u: {  // top-right
            return f7 + (1.0/6.0) * density * ux + (1.0/2.0) * density * uy;
        }
        case 8u: {  // bottom-right
            return f6 + (1.0/6.0) * density * ux - (1.0/2.0) * density * uy;
        }
        default: {
            return inputBuffer[getIndex(x, y, i)];
        }
    }
}

fn handleEastOutflow(x: u32, y: u32, i: u32) -> f32 {
    // Simple copy from interior
    if x > 1u {
        return inputBuffer[getIndex(x - 1u, y, i)];
    }
    return inputBuffer[getIndex(x, y, i)];
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= uniforms.nodes_x || y >= uniforms.nodes_y) {
        return;
    }

    // For each direction, look at where distributions would have streamed FROM
    for (var i = 0u; i < 9u; i++) {
        let src_x = i32(x) - i32(c_x[i]);
        let src_y = i32(y) - i32(c_y[i]);

        var f: f32;

        // Check bounds first
        if (src_x < 0 || src_x >= i32(uniforms.nodes_x) || 
            src_y < 0 || src_y >= i32(uniforms.nodes_y) ||
            applyBoundaryConditions(u32(src_x), u32(src_y))) {
            
            // Handle different types of boundaries
            switch (getBoundaryType(x, y)) {
                case BOUNDARY_MOVING_LID: {
                    f = handleMovingLid(x, y, i);
                }
                case BOUNDARY_NO_SLIP: {
                    f = handleNoSlip(x, y, i);
                }
                case BOUNDARY_ZOUHE_INFLOW: {
                    f = handleWestInflow(x, y, i);
                }
                case BOUNDARY_ZOUHE_OUTLFOW: {
                    f = handleEastOutflow(x, y, i);
                }
                default: {
                    f = max(inputBuffer[getIndex(x, y, i)], 0.0);
                }
            }
        } else {
            // Regular fluid cell handling
            let ux = u32(src_x);
            let uy = u32(src_y);
            
            // Stream from source
            f = inputBuffer[getIndex(ux, uy, i)];

            // Calculate macroscopic properties for BGK collision
            var density = 0.0;
            var momentum_x = 0.0;
            var momentum_y = 0.0;

            for (var j = 0u; j < 9u; j++) {
                let fj = inputBuffer[getIndex(ux, uy, j)];
                density += fj;
                momentum_x += c_x[j] * fj;
                momentum_y += c_y[j] * fj;
            }

            // Ensure positive density
            density = max(density, 0.1);
            
            let vel_x = momentum_x / density;
            let vel_y = momentum_y / density;

            // Compute equilibrium
            let feq = computeEquilibrium(density, vel_x, vel_y, i);
            
            // BGK collision with conservative omega
            f = f - omega * (f - feq);
            
            // Ensure non-negative distributions
            f = max(f, 0.0);
        }

        outputBuffer[getIndex(x, y, i)] = f;
    }
}