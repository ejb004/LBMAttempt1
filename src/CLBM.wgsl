struct Uniforms {
    nodes_x: u32,
    nodes_y: u32,
    inlet_velocity: vec2<f32>,
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

// Wall bit positions
const NORTH_WALL_SHIFT: u32 = 0u;
const SOUTH_WALL_SHIFT: u32 = 8u;
const EAST_WALL_SHIFT: u32 = 16u;
const WEST_WALL_SHIFT: u32 = 24u;

// Physical parameters for Navier-Stokes
const dt: f32 = 0.01;           // Time step
const dx: f32 = 1.0;            // Grid spacing
const dy: f32 = 1.0;            // Grid spacing
const nu: f32 = 0.001;          // Kinematic viscosity (much lower for high Re)
const rho0: f32 = 1.0;          // Reference density

// Finite difference stencil weights
const c1: f32 = 1.0/12.0;      // Fourth-order accurate
const c2: f32 = -8.0/12.0;
const c3: f32 = 8.0/12.0;
const c4: f32 = -1.0/12.0;

// Encoding scheme for velocity/pressure in LBM-like format
// We'll encode: [p, ux, uy, dux/dx, dux/dy, duy/dx, duy/dy, vorticity, divergence]
// This maps to the 9 LBM directions but stores different physics

// Wall type getters (same as LBM)
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

fn getIndex(x: u32, y: u32, component: u32) -> u32 {
    return (y * uniforms.nodes_x + x) * 9u + component;
}

// Extract velocity/pressure from encoded format
fn getPressure(x: u32, y: u32) -> f32 {
    return inputBuffer[getIndex(x, y, 0u)];
}

fn getVelocityX(x: u32, y: u32) -> f32 {
    return inputBuffer[getIndex(x, y, 1u)];
}

fn getVelocityY(x: u32, y: u32) -> f32 {
    return inputBuffer[getIndex(x, y, 2u)];
}

fn getVorticity(x: u32, y: u32) -> f32 {
    return inputBuffer[getIndex(x, y, 7u)];
}

// Store values in encoded format
fn setPressure(x: u32, y: u32, p: f32) {
    outputBuffer[getIndex(x, y, 0u)] = p;
}

fn setVelocityX(x: u32, y: u32, ux: f32) {
    outputBuffer[getIndex(x, y, 1u)] = ux;
}

fn setVelocityY(x: u32, y: u32, uy: f32) {
    outputBuffer[getIndex(x, y, 2u)] = uy;
}

fn setVorticity(x: u32, y: u32, omega: f32) {
    outputBuffer[getIndex(x, y, 7u)] = omega;
}

// Fourth-order accurate finite differences
fn ddx_4th(f_m2: f32, f_m1: f32, f_p1: f32, f_p2: f32) -> f32 {
    return (c1 * f_m2 + c2 * f_m1 + c3 * f_p1 + c4 * f_p2) / dx;
}

fn ddy_4th(f_m2: f32, f_m1: f32, f_p1: f32, f_p2: f32) -> f32 {
    return (c1 * f_m2 + c2 * f_m1 + c3 * f_p1 + c4 * f_p2) / dy;
}

// Second derivatives for viscous terms
fn d2dx2(f_m1: f32, f_c: f32, f_p1: f32) -> f32 {
    return (f_m1 - 2.0 * f_c + f_p1) / (dx * dx);
}

fn d2dy2(f_m1: f32, f_c: f32, f_p1: f32) -> f32 {
    return (f_m1 - 2.0 * f_c + f_p1) / (dy * dy);
}

// Safe array access with bounds checking
fn safeGetVelX(x: i32, y: i32) -> f32 {
    if (x < 0 || x >= i32(uniforms.nodes_x) || y < 0 || y >= i32(uniforms.nodes_y)) {
        return 0.0;
    }
    return getVelocityX(u32(x), u32(y));
}

fn safeGetVelY(x: i32, y: i32) -> f32 {
    if (x < 0 || x >= i32(uniforms.nodes_x) || y < 0 || y >= i32(uniforms.nodes_y)) {
        return 0.0;
    }
    return getVelocityY(u32(x), u32(y));
}

fn safeGetPressure(x: i32, y: i32) -> f32 {
    if (x < 0 || x >= i32(uniforms.nodes_x) || y < 0 || y >= i32(uniforms.nodes_y)) {
        return 1.0;
    }
    return getPressure(u32(x), u32(y));
}

// Navier-Stokes momentum equation solver
fn solveNSMomentum(x: u32, y: u32) -> vec2<f32> {
    let ix = i32(x);
    let iy = i32(y);
    
    // Current velocity and pressure
    let ux = getVelocityX(x, y);
    let uy = getVelocityY(x, y);
    let p = getPressure(x, y);
    
    // Pre-calculate positions for cleaner code
    let x_m2 = ix - 2;
    let x_m1 = ix - 1;
    let x_p1 = ix + 1;
    let x_p2 = ix + 2;
    let y_m2 = iy - 2;
    let y_m1 = iy - 1;
    let y_p1 = iy + 1;
    let y_p2 = iy + 2;
    
    // Compute velocity derivatives using 4th order finite differences
    let dux_dx = ddx_4th(
        safeGetVelX(x_m2, iy),
        safeGetVelX(x_m1, iy),
        safeGetVelX(x_p1, iy),
        safeGetVelX(x_p2, iy)
    );
    
    let dux_dy = ddy_4th(
        safeGetVelX(ix, y_m2),
        safeGetVelX(ix, y_m1),
        safeGetVelX(ix, y_p1),
        safeGetVelX(ix, y_p2)
    );
    
    let duy_dx = ddx_4th(
        safeGetVelY(x_m2, iy),
        safeGetVelY(x_m1, iy),
        safeGetVelY(x_p1, iy),
        safeGetVelY(x_p2, iy)
    );
    
    let duy_dy = ddy_4th(
        safeGetVelY(ix, y_m2),
        safeGetVelY(ix, y_m1),
        safeGetVelY(ix, y_p1),
        safeGetVelY(ix, y_p2)
    );
    
    // Pressure derivatives
    let dp_dx = ddx_4th(
        safeGetPressure(x_m2, iy),
        safeGetPressure(x_m1, iy),
        safeGetPressure(x_p1, iy),
        safeGetPressure(x_p2, iy)
    );
    
    let dp_dy = ddy_4th(
        safeGetPressure(ix, y_m2),
        safeGetPressure(ix, y_m1),
        safeGetPressure(ix, y_p1),
        safeGetPressure(ix, y_p2)
    );
    
    // Second derivatives for viscous terms
    let d2ux_dx2 = d2dx2(safeGetVelX(x_m1, iy), ux, safeGetVelX(x_p1, iy));
    let d2ux_dy2 = d2dy2(safeGetVelX(ix, y_m1), ux, safeGetVelX(ix, y_p1));
    let d2uy_dx2 = d2dx2(safeGetVelY(x_m1, iy), uy, safeGetVelY(x_p1, iy));
    let d2uy_dy2 = d2dy2(safeGetVelY(ix, y_m1), uy, safeGetVelY(ix, y_p1));
    
    // Navier-Stokes momentum equations
    // du/dt = -u·∇u - (1/ρ)∇p + ν∇²u
    
    let advection_x = ux * dux_dx + uy * dux_dy;
    let pressure_x = dp_dx / rho0;
    let viscous_x = nu * (d2ux_dx2 + d2ux_dy2);
    
    let advection_y = ux * duy_dx + uy * duy_dy;
    let pressure_y = dp_dy / rho0;
    let viscous_y = nu * (d2uy_dx2 + d2uy_dy2);
    
    // Time integration (explicit Euler)
    let new_ux = ux + dt * (-advection_x - pressure_x + viscous_x);
    let new_uy = uy + dt * (-advection_y - pressure_y + viscous_y);
    
    return vec2<f32>(new_ux, new_uy);
}

// Pressure correction step (simplified)
fn solvePressureCorrection(x: u32, y: u32) -> f32 {
    let ix = i32(x);
    let iy = i32(y);
    
    // Pre-calculate positions
    let x_m2 = ix - 2;
    let x_m1 = ix - 1;
    let x_p1 = ix + 1;
    let x_p2 = ix + 2;
    let y_m2 = iy - 2;
    let y_m1 = iy - 1;
    let y_p1 = iy + 1;
    let y_p2 = iy + 2;
    
    // Velocity divergence
    let div_u = ddx_4th(
        safeGetVelX(x_m2, iy),
        safeGetVelX(x_m1, iy),
        safeGetVelX(x_p1, iy),
        safeGetVelX(x_p2, iy)
    ) + ddy_4th(
        safeGetVelY(ix, y_m2),
        safeGetVelY(ix, y_m1),
        safeGetVelY(ix, y_p1),
        safeGetVelY(ix, y_p2)
    );
    
    // Pressure Poisson equation: ∇²p = (ρ/dt)∇·u
    let p_correction = -rho0 * div_u / dt;
    
    return getPressure(x, y) + 0.1 * p_correction;  // Under-relaxation
}

// Compute vorticity for visualization
fn computeVorticity(x: u32, y: u32) -> f32 {
    let ix = i32(x);
    let iy = i32(y);
    
    // Pre-calculate positions
    let x_m2 = ix - 2;
    let x_m1 = ix - 1;
    let x_p1 = ix + 1;
    let x_p2 = ix + 2;
    let y_m2 = iy - 2;
    let y_m1 = iy - 1;
    let y_p1 = iy + 1;
    let y_p2 = iy + 2;
    
    let duy_dx = ddx_4th(
        safeGetVelY(x_m2, iy),
        safeGetVelY(x_m1, iy),
        safeGetVelY(x_p1, iy),
        safeGetVelY(x_p2, iy)
    );
    
    let dux_dy = ddy_4th(
        safeGetVelX(ix, y_m2),
        safeGetVelX(ix, y_m1),
        safeGetVelX(ix, y_p1),
        safeGetVelX(ix, y_p2)
    );
    
    return duy_dx - dux_dy;
}

// Boundary conditions
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

// Convert from LBM initialization to NS variables
fn initializeFromLBM(x: u32, y: u32) {
    // Extract density and velocity from LBM equilibrium
    var density = 0.0;
    var momentum_x = 0.0;
    var momentum_y = 0.0;
    
    // Decode from LBM format (if first time step)
    for (var i = 0u; i < 9u; i++) {
        let f = inputBuffer[getIndex(x, y, i)];
        density += f;
        // Assume standard D2Q9 lattice vectors for decoding
        if (i == 1u) { momentum_x += f; }
        if (i == 3u) { momentum_x -= f; }
        if (i == 2u) { momentum_y += f; }
        if (i == 4u) { momentum_y -= f; }
        if (i == 5u) { momentum_x += f; momentum_y += f; }
        if (i == 6u) { momentum_x -= f; momentum_y += f; }
        if (i == 7u) { momentum_x -= f; momentum_y -= f; }
        if (i == 8u) { momentum_x += f; momentum_y -= f; }
    }
    
    // Convert to pressure and velocity
    setPressure(x, y, density);
    setVelocityX(x, y, momentum_x / max(density, 0.1));
    setVelocityY(x, y, momentum_y / max(density, 0.1));
    setVorticity(x, y, 0.0);
    
    // Fill remaining components
    outputBuffer[getIndex(x, y, 3u)] = 0.0;  // dux/dx
    outputBuffer[getIndex(x, y, 4u)] = 0.0;  // dux/dy
    outputBuffer[getIndex(x, y, 5u)] = 0.0;  // duy/dx
    outputBuffer[getIndex(x, y, 6u)] = 0.0;  // duy/dy
    outputBuffer[getIndex(x, y, 8u)] = 0.0;  // divergence
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= uniforms.nodes_x || y >= uniforms.nodes_y) {
        return;
    }

    // Check if this looks like LBM data (first time step) or NS data
    let total_density = inputBuffer[getIndex(x, y, 0u)] + inputBuffer[getIndex(x, y, 1u)] + 
                       inputBuffer[getIndex(x, y, 2u)] + inputBuffer[getIndex(x, y, 3u)];
    
    if (total_density > 0.5 && total_density < 5.0) {
        // Looks like LBM data, initialize NS variables
        initializeFromLBM(x, y);
        return;
    }

    // Handle boundaries
    if (applyBoundaryConditions(x, y)) {
        let boundary_type = getBoundaryType(x, y);
        
        switch (boundary_type) {
            case BOUNDARY_MOVING_LID: {
                setPressure(x, y, 1.0);
                setVelocityX(x, y, clamp(uniforms.inlet_velocity.x, -0.5, 0.5));
                setVelocityY(x, y, clamp(uniforms.inlet_velocity.y, -0.5, 0.5));
            }
            case BOUNDARY_NO_SLIP: {
                setPressure(x, y, 1.0);
                setVelocityX(x, y, 0.0);
                setVelocityY(x, y, 0.0);
            }
            case BOUNDARY_ZOUHE_INFLOW: {
                setPressure(x, y, 1.0);
                setVelocityX(x, y, clamp(uniforms.inlet_velocity.x, 0.0, 0.8));
                setVelocityY(x, y, clamp(uniforms.inlet_velocity.y, -0.2, 0.2));
            }
            case BOUNDARY_ZOUHE_OUTLFOW: {
                // Extrapolate from interior
                if (x > 1u) {
                    setPressure(x, y, getPressure(x - 1u, y));
                    setVelocityX(x, y, getVelocityX(x - 1u, y));
                    setVelocityY(x, y, getVelocityY(x - 1u, y));
                } else {
                    setPressure(x, y, 1.0);
                    setVelocityX(x, y, 0.0);
                    setVelocityY(x, y, 0.0);
                }
            }
            default: {
                setPressure(x, y, 1.0);
                setVelocityX(x, y, 0.0);
                setVelocityY(x, y, 0.0);
            }
        }
        
        setVorticity(x, y, 0.0);
        outputBuffer[getIndex(x, y, 3u)] = 0.0;
        outputBuffer[getIndex(x, y, 4u)] = 0.0;
        outputBuffer[getIndex(x, y, 5u)] = 0.0;
        outputBuffer[getIndex(x, y, 6u)] = 0.0;
        outputBuffer[getIndex(x, y, 8u)] = 0.0;
        
        return;
    }

    // Interior points: solve Navier-Stokes
    let new_velocity = solveNSMomentum(x, y);
    let new_pressure = solvePressureCorrection(x, y);
    let new_vorticity = computeVorticity(x, y);
    
    // Store results
    setPressure(x, y, clamp(new_pressure, 0.1, 5.0));
    setVelocityX(x, y, clamp(new_velocity.x, -1.0, 1.0));
    setVelocityY(x, y, clamp(new_velocity.y, -1.0, 1.0));
    setVorticity(x, y, clamp(new_vorticity, -10.0, 10.0));
    
    // Store derivatives for next time step
    outputBuffer[getIndex(x, y, 3u)] = 0.0;  // Placeholder
    outputBuffer[getIndex(x, y, 4u)] = 0.0;  // Placeholder
    outputBuffer[getIndex(x, y, 5u)] = 0.0;  // Placeholder
    outputBuffer[getIndex(x, y, 6u)] = 0.0;  // Placeholder
    outputBuffer[getIndex(x, y, 8u)] = 0.0;  // Placeholder
}