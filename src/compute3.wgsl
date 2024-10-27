struct Uniforms {
    nodes_x: u32,
    nodes_y: u32,
    inlet_velocity: f32, // Inlet flow velocity
    boundary_nodes: u32,
};


@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Define storage buffer 1 (input)
@group(0) @binding(1) var<storage, read> inputBuffer: array<f32>;

// Define storage buffer 2 (output)
@group(0) @binding(2) var<storage, read_write> outputBuffer: array<f32>;

@group(0) @binding(3) var<storage, read> boundaryBuffer: array<u32>;

// D2Q9 velocity vectors stored as var arrays for dynamic access
var<private> c_x: array<f32, 9> = array<f32, 9>(
    0.0,  1.0,  0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0
);
var<private> c_y: array<f32, 9> = array<f32, 9>(
    0.0,  0.0,  1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0
);

// D2Q9 weights as var array
var<private> w: array<f32, 9> = array<f32, 9>(
    4.0/9.0,  1.0/9.0,  1.0/9.0,  1.0/9.0,  1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
);

// Simulation parameters
const tau = 0.55;           // Relaxation time
const omega = 1.0 / tau;   // Relaxation frequency

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
    if (y == 0u || y == uniforms.nodes_y - 1u || x == uniforms.nodes_x - 1u) {
        return true;
    }
    
    for (var i = 0u; i < uniforms.boundary_nodes; i += 1u) {
      if (uniforms.nodes_x * y + x == boundaryBuffer[i]) {
        return true;
      }
    }

    return false;
}

// fn getInletVelocity(y: f32) -> f32 {
//     let h = f32(uniforms.nodes_y);
//     let y_normalized = y / h;
//     // Parabolic profile: zero at walls, maximum at center
//     return uniforms.inlet_velocity * 4.0 * y_normalized * (1.0 - y_normalized);
// }

fn getInletVelocity(y: f32) -> f32 {
    return uniforms.inlet_velocity;
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= uniforms.nodes_x || y >= uniforms.nodes_y) {
        return;
    }

    // Handle inlet (left boundary)
    if (x == 0u) {
        let u_inlet = getInletVelocity(f32(y));
        // Set equilibrium distribution for inlet velocity
        for (var i = 0u; i < 9u; i++) {
            outputBuffer[getIndex(x, y, i)] = computeEquilibrium(1.0, u_inlet, 0.0, i);
        }
        return;
    }

    // // Handle outlet (right boundary) - zero gradient
    // if (x == uniforms.nodes_x - 1u) {
    //     for (var i = 0u; i < 9u; i++) {
    //         outputBuffer[getIndex(x, y, i)] = inputBuffer[getIndex(x - 1u, y, i)];
    //     }
    //     return;
    // }

    // For each direction, look at where distributions would have streamed FROM
    for (var i = 0u; i < 9u; i++) {
        let src_x = i32(x) - i32(c_x[i]);
        let src_y = i32(y) - i32(c_y[i]);
        
        var f: f32;
        
        if (src_x < 0 || src_x >= i32(uniforms.nodes_x) ||
            src_y < 0 || src_y >= i32(uniforms.nodes_y) ||
            applyBoundaryConditions(u32(src_x), u32(src_y))) {
            
            // Different handling for sphere boundary vs walls
            if (y == 0u || y == uniforms.nodes_y - 1u) {
                // Full bounce-back for sphere and walls
                let opposite = (i + 4u) % 8u;
                if (i == 0u) {
                    f = inputBuffer[getIndex(x, y, i)];
                } else {
                    f = inputBuffer[getIndex(x, y, opposite)];
                }
            } else {
                // Handle other boundaries (should not reach here due to inlet/outlet handling)
                f = inputBuffer[getIndex(x, y, i)];
            }
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
