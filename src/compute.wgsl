const TIMERELAX = 0.55;

struct Uniforms {
    nodes_x: u32,
    nodes_y: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Define storage buffer 1 (input)
@group(0) @binding(1) var<storage, read> inputBuffer: array<f32>;

// Define storage buffer 2 (output)
@group(0) @binding(2) var<storage, read_write> outputBuffer: array<f32>;

fn getIndex(x: u32, y: u32) -> u32 {
    return y * uniforms.nodes_x + x;
}

fn getIndexi(x: i32, y: i32, width: i32) -> i32 {
    return y * i32(uniforms.nodes_x) + x;
}

fn distSquared(x:f32, y:f32, x0:f32, y0:f32) -> f32 {
    return (pow((x0-x),2.0) + pow((y0-y),2.0));
}

@compute @workgroup_size(8, 8, 4)
fn cs_main(@builtin(global_invocation_id) grid: vec3<u32>) {
    let x = grid.x;
    let y = grid.y;
    
    // Ensure we are within the grid bounds
    if x >= uniforms.nodes_x || y >= uniforms.nodes_y {
        return;
    }

    let w = uniforms.nodes_x;

    let cell_index: u32 = getIndex(x, y);
    let buffer_index: u32 = cell_index * 9u;

    collision(buffer_index);
    streaming(buffer_index, x, y); 

}

// LATTICE BOLTZMAN METHOD FOR COMPUTING FLUID FLOW //

//  e6   e2   e5 

//  e3   e0   e1

//  e7   e4   e8


// Function to get the direction vector for D2Q9 lattice
fn get_direction(index: u32) -> vec2<i32> {
    switch (index) {
        case 0u: {
            return vec2<i32>(0, 0);
        }
        case 1u: {
            return vec2<i32>(1, 0);
        }
        case 2u: {
            return vec2<i32>(0, 1);
        }
        case 3u: {
            return vec2<i32>(-1, 0);
        }
        case 4u: {
            return vec2<i32>(0, -1);
        }
        case 5u: {
            return vec2<i32>(1, 1);
        }
        case 6u: {
            return vec2<i32>(-1, 1);
        }
        case 7u: {
            return vec2<i32>(-1, -1);
        }
        case 8u: {
            return vec2<i32>(1, -1);
        }
        default: {
            // This should never happen if index is always < 9
            return vec2<i32>(0, 0);
        }
    }
}

// Function to get the weight for D2Q9 lattice
fn get_weight(index: u32) -> f32 {
    switch (index) {
        case 0u: {
            return 4.0 / 9.0;
        }
        case 1u, 2u, 3u, 4u: {
            return 1.0 / 9.0;
        }
        case 5u, 6u, 7u, 8u: {
            return 1.0 / 36.0;
        }
        default: {
            // This should never happen if index is always < 9
            return 0.0;
        }
    }
}

fn compute_rho(buffer_index: u32) -> f32 {
    var rho = 0.0;
    for(var i = 0u; i < 9u; i++) {
        rho += inputBuffer[buffer_index + i];
    }
    return rho;
}

fn compute_uSqr(buffer_index: u32, rho: f32) -> vec2<f32>{
    var velocity = vec2<f32>(0.0, 0.0);
    for (var i: u32 = 0u; i < 9u; i++) {
        velocity += vec2<f32>(f32(get_direction(i).x), f32(get_direction(i).y)) * inputBuffer[buffer_index + i];
    }
    velocity = velocity / rho;

    return velocity;
}

fn compute_feq(i: u32, rho: f32, u: vec2<f32>) -> f32 {
    let eu = dot(vec2<f32>(f32(get_direction(i).x), f32(get_direction(i).y)), u);
    let uu = dot(u, u);
    return get_weight(i) * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uu);
}


fn collision(buffer_index: u32) {
    let rho = compute_rho(buffer_index);
    let u = compute_uSqr(buffer_index, rho);

    let omega = 1.0 / TIMERELAX;

    for(var i: u32 = 0u; i < 9u; i++) {
       outputBuffer[buffer_index + i] = inputBuffer[buffer_index + i] - omega * (inputBuffer[buffer_index + i] - compute_feq(i, rho, u));

    }
}
fn streaming(buffer_index: u32, x: u32, y: u32) {
    for (var i: u32 = 0u; i < 9u; i++) {
        let next_x = i32(i32(x) + get_direction(i).x);
        let next_y = i32(i32(y) + get_direction(i).y);

        if (next_x >= 0 && next_y >= 0 && next_x < i32(uniforms.nodes_x) && next_y < i32(uniforms.nodes_y)) {
            let target_index = getIndexi(next_x, next_y, i32(uniforms.nodes_x));
            outputBuffer[target_index * 9 + i32(i)] = inputBuffer[buffer_index + i];
        }
    }
}


// fn streaming(buffer_index: u32, x: u32, y: u32) {
//     for(var i: u32 = 0u; i < 9u; i++) {
//         let next_x = i32(f32(x) + get_direction(i).x);
//         let next_y = i32(f32(y) + get_direction(i).y);

//         if (next_x>=0 && next_y>=0 && next_x<i32(uniforms.nodes_x) && next_y<i32(uniforms.nodes_y)) {
//             outputBuffer[getIndex(u32(next_x),u32(next_y)) * 9u + i] = inputBuffer[buffer_index + i];
//         }   
//     }
// }

