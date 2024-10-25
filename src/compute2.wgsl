const OMEGA = 1.0;

struct Node {
    distributions: array<f32, 9>,
};

struct Uniforms {
    nodes_x: u32,
    nodes_y: u32,
};


@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Define storage buffer 1 (input)
@group(0) @binding(1) var<storage, read> inputBuffer: array<f32>;

// Define storage buffer 2 (output)
@group(0) @binding(2) var<storage, read_write> outputBuffer: array<f32>;

fn get_c_value(index: i32) -> vec2<i32> {
    switch (index) {
        case 0: { return vec2<i32>(0, 0); }   // Center
        case 1: { return vec2<i32>(1, 0); }   // Right
        case 2: { return vec2<i32>(0, 1); }   // Up
        case 3: { return vec2<i32>(-1, 0); }  // Left
        case 4: { return vec2<i32>(0, -1); }  // Down
        case 5: { return vec2<i32>(1, 1); }   // Top-right
        case 6: { return vec2<i32>(-1, 1); }  // Top-left
        case 7: { return vec2<i32>(-1, -1); } // Bottom-left
        case 8: { return vec2<i32>(1, -1); }  // Bottom-right
        default: { return vec2<i32>(0, 0); }  // Default fallback
    }
}


fn get_index(x: i32, y: i32, width: i32) -> i32 {
    return y * width + x;
}

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

fn get_direction_f32(index: u32) -> vec2<f32> {
    switch (index) {
        case 0u: {
            return vec2<f32>(0.0, 0.0);
        }
        case 1u: {
            return vec2<f32>(1.0, 0.0);
        }
        case 2u: {
            return vec2<f32>(0.0, 1.0);
        }
        case 3u: {
            return vec2<f32>(-1.0, 0.0);
        }
        case 4u: {
            return vec2<f32>(0.0, -1.0);
        }
        case 5u: {
            return vec2<f32>(1.0, 1.0);
        }
        case 6u: {
            return vec2<f32>(-1.0, 1.0);
        }
        case 7u: {
            return vec2<f32>(-1.0, -1.0);
        }
        case 8u: {
            return vec2<f32>(1.0, -1.0);
        }
        default: {
            // This should never happen if index is always < 9
            return vec2<f32>(0.0, 0.0);
        }
    }
}

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

fn add_two_arrays(a: array<f32, 9>, b: array<f32, 9>) -> array<f32, 9> {
    return array<f32, 9>(
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4],
        a[5] + b[5],
        a[6] + b[6],
        a[7] + b[7],
        a[8] + b[8]
    );
}

@compute @workgroup_size(8, 8, 4)
fn cs_main(@builtin(global_invocation_id) grid: vec3<u32>) {
    let x = i32(grid.x);
    let y = i32(grid.y);
    let width = i32(uniforms.nodes_x);
    let index = get_index(x, y, width) * 9;

    if ( x != 0 && y != 0 && x < i32(uniforms.nodes_x - 1u) && y < i32(uniforms.nodes_y - 1u) ) {
        let f_next = stream(x,y,index);
        collide(x,y,u32(index),f_next);
    }
}


fn stream(x: i32, y: i32, index: i32) -> array<f32, 9> {
    var f_next: array<f32, 9>;
    for(var i = 0; i < 9; i++) {
        let target_node = vec2<i32>(x, y) - get_direction(u32(i)); // Note the minus sign for pull scheme
        let target_index = get_index(target_node.x, target_node.y, i32(uniforms.nodes_x)) * 9;
        if(target_index >= 0 && target_index < i32(uniforms.nodes_x * uniforms.nodes_y * 9u)) {
            f_next[i] = outputBuffer[target_index + i];
        } else {
            f_next[i] = outputBuffer[index + i]; // Bounce-back boundary condition
        }
    }
    return f_next;
}


fn collide(x: i32, y: i32, index: u32, f_next: array<f32, 9>) {
    var rho = 0.0;
    for(var i = 0u; i<9u; i++) {
        rho += outputBuffer[index + i];
    }

    var u = vec2<f32>(0.0, 0.0);
    for(var i = 0u; i<9u; i++) {
        u += outputBuffer[index + i] * vec2<f32>(f32(get_direction(i).x), f32(get_direction(i).y));
    }
    u /= rho;

    var feq = array<f32, 9>(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);

    for(var i = 0u; i<9u; i++) {

        let e_dot_u = dot(get_direction_f32(i), u);
        
        feq[i] = get_weight(i) * (rho + 3.0 * e_dot_u - 1.5 * dot(u,u) + 4.5 * pow(e_dot_u, 2.0));
    }

    let next = add_two_arrays(f_next, feq);

    set_elements_in_outputBuffer(index,next);

}

fn set_elements_in_outputBuffer(x: u32, source_array: array<f32, 9>) {
    // Assuming outputBuffer is accessible globally or passed as a parameter
    outputBuffer[x + 0u] = source_array[0];
    outputBuffer[x + 1u] = source_array[1];
    outputBuffer[x + 2u] = source_array[2];
    outputBuffer[x + 3u] = source_array[3];
    outputBuffer[x + 4u] = source_array[4];
    outputBuffer[x + 5u] = source_array[5];
    outputBuffer[x + 6u] = source_array[6];
    outputBuffer[x + 7u] = source_array[7];
    outputBuffer[x + 8u] = source_array[8];
}
