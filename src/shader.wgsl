// Define the uniform buffer
struct VisualisationUniforms {
    nodes_x: u32,
    nodes_y: u32,
    mode: u32,          // 0: velocity magnitude, 1: vorticity, 2: density
    min_value: f32,     // For color scaling
    max_value: f32,
    boundary_nodes: u32,
};

@group(0) @binding(0) var<uniform> uniforms: VisualisationUniforms;
// Define storage buffer 1 (input)
@group(0) @binding(1) var<storage, read> compute_buffer: array<f32>;

@group(0) @binding(2) var<storage, read> boundaryBuffer: array<u32>;


// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};


@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.uv = model.uv;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

// Fragment shader

// D2Q9 velocities for macroscopic calculations
var<private> c_x: array<f32, 9> = array<f32, 9>(
    0.0,  1.0,  0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0
);
var<private> c_y: array<f32, 9> = array<f32, 9>(
    0.0,  0.0,  1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0
);

fn get_bool(index: u32) -> bool {
    let array_index = index >> 5u;    // Divide by 32 (index / 32)
    let bit_index = index & 31u;      // Modulo 32 (index % 32)
    return (boundaryBuffer[array_index] & (1u << bit_index)) != 0u;
}

fn get_bool_2d(x: u32, y: u32) -> bool {
    let index = y * uniforms.nodes_x + x;
    return get_bool(index);
}

fn getIndex(x: u32, y: u32, direction: u32) -> u32 {
    return (y * uniforms.nodes_x + x) * 9u + direction;
}

fn getMacroscopic(x: u32, y: u32) -> vec3<f32> {
    var density = 0.0;
    var momentum_x = 0.0;
    var momentum_y = 0.0;
    
    for (var i = 0u; i < 9u; i++) {
        let f = compute_buffer[getIndex(x, y, i)];
        density += f;
        momentum_x += c_x[i] * f;
        momentum_y += c_y[i] * f;
    }
    
    return vec3<f32>(
        density,
        momentum_x / density,  // ux
        momentum_y / density   // uy
    );
}

// Calculate vorticity at a point
fn getVorticity(x: u32, y: u32) -> f32 {
    if (x == 0u || x == uniforms.nodes_x - 1u || 
        y == 0u || y == uniforms.nodes_y - 1u) {
        return 0.0;
    }
    
    // Get velocities at surrounding points
    let macro_right = getMacroscopic(x + 1u, y);
    let macro_left = getMacroscopic(x - 1u, y);
    let macro_top = getMacroscopic(x, y + 1u);
    let macro_bottom = getMacroscopic(x, y - 1u);
    
    // Central difference for velocity derivatives
    let du_dy = (macro_top.y - macro_bottom.y) / 2.0;
    let dv_dx = (macro_right.z - macro_left.z) / 2.0;
    
    return du_dy - dv_dx;  // 2D vorticity is du_dy - dv_dx
}

// Color mapping function (blue-white-red)
fn getColor(value: f32, min_val: f32, max_val: f32) -> vec3<f32> {
    let normalized = (value - min_val) / (max_val - min_val);
    let clamped = clamp(normalized, 0.0, 1.0);
    
    if (clamped < 0.5) {
        // Blue to white
        let t = clamped * 2.0;
        return vec3<f32>(t, t, 1.0);
    } else {
        // White to red
        let t = (clamped - 0.5) * 2.0;
        return vec3<f32>(1.0, 1.0 - t, 1.0 - t);
    }
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let x = u32(in.uv.x * f32(uniforms.nodes_x));
    let y = u32(in.uv.y * f32(uniforms.nodes_y));

    // Get macroscopic quantities
    let macro_v = getMacroscopic(x, y);
    var value: f32;

    var colour = getColor(0.0,0.0,1.0); 

    if(get_bool_2d(x,y)) {
        return vec4<f32>(1.0,1.0,1.0,1.0);
   }

    
    switch(uniforms.mode) {
        case 0u: {
            // Velocity magnitude
            value = sqrt(macro_v.y * macro_v.y + macro_v.z * macro_v.z);
            colour = getColor(value, uniforms.min_value, uniforms.max_value);
            
            // if value % 0.005 < 0.001 {
                
            // } else {
            //     colour = vec3<f32>(0.0,0.0,0.0);
            // }
        }
        case 1u: {
            // Vorticity
            value = (getVorticity(x, y) * 20.0) * macro_v.x;
            colour = getColor(abs(value), uniforms.min_value, uniforms.max_value);
            if value < 0.0 {
                colour.x = 0.0;
            } else {
                colour.y = 0.0;
            }
            
            colour.z = 0.0;
            
        }
        case 2u: {
    // Artistic topographic-style visualization
    value = macro_v.x;
    
    // Center around typical density
    let centered_value = value - 1.0;
    let normalized = clamp((value - 0.9) / 0.2, 0.0, 1.0);
    
    // Create organic base pattern without repetition
    let base_pattern = normalized;
    
    // Multiple contour frequencies for artistic depth
    let fine_contours = 0.01;    // Very fine lines
    let medium_contours = 0.025; // Medium lines
    let major_contours = 0.1;    // Major elevation lines
    
    // Calculate distances to multiple contour levels
    let fine_dist = abs(centered_value % fine_contours);
    let medium_dist = abs(centered_value % medium_contours);
    let major_dist = abs(centered_value % major_contours);
    
    // Create smooth, artistic contour lines
    let fine_line = exp(-fine_dist * 800.0);
    let medium_line = exp(-medium_dist * 400.0);
    let major_line = exp(-major_dist * 200.0);
    
    // Background shading with smooth gradient
    let terrain_shading = pow(base_pattern, 1.2) * 0.4 + 0.2;
    
    // Add subtle organic variation based on position
    let organic_variation = sin(f32(x) * 0.003 + f32(y) * 0.002) * 0.02;
    
    // Combine all elements
    let contour_strength = fine_line * 0.15 + medium_line * 0.25 + major_line * 0.4;
    let base_color = terrain_shading + organic_variation;
    
    // Apply contours with artistic blending
    colour = vec3<f32>(base_color * (1.0 - contour_strength) + contour_strength * 0.05);
    
    // Add subtle paper texture
    let paper_texture = fract(sin(dot(vec2<f32>(f32(x), f32(y)) * 0.013, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let grain = (paper_texture - 0.5) * 0.03;
    
    // Final composition
    colour = vec3<f32>(colour.x + grain);
    
    // Artistic color tinting
    let sepia_tint = vec3<f32>(0.9, 0.85, 0.75);
    colour = mix(colour, colour * sepia_tint, 0.3);
    
    // Vignette effect
    let center_dist = length(vec2<f32>(f32(x) - 400.0, f32(y) - 300.0)) / 500.0;
    let vignette = 1.0 - smoothstep(0.5, 1.2, center_dist) * 0.3;
    colour *= vignette;
    
    // Ensure good contrast
    colour = clamp(colour, vec3<f32>(0.05), vec3<f32>(0.8));
}

case 3u: {
    // Blueprint-style topographic visualization
    value = macro_v.x;
    
    // Center around typical density
    let centered_value = value - 1.0;
    
    // Multiple contour frequencies for technical drawing
    let fine_contours = 0.01;    // Very fine lines
    let medium_contours = 0.025; // Medium lines
    let major_contours = 0.1;    // Major elevation lines
    
    // Calculate distances to multiple contour levels
    let fine_dist = abs(centered_value % fine_contours);
    let medium_dist = abs(centered_value % medium_contours);
    let major_dist = abs(centered_value % major_contours);
    
    // Create sharp contour lines (blueprint style)
    let fine_line = 1.0 - smoothstep(0.0, 0.001, min(fine_dist, fine_contours - fine_dist));
    let medium_line = 1.0 - smoothstep(0.0, 0.0015, min(medium_dist, medium_contours - medium_dist));
    let major_line = 1.0 - smoothstep(0.0, 0.002, min(major_dist, major_contours - major_dist));
    
    // Blueprint blue background
    colour = vec3<f32>(0.0, 0.15, 0.25);
    
    // Add very fine grid overlay (much smaller cells)
    let grid_size = 8.0;  // Much smaller grid cells
    let grid_x = fract(f32(x) / grid_size);
    let grid_y = fract(f32(y) / grid_size);
    let grid_line = step(0.995, grid_x) + step(0.995, grid_y);  // Much thinner lines
    
    // Major grid lines
    let major_grid_size = 40.0;  // Smaller major grid
    let major_grid_x = fract(f32(x) / major_grid_size);
    let major_grid_y = fract(f32(y) / major_grid_size);
    let major_grid_line = step(0.99, major_grid_x) + step(0.99, major_grid_y);  // Still thin
    
    // White drawing color
    let drawing_color = vec3<f32>(0.9, 0.95, 1.0);
    
    // Apply grid with very low opacity
    colour = mix(colour, drawing_color, grid_line * 0.02);  // Even more subtle
    colour = mix(colour, drawing_color, major_grid_line * 0.04);  // More subtle
    
    // Apply contours with different weights
    colour = mix(colour, drawing_color, fine_line * 0.3);
    colour = mix(colour, drawing_color, medium_line * 0.5);
    colour = mix(colour, drawing_color, major_line * 0.7);
    
    // Add subtle title block area (darker rectangle)
    if f32(x) > 700.0 && f32(y) > 500.0 {
        colour *= 0.8;
    }
    
    // Subtle vignette for blueprint effect
    let center_dist = length(vec2<f32>(f32(x) - 400.0, f32(y) - 300.0)) / 500.0;
    let vignette = 1.0 - smoothstep(0.5, 1.2, center_dist) * 0.15;
    colour *= vignette;
    
    // Final adjustments
    colour = clamp(colour, vec3<f32>(0.0), vec3<f32>(1.0));
}



        default: {
            // Density
            value = pow(macro_v.x,2.0);
            colour = getColor(abs(value), 0.99, 1.13) / 2.0;
        }
            
    }


    
    
    return vec4<f32>(colour, 1.0);
}

// @fragment
// fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
// // Step 1: Use the UV coordinates, ensuring they are clamped within [0.0, 1.0]
//     let uv = clamp(in.uv, vec2(0.0), vec2(1.0));

//     // Step 2: Convert UV coordinates to grid indices
//     let grid_index_x: u32 = u32(uv.x * f32(uniforms.nodes_x));
//     let grid_index_y: u32 = u32(uv.y * f32(uniforms.nodes_y));

//     // Step 3: Ensure grid indices are within bounds
//     let clamped_x = min(grid_index_x, uniforms.nodes_x - 1u);
//     let clamped_y = min(grid_index_y, uniforms.nodes_y - 1u);

//     // Step 4: Convert grid indices to a flat array index
//     let cell_index: u32 = clamped_y * uniforms.nodes_x + clamped_x; // Fixed: removed * uniforms.nodes_y
//     let buffer_index: u32 = cell_index * 9u;  // Base index for the cell

//     // Debug: output the buffer index as grayscale
//     // return vec4<f32>(f32(buffer_index) / f32(100u * 100u * 9u), 0.0, 0.0, 1.0);

//     // Step 5: Sample the buffer
//     var value: f32 = compute_buffer[buffer_index];

//     for (var i: u32 = 1u; i < 9u; i++) {
//         value += abs(compute_buffer[buffer_index+i]);
//     }

//     value /= 2.0;
//     value = pow(value, 5.0);

//     // Step 6: Use the sampled value in fragment processing
//     return vec4<f32>(value, value, value, 1.0);

    
// }