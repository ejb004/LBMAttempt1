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
const BOUNDARY_ZOUHE_OUTLFOW: u32 = 4;

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

pub fn pack_wind_tunnel() -> u32 {
    pack_boundary_conditions(
        BOUNDARY_NO_SLIP,
        BOUNDARY_NO_SLIP,
        BOUNDARY_ZOUHE_OUTLFOW,
        BOUNDARY_ZOUHE_INFLOW,
    )
}

pub fn pack_LDC() -> u32 {
    pack_boundary_conditions(
        BOUNDARY_MOVING_LID,
        BOUNDARY_NO_SLIP,
        BOUNDARY_NO_SLIP,
        BOUNDARY_NO_SLIP,
    )
}

use std::f32::consts::PI;

pub fn sd_airfoil(
    p: [f32; 2],
    center_x: f32,
    center_y: f32,
    rotation: f32,
    chord_length: f32,
) -> f32 {
    // Translate the point to the center
    let shifted_p = [p[0] - center_x, p[1] - center_y];

    // Apply rotation using a rotation matrix
    let cos_theta = rotation.cos();
    let sin_theta = rotation.sin();
    let rotated_p = [
        shifted_p[0] * cos_theta - shifted_p[1] * sin_theta,
        shifted_p[0] * sin_theta + shifted_p[1] * cos_theta,
    ];

    // Airfoil SDF logic
    let x = rotated_p[0] / chord_length; // Normalized x along the chord
    let y = rotated_p[1] / chord_length; // Normalized y along the chord

    // Define upper and lower surfaces of the airfoil
    let upper_y = 0.15 * (1.0 - x * x).sqrt(); // Example airfoil curve (parabolic)
    let lower_y = -0.05 * (1.0 - x * x).sqrt();

    if x < -1.0 || x > 1.0 {
        // Outside chord length, distance to nearest point
        let edge_distance = ((x - x.clamp(-1.0, 1.0)).hypot(y));
        edge_distance
    } else if y > upper_y {
        // Above the upper surface
        y - upper_y
    } else if y < lower_y {
        // Below the lower surface
        lower_y - y
    } else {
        // Inside the airfoil shape
        -y.abs().min(upper_y.abs() - y.abs())
    }
}

pub fn is_point_in_letter(x: f32, y: f32, letter: char, base_x: f32, letter_spacing: f32) -> bool {
    let height = 100.0; // Reduced from 150 to 100 (2/3 size)
    let thickness = 13.0; // Reduced from 20 to ~13 (2/3 size)

    // Adjust x position based on letter position
    let letter_position = match letter {
        'L' => 0,
        'B' => 1,
        'M' => 2,
        _ => return false,
    };
    let x = x - (base_x + letter_position as f32 * letter_spacing);
    let y = 128.0 - y; // Flip y-coordinates and center at y=128

    match letter {
        'L' => {
            // Vertical line
            let in_vertical = x >= 0.0 && x <= thickness && y >= -height / 2.0 && y <= height / 2.0;
            // Horizontal line
            let in_horizontal =
                x >= 0.0 && x <= height / 2.0 && y >= height / 2.0 - thickness && y <= height / 2.0;
            in_vertical || in_horizontal
        }
        'B' => {
            // Vertical line
            let in_vertical = x >= 0.0 && x <= thickness && y >= -height / 2.0 && y <= height / 2.0;
            // Upper loop
            let upper_center_y = height / 4.0;
            let in_upper = {
                let dx = x - thickness;
                let dy = y - upper_center_y;
                dx * dx + dy * dy <= (height / 4.0) * (height / 4.0)
                    && x >= 0.0
                    && x <= thickness + height / 4.0
                    && dx * dx + dy * dy >= 100.0
            };
            // Lower loop
            let lower_center_y = -height / 4.0;
            let in_lower = {
                let dx = x - thickness;
                let dy = y - lower_center_y;
                dx * dx + dy * dy <= (height / 4.0) * (height / 4.0)
                    && x >= 0.0
                    && x <= thickness + height / 4.0
                    && dx * dx + dy * dy >= 100.0
            };
            in_vertical || in_upper || in_lower
        }
        'M' => {
            // Left vertical
            let in_left = x >= 0.0 && x <= thickness && y >= -height / 2.0 && y <= height / 2.0;
            // Right vertical
            let in_right = x >= height / 2.0 - thickness
                && x <= height / 2.0
                && y >= -height / 2.0
                && y <= height / 2.0;
            // Middle vertical
            let in_middle = x >= (height / 4.0 - thickness / 2.0)
                && x <= (height / 4.0 + thickness / 2.0)
                && y >= -height / 2.0
                && y <= height / 2.0;
            // Horizontal connector at bottom
            let in_horizontal = x >= 0.0
                && x <= height / 2.0
                && y >= -height / 2.0
                && y <= -height / 2.0 + thickness;

            in_left || in_right || in_middle || in_horizontal || in_left || in_right || in_middle
        }
        _ => false,
    }
}

pub fn create_multiple_circles(boundary_array: &mut [bool], nx: u32, ny: u32) {
    for x_i in 0..nx {
        for y_i in 0..ny {
            let x = x_i as f32;
            let y = y_i as f32;

            // Large central circle
            let center1_x = nx as f32 / 2.0;
            let center1_y = ny as f32 / 2.0;
            let radius1 = 25.0;
            let dist1 = ((x - center1_x).powi(2) + (y - center1_y).powi(2)).sqrt();

            // Four smaller circles around it
            let radius2 = 8.0;
            let offset = 40.0;

            let centers = [
                (center1_x - offset, center1_y), // Left
                (center1_x + offset, center1_y), // Right
                (center1_x, center1_y - offset), // Top
                (center1_x, center1_y + offset), // Bottom
            ];

            let mut is_boundary = dist1 <= radius1;

            for (cx, cy) in centers.iter() {
                let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                if dist <= radius2 {
                    is_boundary = true;
                }
            }

            if is_boundary {
                boundary_array[(x_i + y_i * nx) as usize] = true;
            }
        }
    }
}

pub fn create_venturi_nozzle(boundary_array: &mut [bool], nx: u32, ny: u32) {
    for x_i in 0..nx {
        for y_i in 0..ny {
            let x = x_i as f32;
            let y = y_i as f32;

            let inlet_width = ny as f32 * 0.8;
            let throat_width = ny as f32 * 0.2;
            let outlet_width = ny as f32 * 0.6;

            let inlet_length = nx as f32 * 0.2;
            let converging_length = nx as f32 * 0.2;
            let throat_length = nx as f32 * 0.2;
            let diverging_length = nx as f32 * 0.3;

            let center_y = ny as f32 / 2.0;

            let mut channel_half_width = 0.0;

            if x < inlet_length {
                // Inlet section
                channel_half_width = inlet_width / 2.0;
            } else if x < inlet_length + converging_length {
                // Converging section
                let t = (x - inlet_length) / converging_length;
                channel_half_width = inlet_width / 2.0 * (1.0 - t) + throat_width / 2.0 * t;
            } else if x < inlet_length + converging_length + throat_length {
                // Throat section
                channel_half_width = throat_width / 2.0;
            } else {
                // Diverging section
                let t = (x - inlet_length - converging_length - throat_length) / diverging_length;
                channel_half_width = throat_width / 2.0 * (1.0 - t) + outlet_width / 2.0 * t;
            }

            let distance_from_center = (y - center_y).abs();

            if distance_from_center >= channel_half_width {
                boundary_array[(x_i + y_i * nx) as usize] = true;
            }
        }
    }
}

pub fn create_wavy_channel(boundary_array: &mut [bool], nx: u32, ny: u32) {
    for x_i in 0..nx {
        for y_i in 0..ny {
            let x = x_i as f32;
            let y = y_i as f32;

            let amplitude = ny as f32 * 0.15;
            let frequency = 4.0 * std::f32::consts::PI / nx as f32;
            let base_width = ny as f32 * 0.3;

            let center_y = ny as f32 / 2.0;
            let wave_offset = amplitude * (frequency * x).sin();

            let top_boundary = center_y + base_width / 2.0 + wave_offset;
            let bottom_boundary = center_y - base_width / 2.0 + wave_offset;

            if y < bottom_boundary || y > top_boundary {
                boundary_array[(x_i + y_i * nx) as usize] = true;
            }
        }
    }
}

pub fn create_heat_exchanger(boundary_array: &mut [bool], nx: u32, ny: u32) {
    for x_i in 0..nx {
        for y_i in 0..ny {
            let x = x_i as f32;
            let y = y_i as f32;

            let fin_spacing = 30.0;
            let fin_thickness = 4.0;
            let fin_height = ny as f32 * 0.7;
            let base_height = ny as f32 * 0.1;

            // Base plate
            if y < base_height || y > ny as f32 - base_height {
                boundary_array[(x_i + y_i * nx) as usize] = true;
                continue;
            }

            // Vertical fins
            let fin_position = x % fin_spacing;
            if fin_position < fin_thickness {
                let center_y = ny as f32 / 2.0;
                let fin_start = center_y - fin_height / 2.0;
                let fin_end = center_y + fin_height / 2.0;

                if y >= fin_start && y <= fin_end {
                    boundary_array[(x_i + y_i * nx) as usize] = true;
                }
            }
        }
    }
}

pub fn create_maze_pattern(boundary_array: &mut [bool], nx: u32, ny: u32) {
    for x_i in 0..nx {
        for y_i in 0..ny {
            let x = x_i as f32;
            let y = y_i as f32;

            let wall_thickness = 3.0;
            let cell_size = 20.0;

            // Create grid pattern
            let grid_x = (x % cell_size) < wall_thickness;
            let grid_y = (y % cell_size) < wall_thickness;

            // Add some gaps to make it navigable
            let gap_size = 8.0;
            let cell_center_x = (x / cell_size).floor() * cell_size + cell_size / 2.0;
            let cell_center_y = (y / cell_size).floor() * cell_size + cell_size / 2.0;

            let near_center_x = (x - cell_center_x).abs() < gap_size;
            let near_center_y = (y - cell_center_y).abs() < gap_size;

            // Create gaps in alternating pattern
            let cell_x_idx = (x / cell_size).floor() as u32;
            let cell_y_idx = (y / cell_size).floor() as u32;
            let has_x_gap = (cell_x_idx + cell_y_idx) % 2 == 0;
            let has_y_gap = (cell_x_idx + cell_y_idx) % 2 == 1;

            let is_wall = (grid_x && !(has_x_gap && near_center_y))
                || (grid_y && !(has_y_gap && near_center_x));

            if is_wall {
                boundary_array[(x_i + y_i * nx) as usize] = true;
            }
        }
    }
}

pub fn create_building_cluster(boundary_array: &mut [bool], nx: u32, ny: u32) {
    let buildings = [
        // (x_start, y_start, width, height)
        (
            nx as f32 * 0.2,
            ny as f32 * 0.1,
            nx as f32 * 0.08,
            ny as f32 * 0.25,
        ),
        (
            nx as f32 * 0.35,
            ny as f32 * 0.15,
            nx as f32 * 0.06,
            ny as f32 * 0.35,
        ),
        (
            nx as f32 * 0.5,
            ny as f32 * 0.1,
            nx as f32 * 0.1,
            ny as f32 * 0.2,
        ),
        (
            nx as f32 * 0.65,
            ny as f32 * 0.2,
            nx as f32 * 0.07,
            ny as f32 * 0.3,
        ),
        (
            nx as f32 * 0.3,
            ny as f32 * 0.6,
            nx as f32 * 0.09,
            ny as f32 * 0.15,
        ),
        (
            nx as f32 * 0.45,
            ny as f32 * 0.65,
            nx as f32 * 0.08,
            ny as f32 * 0.25,
        ),
    ];

    for x_i in 0..nx {
        for y_i in 0..ny {
            let x = x_i as f32;
            let y = y_i as f32;

            for &(bx, by, bw, bh) in buildings.iter() {
                if x >= bx && x <= bx + bw && y >= by && y <= by + bh {
                    boundary_array[(x_i + y_i * nx) as usize] = true;
                    break;
                }
            }
        }
    }
}

pub fn create_square_cylinder_splitter(boundary_array: &mut [bool], nx: u32, ny: u32) {
    for x_i in 0..nx {
        for y_i in 0..ny {
            let x = x_i as f32;
            let y = y_i as f32;

            let cylinder_size = ny as f32 * 0.1;
            let center_x = nx as f32 * 0.3;
            let center_y = ny as f32 * 0.5;

            // Square cylinder
            if (x - center_x).abs() <= cylinder_size / 2.0
                && (y - center_y).abs() <= cylinder_size / 2.0
            {
                boundary_array[(x_i + y_i * nx) as usize] = true;
            }

            // Splitter plate extending downstream
            let plate_length = nx as f32 * 0.2;
            let plate_thickness = 2.0;

            if x >= center_x + cylinder_size / 2.0
                && x <= center_x + cylinder_size / 2.0 + plate_length
                && (y - center_y).abs() <= plate_thickness / 2.0
            {
                boundary_array[(x_i + y_i * nx) as usize] = true;
            }
        }
    }
}

pub fn create_ahmed_body(boundary_array: &mut [bool], nx: u32, ny: u32) {
    for x_i in 0..nx {
        for y_i in 0..ny {
            let x = x_i as f32;
            let y = y_i as f32;

            let body_length = nx as f32 * 0.4;
            let body_height = ny as f32 * 0.2;
            let body_width = ny as f32 * 0.15;

            let start_x = nx as f32 * 0.2;
            let base_y = ny as f32 * 0.2;

            let center_y = ny as f32 * 0.5;

            // Main body
            if x >= start_x
                && x <= start_x + body_length * 0.7
                && y >= base_y
                && y <= base_y + body_height
                && (y - center_y).abs() <= body_width
            {
                boundary_array[(x_i + y_i * nx) as usize] = true;
            }

            // Slanted rear section (25° slant)
            let slant_start = start_x + body_length * 0.7;
            let slant_length = body_length * 0.3;
            let slant_angle = 25.0_f32.to_radians();

            if x >= slant_start && x <= start_x + body_length && (y - center_y).abs() <= body_width
            {
                let x_rel = x - slant_start;
                let slant_height = body_height - x_rel * slant_angle.tan();

                if y >= base_y && y <= base_y + slant_height {
                    boundary_array[(x_i + y_i * nx) as usize] = true;
                }
            }
        }
    }
}
