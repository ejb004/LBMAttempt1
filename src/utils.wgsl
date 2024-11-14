// fn computeEquilibrium(density: f32, ux: f32, uy: f32, direction: u32) -> f32 {
//     let cu = c_x[direction] * ux + c_y[direction] * uy;
//     let usqr = ux * ux + uy * uy;
//     return w[direction] * density * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr);
// }