fn handleWestInflow(x: u32, y: u32, i: u32) -> f32 {
    // For west (left) wall, unknown distributions are:
    // f1 (right), f5 (top-right), f8 (bottom-right)
    
    let ux = uniforms.inlet_velocity.x;  // Specified inlet velocity
    let uy = uniforms.inlet_velocity.y;
    
    // // Calculate density from known distributions
    // // Known: f0,f2,f3,f4,f6,f7
    // let density = (1.0 / (1.0 - ux)) * (
    //     inputBuffer[getIndex(x, y, 0u)] +    // rest
    //     inputBuffer[getIndex(x, y, 2u)] +    // up
    //     inputBuffer[getIndex(x, y, 3u)] +    // left
    //     inputBuffer[getIndex(x, y, 4u)] +    // down
    //     inputBuffer[getIndex(x, y, 6u)] +    // top-left
    //     inputBuffer[getIndex(x, y, 7u)] +    // bottom-left
    //     2.0 * (inputBuffer[getIndex(x, y, 3u)] +    // left
    //            inputBuffer[getIndex(x, y, 6u)] +    // top-left
    //            inputBuffer[getIndex(x, y, 7u)])     // bottom-left
    // );

    // // Calculate unknown distributions
    // switch(i) {
    //     case 1u: {  // right
    //         return inputBuffer[getIndex(x, y, 3u)] + 
    //                (2.0/3.0) * density * ux;
    //     }
    //     case 5u: {  // top-right
    //         return inputBuffer[getIndex(x, y, 7u)] + 
    //                (1.0/6.0) * density * ux + 
    //                (1.0/2.0) * density * uy -
    //                (1.0/2.0) * (inputBuffer[getIndex(x, y, 2u)] - 
    //                            inputBuffer[getIndex(x, y, 4u)]);
    //     }
    //     case 8u: {  // bottom-right
    //         return inputBuffer[getIndex(x, y, 6u)] + 
    //                (1.0/6.0) * density * ux - 
    //                (1.0/2.0) * density * uy +
    //                (1.0/2.0) * (inputBuffer[getIndex(x, y, 2u)] - 
    //                            inputBuffer[getIndex(x, y, 4u)]);
    //     }
    //     default: {
    //         return inputBuffer[getIndex(x, y, i)];
    //     }
    // }

    return computeEquilibrium(1.0, ux, uy, i);
}