fn handleWestInflow(x: u32, y: u32, i: u32) -> f32 {
    // For west (left) wall, unknown distributions are:
    // f1 (right), f5 (top-right), f8 (bottom-right)
    
    let ux = uniforms.inlet_velocity.x;  // Specified inlet velocity
    let uy = uniforms.inlet_velocity.y;
    
    // Calculate density from known distributions
    // Known: f0,f2,f3,f4,f6,f7
    let density = (1.0 / (1.0 - ux)) * (
        inputBuffer[getIndex(x, y, 0u)] +    // rest
        inputBuffer[getIndex(x, y, 2u)] +    // up
        inputBuffer[getIndex(x, y, 4u)] +    // down
        2.0 * (inputBuffer[getIndex(x, y, 3u)] +    // left
               inputBuffer[getIndex(x, y, 6u)] +    // top-left
               inputBuffer[getIndex(x, y, 7u)])     // bottom-left
    );

    // Calculate unknown distributions
    switch(i) {
        case 1u: {  // right
            return inputBuffer[getIndex(x, y, 3u)] + 
                   (2.0/3.0) * density * ux;
        }
        case 5u: {  // top-right
            return inputBuffer[getIndex(x, y, 7u)] + 
                   (1.0/6.0) * density * ux + 
                   (1.0/2.0) * density * uy;
        }
        case 8u: {  // bottom-right
            return inputBuffer[getIndex(x, y, 6u)] + 
                   (1.0/6.0) * density * ux - 
                   (1.0/2.0) * density * uy;
        }
        default: {
            return inputBuffer[getIndex(x, y, i)];
        }
    }

    // return computeEquilibrium(1.0, ux, uy, i);
}

fn handleEastOutflow(x: u32, y: u32, i: u32) -> f32 {

    let f0 = inputBuffer[getIndex(x, y, 0u)];
    let f1 = inputBuffer[getIndex(x, y, 1u)];
    let f2 = inputBuffer[getIndex(x, y, 2u)];
    let f3 = inputBuffer[getIndex(x, y, 3u)];
    let f4 = inputBuffer[getIndex(x, y, 4u)];
    let f5 = inputBuffer[getIndex(x, y, 5u)];
    let f6 = inputBuffer[getIndex(x, y, 6u)];
    let f7 = inputBuffer[getIndex(x, y, 7u)];
    let f8 = inputBuffer[getIndex(x, y, 8u)];

    let p = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
 
    let ux = (f1 + f5 + f8 - (f3 + f6 + f7)) / p;
    let uy = (f2 + f5 + f6 - (f4 + f7 + f8)) / p;

    switch(i) {
        case 3u: {
            return f1 + (2.0/3.0) * p * ux;
        }
        case 6u: {
            return f8 + (1.0/6.0) * p * ux + (1.0/2.0) * p * uy;
        }
        case 7u: {
            return f5 + (1.0/6.0) * p * ux + (1.0/2.0) * p * uy;
        }
        default: {
            return inputBuffer[getIndex(x, y, i)];
        }
    }

}
