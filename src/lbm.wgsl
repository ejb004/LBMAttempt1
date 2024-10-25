// LATTICE BOLTZMAN METHOD FOR COMPUTING FLUID FLOW //

//  e6   e2   e5 

//  e3   e0   e1

//  e7   e4   e8


const EX : array<i32, 9> = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const EY : array<i32, 9> = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);

const WEIGHTS : array<f32, 9> = array<f32, 9>(4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0);



