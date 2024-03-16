extern "C" {
#include "LBM.cuh"
}
#include "BenchmarkSummary.cuh"

void d2q9_initial_value_d_caller(Grid grid, PRECISION *subgrid, int subgridX, int subgridY, int d)
{
    recordStart("initialize");
    d2q9_initial_value_d<<<1024, 256>>>(grid, subgrid, subgridX, subgridY, d);
    recordEnd("initialize");
}

void d2q9_save_reduce_caller(Grid grid, PRECISION *base_subgrid, PRECISION *reduced_subgrid, int subgridX, int subgridY, int d)
{
    recordStart("save_reduce");
    d2q9_save_reduce<<<1024, 256>>>(grid, base_subgrid, reduced_subgrid, subgridX, subgridY, d);
    recordEnd("save_reduce");
}

void d2q9_read_horizontal_slices_caller(Grid grid, PRECISION **subgrid_d, PRECISION *interface_left, PRECISION *interface_right, int subgridX, int subgridY)
{
    // wrap the array
    SubgridArray subgrid_D_wrapped;
    for(int d=0;d<grid.directionsNumber;d++)
    {
        subgrid_D_wrapped.subgrid[d] = subgrid_d[d];
    }

    int line_num = grid.subgridTrueSize[1] * grid.conservativesNumber * grid.directionsNumber * 2;
    int thread_num = 128;
    int block_num = line_num;
    while(block_num > 1500) // Fine-tuned, it appears that the best block number is around 1000
    {
        block_num /= 2;
    }

    recordStart("read_horizontal_slices");
    d2q9_read_horizontal_slices<<<block_num, thread_num>>>(grid, subgrid_D_wrapped, interface_left, interface_right, subgridX, subgridY);
    recordEnd("read_horizontal_slices");
}

void d2q9_write_horizontal_slices_caller(Grid grid, PRECISION **subgrid_d, PRECISION *interface_left, PRECISION *interface_right, int subgridX, int subgridY)
{
    // wrap the array
    SubgridArray subgrid_D_wrapped;
    for(int d=0;d<grid.directionsNumber;d++)
    {
        subgrid_D_wrapped.subgrid[d] = subgrid_d[d];
    }

    int line_num = grid.subgridTrueSize[1] * grid.conservativesNumber * grid.directionsNumber * 2;
    int thread_num = 128;
    int block_num = line_num;
    while(block_num > 1500) // Fine-tuned, it appears that the best block number is around 1000
    {
        block_num /= 2;
    }

    recordStart("write_horizontal_slices");
    d2q9_write_horizontal_slices<<<block_num, thread_num>>>(grid, subgrid_D_wrapped, interface_left, interface_right, subgridX, subgridY);
    recordEnd("write_horizontal_slices");
}

void d2q9_read_vertical_slices_caller(Grid grid, PRECISION **subgrid_d, PRECISION *interface_down, PRECISION *interface_up, int subgridX, int subgridY)
{
    // wrap the array
    SubgridArray subgrid_D_wrapped;
    for(int d=0;d<grid.directionsNumber;d++)
    {
        subgrid_D_wrapped.subgrid[d] = subgrid_d[d];
    }

    recordStart("read_vertical_slices");
    d2q9_read_vertical_slices<<<READ_VERTICAL_SLICES_BLOCK_NUM, READ_VERTICAL_SLICES_THREAD_NUM>>>(grid, subgrid_D_wrapped, interface_down, interface_up, subgridX, subgridY);
    recordEnd("read_vertical_slices");
}

void d2q9_LBM_step_caller(Grid grid,
                PRECISION **subgrid_FROM_D,
                PRECISION **subgrid_TO_D,
                int horizontal_uncomputed_number, int vertical_uncomputed_number,
                bool has_from_interface_horizontal,
                bool has_from_interface_vertical,
                bool has_to_interface_horizontal,
                bool has_to_interface_vertical,
                PRECISION *interface_down, PRECISION *interface_up,
                int subgridX, int subgridY)
{
    // wrap the arrays
    SubgridArray subgrid_FROM_D_wrapped;
    SubgridArray subgrid_TO_D_wrapped;
    for(int d=0;d<grid.directionsNumber;d++)
    {
        subgrid_FROM_D_wrapped.subgrid[d] = subgrid_FROM_D[d];
        subgrid_TO_D_wrapped.subgrid[d] = subgrid_TO_D[d];
    }

    // printf("d2q9_LBM_step_caller: has_from_interface_horizontal=%d, has_from_interface_vertical=%d, has_to_interface_horizontal=%d, has_to_interface_vertical=%d\n",
    //     has_from_interface_horizontal, has_from_interface_vertical, has_to_interface_horizontal, has_to_interface_vertical);

    int cellNum = (grid.subgridTrueSize[0]-2*horizontal_uncomputed_number) * (grid.subgridTrueSize[1]-2*vertical_uncomputed_number);
    int thread_num = 128; // Fine-tuned
    int block_num = (cellNum + thread_num - 1) / thread_num;
    while(block_num > 16384)
    {
        block_num /= 2;
    }
    
    recordStart("LBM_step");
    d2q9_LBM_step<<<//grid.subgridTrueSize[1]-vertical_uncomputed_number*2
                block_num, thread_num>>>(grid,
                subgrid_FROM_D_wrapped,
                subgrid_TO_D_wrapped,
                horizontal_uncomputed_number, vertical_uncomputed_number,
                has_from_interface_horizontal,
                has_from_interface_vertical,
                has_to_interface_horizontal,
                has_to_interface_vertical,
                interface_down, interface_up,
                subgridX, subgridY);
    recordEnd("LBM_step");
    gpuErrchk(cudaPeekAtLastError());

}



__host__ __device__
constexpr int get_dir(int i, int j) // constexpr to be sure it is never called
{
    // const int dirs[9][2] = {
    //     {-1, -1},
    //     {0, -1},
    //     {1, -1},
    //     {-1, 0},
    //     {0, 0},
    //     {1, 0},
    //     {-1, 1},
    //     {0, 1},
    //     {1, 1}
    // };

    // return dirs[i][j];
    return j?i/3-1:i%3-1;
}

__host__ __device__
void fluid_to_kin(const PRECISION *w, PRECISION *f)
{
    static const PRECISION c2 = 1. / 3.;
    PRECISION dotvel = 0, vel2 = 0, l2 = 0, l4 = 0, c4 = 0;

    l2 = MAXIMUM_VELOCITY * MAXIMUM_VELOCITY;
    PRECISION l2_ov_c2 = l2 / c2;

    l4 = l2 * l2;
    c4 = c2 * c2;

    vel2 = (w[1] * w[1] + w[2] * w[2]) / (w[0] * w[0]);
    dotvel = sqrt(l2) * ((PRECISION)get_dir(4,0) * w[1] + (PRECISION)get_dir(4,1) * w[2]) / w[0];

    f[4] = (4. / 9.) * w[0] *
    (1.0 + (l2_ov_c2)*dotvel + l4 / (2. * c4) * dotvel * dotvel -
    l2 / (2. * c2) * vel2);

    // perpendicular directions
    #pragma unroll
    for (size_t i = 1; i < 9; i+=2) {
        dotvel = sqrt(l2) * ((PRECISION)get_dir(i,0) * w[1] + (PRECISION)get_dir(i,1) * w[2]) / w[0];
        f[i] = (1. / 9.) * w[0] *
        (1.0 + (l2_ov_c2)*dotvel + l4 / (2. * c4) * dotvel * dotvel -
        l2 / (2. * c2) * vel2);
    }
    // diagonal directions
    #pragma unroll
    for (size_t it = 0; it < 4; it++) {
        size_t i = it * 2 + 2*(it>1);
        dotvel = sqrt(l2) * ((PRECISION)get_dir(i,0) * w[1] + (PRECISION)get_dir(i,1) * w[2]) / w[0];
        f[i] = (1. / 36.) * w[0] *
        (1.0 + (l2_ov_c2)*dotvel + l4 / (2. * c4) * dotvel * dotvel -
        l2 / (2. * c2) * vel2);
    }
}

__host__ __device__
void kin_to_fluid(const PRECISION *f, PRECISION *w)
{
    w[0] = 0;
    w[1] = 0;
    w[2] = 0;

    #pragma unroll
    for (int i = 0; i < 9; i++) {
        w[0] = w[0] + f[i];
        w[1] = w[1] + MAXIMUM_VELOCITY * (PRECISION)get_dir(i,0) * f[i];
        w[2] = w[2] + MAXIMUM_VELOCITY * (PRECISION)get_dir(i,1) * f[i];
    }
}

__device__
bool is_in_cylinder(PRECISION x, PRECISION y)
{
    return
        (x - CYLINDER_CENTER_X) * (x - CYLINDER_CENTER_X) + (y - CYLINDER_CENTER_Y) * (y - CYLINDER_CENTER_Y) < CYLINDER_RADIUS * CYLINDER_RADIUS;
        //|| x < -0.9 || x > 0.9 || y < -0.9 || y > 0.9;
}

__device__
void d2q9_t0(PRECISION w[3], PRECISION x, PRECISION y)
{
    PRECISION rho, u, v;
    rho = 1;
    u = 0.03;
    v = 0.00001; // to get instability
    // u = 0.03 + 0.01*sin(x+2*y) + 0.05*sin(cos(3*x + y*y) + x*x*y); // TODO ERASE
    // v = 0.00001 + 0.01*sin(x*7+2*y)+ 0.05*sin(cos(7*y + y*x) + y*x*y); // TODO ERASE

    if (is_in_cylinder(x, y)) {
        u = 0;
    }

    w[0] = rho;
    w[1] = rho * u;
    w[2] = rho * v;

    // w[0] = rho;
    // w[1] = rho * u;
    // w[2] = rho * v;
}


// Initialize all the values including the ghost cells
__global__
void d2q9_initial_value_d(Grid grid, PRECISION *subgrid, int subgridX, int subgridY, int d)
{
    (void)d; // same for all directions

    int stride = blockDim.x * gridDim.x;

	int cellNum = grid.subgridTrueSize[0] * grid.subgridTrueSize[1];

	for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < cellNum; id += stride)
	{
        int true_x = id % grid.subgridTrueSize[0];
        int true_y = id / grid.subgridTrueSize[0];

        int logical_x = true_x - grid.overlapSize[0];
        int logical_y = true_y - grid.overlapSize[1];

        int xgInt = logical_x + subgridX * grid.subgridOwnedSize[0];
        int ygInt = logical_y + subgridY * grid.subgridOwnedSize[1];

        PRECISION xg = grid.physicalMinCoords[0] + (((PRECISION)xgInt + 0.5) * grid.physicalSize[0] / (PRECISION)grid.size[0]);
        PRECISION yg = grid.physicalMinCoords[1] + (((PRECISION)ygInt + 0.5) * grid.physicalSize[1] / (PRECISION)grid.size[1]);

        PRECISION w[3];
        d2q9_t0(w, xg, yg);
        PRECISION f[3][3];
        fluid_to_kin(w, &f[0][0]);

        for(int c=0;c<grid.conservativesNumber;++c)
        {
            int subgrid_id = c * cellNum + id;
            subgrid[subgrid_id] = f[d][c];
        }
    }
}

// kernel used for the reduction for the save
// agregates the values in the stencil by doing an average
__global__
void d2q9_save_reduce(Grid grid, PRECISION *base_subgrid, PRECISION *reduced_subgrid, int subgridX, int subgridY, int d)
{
    int stride = blockDim.x * gridDim.x;

    int stencil_num_x_in_subgrid = grid.subgridOwnedSize[0] / grid.saveStencilSize[0];
    int stencil_num_y_in_subgrid = grid.subgridOwnedSize[1] / grid.saveStencilSize[1];

    assert(stencil_num_x_in_subgrid * grid.saveStencilSize[0] == grid.subgridOwnedSize[0]);
    assert(stencil_num_y_in_subgrid * grid.saveStencilSize[1] == grid.subgridOwnedSize[1]);

    int cellNum = stencil_num_x_in_subgrid * stencil_num_y_in_subgrid * grid.conservativesNumber;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < cellNum; id += stride)
    {
        int local_stencil_x = id % stencil_num_x_in_subgrid;
        int local_stencil_y = (id / stencil_num_x_in_subgrid) % stencil_num_y_in_subgrid;
        int c = id / (stencil_num_x_in_subgrid * stencil_num_y_in_subgrid);

        int global_stencil_x = local_stencil_x + subgridX * stencil_num_x_in_subgrid;
        int global_stencil_y = local_stencil_y + subgridY * stencil_num_y_in_subgrid;

        PRECISION average = 0;

        for (int i = 0; i < grid.saveStencilSize[0]; i++) {
            for (int j = 0; j < grid.saveStencilSize[1]; j++) {
                int true_x = local_stencil_x * grid.saveStencilSize[0] + i + grid.overlapSize[0];
                int true_y = local_stencil_y * grid.saveStencilSize[1] + j + grid.overlapSize[1];

                int base_subgrid_id = c * grid.subgridTrueSize[0] * grid.subgridTrueSize[1] + true_y * grid.subgridTrueSize[0] + true_x;

                average += base_subgrid[base_subgrid_id];
            }
        }

        average /= grid.saveStencilSize[0] * grid.saveStencilSize[1];

        int reduced_subgrid_id =
                        d * grid.sizeOfSavedData[0] * grid.sizeOfSavedData[1] * grid.conservativesNumber
                        + c * grid.sizeOfSavedData[0] * grid.sizeOfSavedData[1]
                        + global_stencil_y * grid.sizeOfSavedData[0]
                        + global_stencil_x;

        reduced_subgrid[reduced_subgrid_id] = average;
    }
}


// Read the subgrid and write to the horizontal slices
__global__ void d2q9_read_horizontal_slices(Grid grid, SubgridArray subgridWrapped, PRECISION *interface_left, PRECISION *interface_right, int subgridX, int subgridY)
{
    int stride = blockDim.x * gridDim.x;

    int cellNum = grid.overlapSize[0] * grid.subgridTrueSize[1] * grid.conservativesNumber * grid.directionsNumber * 2;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < cellNum; id += stride)
    {
        int true_x = id % grid.overlapSize[0];
        int true_y = (id / grid.overlapSize[0]) % grid.subgridTrueSize[1];
        int c = (id / (grid.overlapSize[0] * grid.subgridTrueSize[1])) % grid.conservativesNumber;
        int d = (id / (grid.overlapSize[0] * grid.subgridTrueSize[1] * grid.conservativesNumber)) % grid.directionsNumber;
        int side = id / (grid.overlapSize[0] * grid.subgridTrueSize[1] * grid.conservativesNumber * grid.directionsNumber); // 0 for left, 1 for right

        true_x += side * (grid.subgridLogicalSize[0]); // Displace to the right if we target the right side
        true_x += (1-side) * grid.overlapSize[0]; //

        if(side == 0)
        {
            int subgrid_id = 
                    c * grid.subgridTrueSize[0] * grid.subgridTrueSize[1]
                    + true_y * grid.subgridTrueSize[0]
                    + true_x;

            interface_left[id] = subgridWrapped.subgrid[d][subgrid_id];
        }
        else
        {
            int subgrid_id = 
                    c * grid.subgridTrueSize[0] * grid.subgridTrueSize[1]
                    + true_y * grid.subgridTrueSize[0]
                    + true_x;

            interface_right[id - grid.overlapSize[0] * grid.subgridTrueSize[1] * grid.conservativesNumber * grid.directionsNumber] = subgridWrapped.subgrid[d][subgrid_id];
        }
    }
}

// Write to the subgrid from the horizontal slices
__global__ void d2q9_write_horizontal_slices(Grid grid, SubgridArray subgridWrapped, PRECISION *interface_left, PRECISION *interface_right, int subgridX, int subgridY)
{
    int stride = blockDim.x * gridDim.x;

    int cellNum = grid.overlapSize[0] * grid.subgridTrueSize[1] * grid.conservativesNumber * grid.directionsNumber * 2;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < cellNum; id += stride)
    {
        int true_x = id % grid.overlapSize[0];
        int true_y = (id / grid.overlapSize[0]) % grid.subgridTrueSize[1];
        int c = (id / (grid.overlapSize[0] * grid.subgridTrueSize[1])) % grid.conservativesNumber;
        int d = (id / (grid.overlapSize[0] * grid.subgridTrueSize[1] * grid.conservativesNumber)) % grid.directionsNumber;
        int side = id / (grid.overlapSize[0] * grid.subgridTrueSize[1] * grid.conservativesNumber * grid.directionsNumber); // 0 for left, 1 for right
// printf("grid.overlapSize[0] = %d\n", grid.overlapSize[0]);
// printf("grid.conservativesNumber = %d\n", grid.conservativesNumber);
// printf("cellNum = %d\n", cellNum);
// printf("true_x (2) = %d\n", true_x);
// printf("true_y = %d\n", true_y);
// printf("grid.subgridTrueSize[0]=%d\n", grid.subgridTrueSize[0]);
// printf("grid.subgridTrueSize[1]=%d\n", grid.subgridTrueSize[1]);
// printf("grid.overlapSize[0]=%d\n", grid.overlapSize[0]);
// printf("grid.overlapSize[1]=%d\n", grid.overlapSize[1]);
// printf("d=%d\n", d);

        true_x += side * (grid.subgridTrueSize[0] - grid.overlapSize[0]); // Displace to the right if we target the right side

        if(side == 0)
        {
            int subgrid_id = 
                    c * grid.subgridTrueSize[0] * grid.subgridTrueSize[1]
                    + true_y * grid.subgridTrueSize[0]
                    + true_x;

            subgridWrapped.subgrid[d][subgrid_id] = interface_left[id];
        }
        else
        {
// //printf("dzad %f\n", 3.14);
//             int subgrid_id = 
//                     c * grid.subgridTrueSize[0] * grid.subgridTrueSize[1]
//                     + true_y * grid.subgridTrueSize[0]
//                     + true_x;
// printf("write %f (interface[%d]) (%p) to subgrid[%d][%d]\n",
//     3.14, id - grid.overlapSize[0] * grid.subgridTrueSize[1] * grid.conservativesNumber * grid.directionsNumber, interface_right, d, subgrid_id);
//             subgridWrapped.subgrid[d][subgrid_id] = interface_right[id - grid.overlapSize[0] * grid.subgridTrueSize[1] * grid.conservativesNumber * grid.directionsNumber];

            int subgrid_id = 
                    c * grid.subgridTrueSize[0] * grid.subgridTrueSize[1]
                    + true_y * grid.subgridTrueSize[0]
                    + true_x;

            subgridWrapped.subgrid[d][subgrid_id] = interface_right[id - grid.overlapSize[0] * grid.subgridTrueSize[1] * grid.conservativesNumber * grid.directionsNumber];
        }
    }
}

// Write the upper and lower slices to corresponding interface
// Only the relevant direction is written, i.e., subgrid_d.subgrid[0] goes to interface_down and subgrid_d.subgrid[2] goes to interface_up
__global__
void d2q9_read_vertical_slices(Grid grid, SubgridArray subgridWrapped, PRECISION *interface_down, PRECISION *interface_up, int subgridX, int subgridY)
{
    int stride = blockDim.x * gridDim.x;

    // assert(grid.overlapSize[1] == 1);
    // grid.overlapSize[1] has been replaced with "1"

    int cellNum = 1 * grid.subgridTrueSize[0] * grid.conservativesNumber * 2;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < cellNum; id += stride)
    {
        int true_x = id % grid.subgridTrueSize[0];
        int true_y = 0;//(id / grid.subgridTrueSize[0]) % 1;
        int c = (id / (grid.subgridTrueSize[0] * 1)) % grid.conservativesNumber;
        int side = id / (grid.subgridTrueSize[0] * 1 * grid.conservativesNumber); // 0 for down, 1 for up

        //true_y += side * (grid.subgridTrueSize[1] - 1 - grid.overlapSize[1]); // Displace to the up if we target the up side
        true_y += side * (grid.subgridLogicalSize[1]); // Displace to the up if we target the up side
        true_y += (1-side) * (grid.overlapSize[1]); //

        if(side == 0)
        {
            int subgrid_id =
                    c * grid.subgridTrueSize[0] * grid.subgridTrueSize[1]
                    + true_y * grid.subgridTrueSize[0]
                    + true_x;

            interface_down[id] = subgridWrapped.subgrid[0][subgrid_id];
            //interface_down[id] = subgridY*10;
            // if(c == 1)
            // printf("write %f (subgrid(%d,%d)[%d,%d,%d][%d](%p)) to interface_down[%d]\n",
            //     subgridWrapped.subgrid[2][subgrid_id], subgridX, subgridY, c, true_y, true_x, subgrid_id, subgridWrapped.subgrid[2], id);
        }
        else
        {
            int subgrid_id =
                    c * grid.subgridTrueSize[0] * grid.subgridTrueSize[1]
                    + true_y * grid.subgridTrueSize[0]
                    + true_x;

            interface_up[id - 1 * grid.subgridTrueSize[0] * grid.conservativesNumber] = subgridWrapped.subgrid[2][subgrid_id];
            //interface_up[id - 1 * grid.subgridTrueSize[0] * grid.conservativesNumber] = subgridY*100;
            // if(c == 1)
            // printf("write %f (subgrid(%d,%d)[%d,%d,%d][%d](%p)) to interface_up[%d]\n",
            //     subgridWrapped.subgrid[0][subgrid_id], subgridX, subgridY, c, true_y, true_x, subgrid_id, subgridWrapped.subgrid[2], id - 1 * grid.subgridTrueSize[0] * grid.conservativesNumber);
        }
    }
}





// void d2q9_relax(d2q9 *lbm) {

// #pragma omp for schedule(static) nowait
//   for (size_t i = 0; i < lbm->nx; i++) {
//     for (size_t j = 0; j < lbm->ny; j++) {
//       PRECISION f[9];
//       PRECISION feq[9];
//       for (size_t k = 0; k < 9; k++) {
//         f[k] = lbm->fnext[k][i * lbm->ny + j];
//       }
//       PRECISION w[3];
//       kin_to_fluid(f, w, lbm);
//       for (size_t k = 0; k < 3; k++) {
//         lbm->w[k][i * lbm->ny + j] = w[k];
//         // printf("u=%f\n",w[1]/w[0]);
//       }
//       fluid_to_kin(w, feq, lbm);
//       for (size_t k = 0; k < 9; k++) {
//         lbm->f[k][i * lbm->ny + j] =
// 	  _RELAX * feq[k] + (1 - _RELAX) * lbm->fnext[k][i * lbm->ny + j];
//       }
//     }
//   }
// }

// void d2q9_boundary(d2q9 *lbm) {

// #pragma omp for schedule(static) nowait
//   for (size_t i = 0; i < lbm->nx; i++) {
//     for (size_t j = 0; j < lbm->ny; j++) {
//       PRECISION x = i * lbm->dx;
//       PRECISION y = j * lbm->dx;
//       if (mask(x, y)) {
//         PRECISION wb[3];
//         imposed_data(x, y, lbm->tnow, wb);
//         PRECISION fb[9];
//         fluid_to_kin(wb, fb, lbm);
//         for (size_t k = 0; k < 9; k++) {
//           lbm->f[k][i * lbm->ny + j] =
// 	    _RELAX * fb[k] + (1 - _RELAX) * lbm->f[k][i * lbm->ny + j];
//         }
//       }
//     }
//   }
// }



// time step kernel
__global__
void d2q9_LBM_step(Grid grid,
                        SubgridArray subgrid_FROM_D,
                        SubgridArray subgrid_TO_D,
                        int horizontal_uncomputed_number, int vertical_uncomputed_number,
                        bool has_from_interface_horizontal,
                        bool has_from_interface_vertical,
                        bool has_to_interface_horizontal,
                        bool has_to_interface_vertical,
                        PRECISION *interface_down, PRECISION *interface_up,
                        int subgridX, int subgridY)
{
    int stride = blockDim.x * gridDim.x;

    int cellNum = grid.subgridTrueSize[0] * grid.subgridTrueSize[1];

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < cellNum; id += stride)
    {
        int subgrid_true_x = id % grid.subgridTrueSize[0];
        int subgrid_true_y = id / grid.subgridTrueSize[0];

        bool isInComputationArea=
            !(
            subgrid_true_x < horizontal_uncomputed_number || subgrid_true_x >= grid.subgridTrueSize[0] - horizontal_uncomputed_number ||
            subgrid_true_y < vertical_uncomputed_number || subgrid_true_y >= grid.subgridTrueSize[1] - vertical_uncomputed_number
            );


        PRECISION f[3][3];

        // shift
        #pragma unroll
        for(int d=0; d<grid.directionsNumber; d++)
        {
            PRECISION *target_FROM_subgrid = subgrid_FROM_D.subgrid[d];

            #pragma unroll
            for(int c=0; c<grid.conservativesNumber; c++)
            {
                int i=c+d*grid.conservativesNumber;

                int target_true_x = subgrid_true_x - (PRECISION)get_dir(i,0);
                int target_true_y = subgrid_true_y - (PRECISION)get_dir(i,1);

                int position_in_interface_down_x = target_true_x;
                int position_in_interface_down_y = target_true_y;
                int position_in_interface_up_x = target_true_x;
                int position_in_interface_up_y = target_true_y - grid.subgridTrueSize[1] + grid.overlapSize[1];


                if(has_from_interface_vertical && position_in_interface_down_y >= 0 && position_in_interface_down_y < grid.overlapSize[1] && d==2)
                { // Read from the down interface
                    assert(target_true_y == 0);
                    assert(grid.overlapSize[1] == 1);
                    f[d][c] = interface_down[c*grid.subgridTrueSize[0] + position_in_interface_down_x];
                }
                else if(has_from_interface_vertical && position_in_interface_up_y >= 0 && position_in_interface_up_y < grid.overlapSize[1] && d==0)
                { // Read from the up interface
                    assert(target_true_y == grid.subgridTrueSize[1] - 1);
                    assert(grid.overlapSize[1] == 1);
                    f[d][c] = interface_up[c*grid.subgridTrueSize[0] + position_in_interface_up_x];
                }
                else if(isInComputationArea)
                { // Main case: in the logical space
                    f[d][c] = target_FROM_subgrid[c*grid.subgridTrueSize[0]*grid.subgridTrueSize[1] + target_true_y * grid.subgridTrueSize[0] + target_true_x];
                }
            }
        }

        // relax
        
        if(isInComputationArea)
        {
            // compute x and y as PRECISIONs on the whole grid (xg and yg)
            int true_x = subgrid_true_x;
            int true_y = subgrid_true_y;
            int logical_x = true_x - grid.overlapSize[0];
            int logical_y = true_y - grid.overlapSize[1];
            int xgInt = logical_x + subgridX * grid.subgridOwnedSize[0];
            int ygInt = logical_y + subgridY * grid.subgridOwnedSize[1];
            PRECISION xg = grid.physicalMinCoords[0] + (((PRECISION)xgInt + 0.5) * grid.physicalSize[0] / (PRECISION)grid.size[0]);
            PRECISION yg = grid.physicalMinCoords[1] + (((PRECISION)ygInt + 0.5) * grid.physicalSize[1] / (PRECISION)grid.size[1]);

            PRECISION w[3];

            // f equilibrium
            kin_to_fluid(&f[0][0], w);
            PRECISION feq[3][3];
            fluid_to_kin(w, &feq[0][0]);

            // f fixed (for boundary conditions)
            PRECISION ffixed[3][3];
            if(is_in_cylinder(xg, yg))
            {
                d2q9_t0(w, xg, yg);
                fluid_to_kin(w, &ffixed[0][0]);
            }


            for(int d=0; d<grid.directionsNumber; d++)
            {
                for(int c=0; c<grid.conservativesNumber; c++)
                {

                    f[d][c] = OMEGA_RELAX*feq[d][c] + (1.0 - OMEGA_RELAX)*f[d][c];
                    if(is_in_cylinder(xg, yg))
                    {
                        f[d][c] = OMEGA_RELAX*ffixed[d][c] + (1.0 - OMEGA_RELAX)*f[d][c];
                    }
                }
            }
        }

        int position_in_interface_right_x = subgrid_true_x - grid.subgridOwnedSize[0];
        int position_in_interface_right_y = subgrid_true_y;

        int position_in_interface_down_x = subgrid_true_x;
        int position_in_interface_down_y = subgrid_true_y - grid.overlapSize[1];
        int position_in_interface_up_x = subgrid_true_x;
        int position_in_interface_up_y = subgrid_true_y - grid.subgridOwnedSize[1];

        #pragma unroll
        for(int d=0; d<grid.directionsNumber; d++)
        {
            PRECISION *target_TO_subgrid = subgrid_TO_D.subgrid[d];
            #pragma unroll
            for(int c=0; c<grid.conservativesNumber; c++)
            {
                if(has_to_interface_vertical && position_in_interface_down_y >= 0 && position_in_interface_down_y < grid.overlapSize[1] && d==0)
                {
                    assert(grid.overlapSize[1] == 1);
                    interface_down[c*grid.subgridTrueSize[0] + position_in_interface_down_x] = f[d][c];
                }
                if(has_to_interface_vertical && position_in_interface_up_y >= 0 && position_in_interface_up_y < grid.overlapSize[1] && d==2)
                {
                    assert(grid.overlapSize[1] == 1);
                    interface_up[c*grid.subgridTrueSize[0] + position_in_interface_up_x] = f[d][c];
                }
                
                // In we are in the computation area, we write to the subgrid
                if(isInComputationArea)
                {
                    target_TO_subgrid[c*grid.subgridTrueSize[0]*grid.subgridTrueSize[1] + subgrid_true_y * grid.subgridTrueSize[0] + subgrid_true_x] = f[d][c];
                }
            }
        }
    }
}

// Same kernel, but opimized
__global__
void d2q9_LBM_step_optimized(Grid grid,
                        SubgridArray subgrid_FROM_D,
                        SubgridArray subgrid_TO_D,
                        int horizontal_uncomputed_number, int vertical_uncomputed_number,
                        bool has_from_interface_horizontal,
                        bool has_from_interface_vertical,
                        bool has_to_interface_horizontal,
                        bool has_to_interface_vertical,
                        PRECISION *interface_down, PRECISION *interface_up,
                        int subgridX, int subgridY)
{
    int x_min = horizontal_uncomputed_number;
    int x_max = grid.subgridTrueSize[0] - horizontal_uncomputed_number - 1;
    int y_min = vertical_uncomputed_number;
    int y_max = grid.subgridTrueSize[1] - vertical_uncomputed_number - 1;

    for(int yt=y_min+blockIdx.x;yt<=y_max;yt+=gridDim.x)
    {
        for(int xt=x_min+threadIdx.x;xt<=x_max;xt+=blockDim.x)
        {
            PRECISION f[3][3];
            // shift
            for(int d=0; d<grid.directionsNumber; d++)
            {
                PRECISION *target_FROM_subgrid = subgrid_FROM_D.subgrid[d];

                for(int c=0; c<grid.conservativesNumber; c++)
                {
                    int i=c+d*grid.conservativesNumber;

                    int offset_x = 1-c; // For some reason, get_dir is very slow
                    int offset_y = 1-d;

                    if(has_from_interface_vertical && d==2 && yt==y_min)
                    {
                        f[d][c] = interface_down[c*grid.subgridTrueSize[0] + xt];
                        // f[d][c] = 0.0;
                    }
                    else if(has_from_interface_vertical && d==0 && yt==y_max)
                    {
                        f[d][c] = interface_up[c*grid.subgridTrueSize[0] + xt];
                        // f[d][c] = 0.0;
                    }
                    else
                    {
                        int true_id = c*grid.subgridTrueSize[0]*grid.subgridTrueSize[1] + (yt+offset_y) * grid.subgridTrueSize[0] + (xt+offset_x);
                        f[d][c] = target_FROM_subgrid[true_id];
                    }


                    // int true_id = c*grid.subgridTrueSize[0]*grid.subgridTrueSize[1] + (yt+offset_y) * grid.subgridTrueSize[0] + (xt+offset_x);
                    // f[d][c] = target_FROM_subgrid[true_id];
                }
            }

            int xgInt = xt + subgridX * grid.subgridOwnedSize[0];
            int ygInt = yt + subgridY * grid.subgridOwnedSize[1];
            PRECISION xg = grid.physicalMinCoords[0] + (((PRECISION)xgInt + 0.5) * grid.physicalSize[0] / (PRECISION)grid.size[0]);
            PRECISION yg = grid.physicalMinCoords[1] + (((PRECISION)ygInt + 0.5) * grid.physicalSize[1] / (PRECISION)grid.size[1]);

            PRECISION w[3];

            // f equilibrium
            kin_to_fluid(&f[0][0], w);
            PRECISION feq[3][3];
            fluid_to_kin(w, &feq[0][0]);

            // f fixed (for boundary conditions)
            PRECISION ffixed[3][3];
            if(is_in_cylinder(xg, yg))
            {
                d2q9_t0(w, xg, yg);
                fluid_to_kin(w, &ffixed[0][0]);
            }


            for(int d=0; d<grid.directionsNumber; d++)
            {
                for(int c=0; c<grid.conservativesNumber; c++)
                {
                    f[d][c] = OMEGA_RELAX*feq[d][c] + (1.0 - OMEGA_RELAX)*f[d][c];
                    if(is_in_cylinder(xg, yg))
                    {
                        f[d][c] = OMEGA_RELAX*ffixed[d][c] + (1.0 - OMEGA_RELAX)*f[d][c];
                    }
                }
            }

            // Write to the subgrid
            for(int d=0; d<grid.directionsNumber; d++)
            {
                PRECISION *target_TO_subgrid = subgrid_TO_D.subgrid[d];
                for(int c=0; c<grid.conservativesNumber; c++)
                {
                    target_TO_subgrid[c*grid.subgridTrueSize[0]*grid.subgridTrueSize[1] + yt * grid.subgridTrueSize[0] + xt] = f[d][c];

                    if(has_to_interface_vertical && d==0 && yt==y_min)
                    {
                        interface_down[c*grid.subgridTrueSize[0] + xt] = f[d][c];
                    }
                    if(has_to_interface_vertical && d==2 && yt==y_max)
                    {
                        interface_up[c*grid.subgridTrueSize[0] + xt] = f[d][c];
                    }


                    // target_TO_subgrid[c*grid.subgridTrueSize[0]*grid.subgridTrueSize[1] + yt * grid.subgridTrueSize[0] + xt] = f[d][c];
                }
            }
        }
    }

//     // Just copy the 3 FROM subgrids to the 3 TO subgrids
//     for(int yt=y_min+blockIdx.x;yt<=y_max;yt+=gridDim.x)
//     {
//         for(int xt=x_min+threadIdx.x;xt<=x_max;xt+=blockDim.x)
//         {
//             PRECISION f[3][3];
//             for(int d=0; d<grid.directionsNumber; d++)
//             {
//                 for(int c=0; c<grid.conservativesNumber; c++)
//                 {
//                     int i=c+d*grid.conservativesNumber;
//                     int offset_x = c-1;
//                     int offset_y = d-1;

//                     int true_id = c*grid.subgridTrueSize[0]*grid.subgridTrueSize[1] + (yt+offset_y) * grid.subgridTrueSize[0] + (xt+offset_x);
//                     f[d][c] = subgrid_FROM_D.subgrid[d][true_id];
//                 }
//             }
//             for(int d=0; d<grid.directionsNumber; d++)
//             {
//                 for(int c=0; c<grid.conservativesNumber; c++)
//                 {
//                     subgrid_TO_D.subgrid[d][c*grid.subgridTrueSize[0]*grid.subgridTrueSize[1] + yt * grid.subgridTrueSize[0] + xt] = f[d][c];
//                 }
//             }
//         }
//     }
}

