extern "C" {
#include "LBM.cuh"
}

void d2q9_initial_value_d_caller(Grid grid, double *subgrid, int subgridX, int subgridY, int d)
{
    d2q9_initial_value_d<<<256, 256>>>(grid, subgrid, subgridX, subgridY, d);
}

void d2q9_save_reduce_caller(Grid grid, double *base_subgrid, double *reduced_subgrid, int subgridX, int subgridY, int d)
{
    d2q9_save_reduce<<<256, 256>>>(grid, base_subgrid, reduced_subgrid, subgridX, subgridY, d);
}

void d2q9_read_horizontal_slices_caller(Grid grid, double **subgrid_d, double *interface_left, double *interface_right, int subgridX, int subgridY)
{
    // wrap the array
    SubgridArray subgrid_D_wrapped;
    for(int d=0;d<grid.directionsNumber;d++)
    {
        subgrid_D_wrapped.subgrid[d] = subgrid_d[d];
    }

    d2q9_read_horizontal_slices<<<256, 256>>>(grid, subgrid_D_wrapped, interface_left, interface_right, subgridX, subgridY);
}

void d2q9_write_horizontal_slices_caller(Grid grid, double **subgrid_d, double *interface_left, double *interface_right, int subgridX, int subgridY)
{
    // wrap the array
    SubgridArray subgrid_D_wrapped;
    for(int d=0;d<grid.directionsNumber;d++)
    {
        subgrid_D_wrapped.subgrid[d] = subgrid_d[d];
    }

    d2q9_write_horizontal_slices<<<256, 256>>>(grid, subgrid_D_wrapped, interface_left, interface_right, subgridX, subgridY);
}

void d2q9_read_vertical_slices_caller(Grid grid, double **subgrid_d, double *interface_down, double *interface_up, int subgridX, int subgridY)
{
    // wrap the array
    SubgridArray subgrid_D_wrapped;
    for(int d=0;d<grid.directionsNumber;d++)
    {
        subgrid_D_wrapped.subgrid[d] = subgrid_d[d];
    }

    d2q9_read_vertical_slices<<<256, 256>>>(grid, subgrid_D_wrapped, interface_down, interface_up, subgridX, subgridY);
}

void d2q9_LBM_step_caller(Grid grid,
                double **subgrid_FROM_D,
                double **subgrid_TO_D,
                int horizontal_uncomputed_number, int vertical_uncomputed_number,
                bool has_from_interface_horizontal,
                bool has_from_interface_vertical,
                bool has_to_interface_horizontal,
                bool has_to_interface_vertical,
                double *interface_down, double *interface_up,
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

    d2q9_LBM_step<<<256, 256>>>(grid,
                subgrid_FROM_D_wrapped,
                subgrid_TO_D_wrapped,
                horizontal_uncomputed_number, vertical_uncomputed_number,
                has_from_interface_horizontal,
                has_from_interface_vertical,
                has_to_interface_horizontal,
                has_to_interface_vertical,
                interface_down, interface_up,
                subgridX, subgridY);
}



__device__
int get_dir(int i, int j)
{
    const int dirs[9][2] = {
        {-1, -1},
        {0, -1},
        {1, -1},
        {-1, 0},
        {0, 0},
        {1, 0},
        {-1, 1},
        {0, 1},
        {1, 1}
    };

    return dirs[i][j];
}

__device__
void fluid_to_kin(const double *w, double *f)
{
    static const double c2 = 1. / 3.;
    double dotvel = 0, vel2 = 0, l2 = 0, l4 = 0, c4 = 0;

    l2 = MAXIMUM_VELOCITY * MAXIMUM_VELOCITY;
    double l2_ov_c2 = l2 / c2;

    l4 = l2 * l2;
    c4 = c2 * c2;

    vel2 = (w[1] * w[1] + w[2] * w[2]) / (w[0] * w[0]);
    dotvel = sqrt(l2) * (get_dir(4,0) * w[1] + get_dir(4,1) * w[2]) / w[0];

    f[4] = (4. / 9.) * w[0] *
    (1.0 + (l2_ov_c2)*dotvel + l4 / (2. * c4) * dotvel * dotvel -
    l2 / (2. * c2) * vel2);

    // perpendicular directions
    for (size_t i = 1; i < 9; i+=2) {
        dotvel = sqrt(l2) * (get_dir(i,0) * w[1] + get_dir(i,1) * w[2]) / w[0];
        f[i] = (1. / 9.) * w[0] *
        (1.0 + (l2_ov_c2)*dotvel + l4 / (2. * c4) * dotvel * dotvel -
        l2 / (2. * c2) * vel2);
    }
    // diagonal directions
    for (size_t it = 0; it < 4; it++) {
        size_t i = it * 2 + 2*(it>1);
        dotvel = sqrt(l2) * (get_dir(i,0) * w[1] + get_dir(i,1) * w[2]) / w[0];
        f[i] = (1. / 36.) * w[0] *
        (1.0 + (l2_ov_c2)*dotvel + l4 / (2. * c4) * dotvel * dotvel -
        l2 / (2. * c2) * vel2);
    }
}

__device__
void kin_to_fluid(const double *f, double *w)
{
    w[0] = 0;
    w[1] = 0;
    w[2] = 0;

    for (int i = 0; i < 9; i++) {
        w[0] = w[0] + f[i];
        w[1] = w[1] + MAXIMUM_VELOCITY * get_dir(i,0) * f[i];
        w[2] = w[2] + MAXIMUM_VELOCITY * get_dir(i,1) * f[i];
    }
}


__device__
void d2q9_t0(double w[3], double x, double y)
{
    double rho, u, v;
    rho = 1;
    // u = 0.03;
    // v = 0.00001; // to get instability
    u = 0.03 + 0.01*sin(x+2*y) + 0.05*sin(cos(3*x + y*y) + x*x*y); // TODO ERASE
    v = 0.00001 + 0.01*sin(x*7+2*y)+ 0.05*sin(cos(7*y + y*x) + y*x*y); // TODO ERASE

    if ((x - CYLINDER_CENTER_X) * (x - CYLINDER_CENTER_X) + (y - CYLINDER_CENTER_Y) * (y - CYLINDER_CENTER_Y) < CYLINDER_RADIUS * CYLINDER_RADIUS) {
        u = 0;
        v = 0;
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
void d2q9_initial_value_d(Grid grid, double *subgrid, int subgridX, int subgridY, int d)
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

        double xg = grid.physicalMinCoords[0] + (((double)xgInt + 0.5) * grid.physicalSize[0] / (double)grid.size[0]);
        double yg = grid.physicalMinCoords[1] + (((double)ygInt + 0.5) * grid.physicalSize[1] / (double)grid.size[1]);

        double w[3];
        d2q9_t0(w, xg, yg);
        double f[3][3];
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
void d2q9_save_reduce(Grid grid, double *base_subgrid, double *reduced_subgrid, int subgridX, int subgridY, int d)
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

        double average = 0;

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
__global__ void d2q9_read_horizontal_slices(Grid grid, SubgridArray subgridWrapped, double *interface_left, double *interface_right, int subgridX, int subgridY)
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
__global__ void d2q9_write_horizontal_slices(Grid grid, SubgridArray subgridWrapped, double *interface_left, double *interface_right, int subgridX, int subgridY)
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
void d2q9_read_vertical_slices(Grid grid, SubgridArray subgridWrapped, double *interface_down, double *interface_up, int subgridX, int subgridY)
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

        // if(side == 0)
        // {
        //     int subgrid_id = 
        //             c * grid.subgridTrueSize[0] * grid.subgridTrueSize[1]
        //             + true_y * grid.subgridTrueSize[0]
        //             + true_x;

        //     interface_down[id] = subgridWrapped.subgrid[0][subgrid_id];
        // }
        // else
        // {
        //     int subgrid_id = 
        //             c * grid.subgridTrueSize[0] * grid.subgridTrueSize[1]
        //             + true_y * grid.subgridTrueSize[0]
        //             + true_x;

        //     interface_up[id - 1 * grid.subgridTrueSize[0] * grid.conservativesNumber] = subgridWrapped.subgrid[2][subgrid_id];
        // }

// good
// kernel vertical_interface CPU (0 1 1 3) [45-49] = 0.117939 0.118259 0.118584 0.118913 0.119244
// kernel vertical_interface CPU (0 1 0 3) [45-49] = 0.102232 0.101978 0.101723 0.101467 0.101212
// bad
// kernel vertical_interface CPU (0 1 0 3) [45-49] = 0.099287 0.099078 0.098879 0.098692 0.098518
// kernel vertical_interface CPU (0 1 1 3) [45-49] = 0.121833 0.122129 0.122415 0.122689 0.122952

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
                        double *interface_down, double *interface_up,
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


        double f[3][3];

        // shift
        for(int d=0; d<grid.directionsNumber; d++)
        {
            double *target_FROM_subgrid = subgrid_FROM_D.subgrid[d];

            for(int c=0; c<grid.conservativesNumber; c++)
            {
                int i=c+d*grid.conservativesNumber;

                int target_true_x = subgrid_true_x - get_dir(i,0);
                int target_true_y = subgrid_true_y - get_dir(i,1);

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
            double w[3];
            kin_to_fluid(&f[0][0], w);
            double feq[3][3];
            fluid_to_kin(w, &feq[0][0]);
            for(int d=0; d<grid.directionsNumber; d++)
            {
                for(int c=0; c<grid.conservativesNumber; c++)
                {
                    f[d][c] = OMEGA_RELAX*feq[d][c] + (1.0 - OMEGA_RELAX)*f[d][c];
                }
            }
        }
        
        int position_in_interface_right_x = subgrid_true_x - grid.subgridOwnedSize[0];
        int position_in_interface_right_y = subgrid_true_y;

        int position_in_interface_down_x = subgrid_true_x;
        int position_in_interface_down_y = subgrid_true_y - grid.overlapSize[1];
        int position_in_interface_up_x = subgrid_true_x;
        int position_in_interface_up_y = subgrid_true_y - grid.subgridOwnedSize[1];

        for(int d=0; d<grid.directionsNumber; d++)
        {
            double *target_TO_subgrid = subgrid_TO_D.subgrid[d];
            for(int c=0; c<grid.conservativesNumber; c++)
            {
                /*if(has_to_interface_vertical && position_in_interface_down_x >= 0 && position_in_interface_down_x < grid.overlapSize[1])
                { // Write to the up interface
                    assert(position_in_interface_down_y == 0);
                    assert(grid.overlapSize[1] == 1);
                    interface_down[c*grid.subgridTrueSize[0] + position_in_interface_down_x] = f[d][c];
                }
                if(has_to_interface_vertical && position_in_interface_up_x >= 0 && position_in_interface_up_x < grid.overlapSize[1])
                { // Write to the down interface
                    assert(position_in_interface_up_y == 0);
                    assert(grid.overlapSize[1] == 1);
                    interface_up[c*grid.subgridTrueSize[0] + position_in_interface_up_x] = f[d][c];
                }
                if(has_to_interface_horizontal && position_in_interface_left_y >= 0 && position_in_interface_left_y < grid.overlapSize[1])
                { // Write to the left interface
                    interface_left[c*grid.overlapSize[0]*grid.subgridTrueSize[1] + position_in_interface_left_y*grid.overlapSize[0] + position_in_interface_left_x] = f[d][c];
                }
                if(has_to_interface_horizontal && position_in_interface_right_y >= 0 && position_in_interface_right_y < grid.overlapSize[1])
                { // Write to the right interface
                    interface_right[c*grid.overlapSize[0]*grid.subgridTrueSize[1] + position_in_interface_right_y*grid.overlapSize[0] + position_in_interface_right_x] = f[d][c];
                }*/

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
