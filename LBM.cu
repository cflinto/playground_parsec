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

__device__
void d2q9_t0(double *w, double x, double y, int d)
{
    (void)d; // same for all directions

    double rho, u, v;
    rho = 1;
    u = 0.03;
    v = 0.00001; // to get instability

    if ((x - CYLINDER_CENTER_X) * (x - CYLINDER_CENTER_X) + (y - CYLINDER_CENTER_Y) * (y - CYLINDER_CENTER_Y) < CYLINDER_RADIUS * CYLINDER_RADIUS) {
        u = 0;
        // printf("x=%f y=%f\n",x,y);
    }

    w[0] = rho;
    w[1] = rho * u;
    w[2] = rho * v;
}


// Initialize all the values including the ghost cells
__global__
void d2q9_initial_value_d(Grid grid, double *subgrid, int subgridX, int subgridY, int d)
{
    (void)d; // same for all directions

    int stride = blockDim.x * gridDim.x;

	int cellNum = grid.subgridTrueSize[0] * grid.subgridTrueSize[1] * grid.conservativesNumber;

	for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < cellNum; id += stride)
	{
        int true_x = id % grid.subgridTrueSize[0];
        int true_y = (id / grid.subgridTrueSize[0]) % grid.subgridTrueSize[1];
        int c = id / (grid.subgridTrueSize[0] * grid.subgridTrueSize[1]);

        int logical_x = true_x - grid.overlapSize[0];
        int logical_y = true_y - grid.overlapSize[1];

        int xgInt = logical_x + subgridX * grid.subgridOwnedSize[0];
        int ygInt = logical_y + subgridY * grid.subgridOwnedSize[1];

        double xg = grid.physicalMinCoords[0] + (((double)xgInt + 0.5) * grid.physicalSize[0] / (double)grid.size[0]);
        double yg = grid.physicalMinCoords[1] + (((double)ygInt + 0.5) * grid.physicalSize[1] / (double)grid.size[1]);

        double *w = &subgrid[id];

        d2q9_t0(w, xg, yg, d);
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
