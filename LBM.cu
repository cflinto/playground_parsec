extern "C" {
#include "LBM.cuh"
}

void d2q9_initial_value_d_caller(Grid grid, double *subgrid, int subgridX, int subgridY, int d)
{
    d2q9_initial_value_d<<<256, 256>>>(grid, subgrid, subgridX, subgridY, d);
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
