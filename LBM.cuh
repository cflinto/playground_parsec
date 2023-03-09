#ifndef LBM_CUH_INCLUDED
#define LBM_CUH_INCLUDED

#include <assert.h>
#include <stdio.h>

#include "LBM_common.h"


__device__ void d2q9_t0(double *w, double x, double y, int d);

// Initialize all the values including the ghost cells
__global__ void d2q9_initial_value_d(Grid grid, double *subgrid, int subgridX, int subgridY, int d);

__global__ void d2q9_save_reduce(Grid grid, double *base_subgrid, double *reduced_subgrid, int subgridX, int subgridY, int d);

__global__ void d2q9_LBM_step(Grid grid, double (*subgrid_FROM_D)[], double (*subgrid_TO_D)[], int subgridX, int subgridY);

#endif // LBM_CUH_INCLUDED