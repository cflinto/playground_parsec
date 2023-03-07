#ifndef LBM_CUH_INCLUDED
#define LBM_CUH_INCLUDED

#include "LBM_common.h"


__device__ void d2q9_t0(double *w, double x, double y, int d);

// Initialize all the values including the ghost cells
__global__ void d2q9_initial_value_d(Grid grid, double *subgrid, int subgridX, int subgridY, int d);

#endif // LBM_CUH_INCLUDED