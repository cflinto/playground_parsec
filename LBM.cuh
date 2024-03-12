#ifndef LBM_CUH_INCLUDED
#define LBM_CUH_INCLUDED

#include <assert.h>
#include <stdio.h>

#include "LBM_common.h"

#define gpuErrchk(ans)                        \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__, true); \
	}

void gpuAssert(cudaError_t code, const char *file, int line, bool abort);


__host__ __device__ void kin_to_fluid(const double *f, double *w);
__host__ __device__ void fluid_to_kin(const double *w, double *f);

__device__ void d2q9_t0(double *w, double x, double y, int d);

// Initialize all the values including the ghost cells
__global__ void d2q9_initial_value_d(Grid grid, double *subgrid, int subgridX, int subgridY, int d);

__global__ void d2q9_save_reduce(Grid grid, double *base_subgrid, double *reduced_subgrid, int subgridX, int subgridY, int d);


__global__ void d2q9_read_horizontal_slices(Grid grid, SubgridArray subgrid_d, double *interface_left, double *interface_right, int subgridX, int subgridY);
__global__ void d2q9_write_horizontal_slices(Grid grid, SubgridArray subgrid_d, double *interface_left, double *interface_right, int subgridX, int subgridY);
__global__ void d2q9_read_vertical_slices(Grid grid, SubgridArray subgrid_d, double *interface_down, double *interface_up, int subgridX, int subgridY);

__global__ void d2q9_LBM_step(Grid grid,
                        SubgridArray subgrid_FROM_D,
                        SubgridArray subgrid_TO_D,
                        int horizontal_uncomputed_number, int vertical_uncomputed_number,
                        bool has_from_interface_horizontal,
                        bool has_from_interface_vertical,
                        bool has_to_interface_horizontal,
                        bool has_to_interface_vertical,
                        double *interface_down, double *interface_up,
                        int subgridX, int subgridY);

#endif // LBM_CUH_INCLUDED