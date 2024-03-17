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


__host__ __device__ void kin_to_fluid(const PRECISION *f, PRECISION *w);
__host__ __device__ void fluid_to_kin(const PRECISION *w, PRECISION *f);

__device__ void d2q9_t0(PRECISION *w, PRECISION x, PRECISION y, int d);

// Initialize all the values including the ghost cells
__global__ void d2q9_initial_value_d(Grid grid, PRECISION *subgrid, int subgridX, int subgridY, int d);

__global__ void d2q9_save_reduce(Grid grid, PRECISION *base_subgrid, PRECISION *reduced_subgrid, int subgridX, int subgridY, int d);


__global__ void d2q9_read_horizontal_slices(Grid grid, SubgridArray subgrid_d, PRECISION *interface_left, PRECISION *interface_right, int subgridX, int subgridY);
__global__ void d2q9_write_horizontal_slices(Grid grid, SubgridArray subgrid_d, PRECISION *interface_left, PRECISION *interface_right, int subgridX, int subgridY);
__global__ void d2q9_read_vertical_slices(Grid grid, SubgridArray subgrid_d, PRECISION *interface_down, PRECISION *interface_up, int subgridX, int subgridY);

__global__ void d2q9_LBM_step_original(Grid grid,
                        SubgridArray subgrid_FROM_D,
                        SubgridArray subgrid_TO_D,
                        int horizontal_uncomputed_number, int vertical_uncomputed_number,
                        bool has_from_interface_horizontal,
                        bool has_from_interface_vertical,
                        bool has_to_interface_horizontal,
                        bool has_to_interface_vertical,
                        PRECISION *interface_down, PRECISION *interface_up,
                        int subgridX, int subgridY);
__global__ void d2q9_LBM_step_all_threads_domain(Grid grid,
                        SubgridArray subgrid_FROM_D,
                        SubgridArray subgrid_TO_D,
                        int horizontal_uncomputed_number, int vertical_uncomputed_number,
                        bool has_from_interface_horizontal,
                        bool has_from_interface_vertical,
                        bool has_to_interface_horizontal,
                        bool has_to_interface_vertical,
                        PRECISION *interface_down, PRECISION *interface_up,
                        int subgridX, int subgridY);
__global__ void d2q9_LBM_step_one_block_per_line(Grid grid,
                        SubgridArray subgrid_FROM_D,
                        SubgridArray subgrid_TO_D,
                        int horizontal_uncomputed_number, int vertical_uncomputed_number,
                        bool has_from_interface_horizontal,
                        bool has_from_interface_vertical,
                        bool has_to_interface_horizontal,
                        bool has_to_interface_vertical,
                        PRECISION *interface_down, PRECISION *interface_up,
                        int subgridX, int subgridY);

#endif // LBM_CUH_INCLUDED