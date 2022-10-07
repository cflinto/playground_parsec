#include "parsec.h"
#include "parsec/parsec_internal.h"
#include "parsec/ayudame.h"
#include "parsec/execution_stream.h"
#if defined(PARSEC_HAVE_CUDA)
#include "parsec/mca/device/cuda/device_cuda.h"
#endif  /* defined(PARSEC_HAVE_CUDA) */
#if defined(_MSC_VER) || defined(__MINGW32__)
#  include <malloc.h>
#else
#  include <alloca.h>
#endif  /* defined(_MSC_VER) || defined(__MINGW32__) */

#define PARSEC_LBM_NB_TASK_CLASSES 4
#define PARSEC_LBM_NB_DATA 1

typedef struct __parsec_LBM_internal_taskpool_s __parsec_LBM_internal_taskpool_t;
struct parsec_LBM_internal_taskpool_s;

/** Predeclarations of the parsec_task_class_t */
static const parsec_task_class_t LBM_WriteBack;
static const parsec_task_class_t LBM_Exchange;
static const parsec_task_class_t LBM_LBM_STEP;
static const parsec_task_class_t LBM_FillGrid;
/** Predeclarations of the parameters */
static const parsec_flow_t flow_of_LBM_WriteBack_for_FINAL_GRID;
static const parsec_flow_t flow_of_LBM_Exchange_for_GRID_TO;
static const parsec_flow_t flow_of_LBM_LBM_STEP_for_GRID_CD_0_0;
static const parsec_flow_t flow_of_LBM_LBM_STEP_for_GRID_CD_1_0;
static const parsec_flow_t flow_of_LBM_LBM_STEP_for_GRID_CD_2_0;
static const parsec_flow_t flow_of_LBM_LBM_STEP_for_GRID_CD_0_1;
static const parsec_flow_t flow_of_LBM_LBM_STEP_for_GRID_CD_1_1;
static const parsec_flow_t flow_of_LBM_LBM_STEP_for_GRID_CD_2_1;
static const parsec_flow_t flow_of_LBM_LBM_STEP_for_X;
static const parsec_flow_t flow_of_LBM_FillGrid_for_INITIAL_GRID;
#line 3 "LBM.jdf"

/**
 * This second example shows how to create a simple jdf that has only one single task.
 *    JDF syntax
 *    parsec_JDFNAME_New()
 *    parsec_context_add_taskpool()
 *    parsec_data_collection_init()
 *
 * Can play with the HelloWorld bounds to show embarissingly parallel algorithm.
 *
 * @version 3.0
 * @email parsec-users@icl.utk.edu
 *
 */

#define gpuErrchk(ans)                        \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__, true); \
	}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

#include <math.h>
#include "parsec.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/multidimensional_grid.h"

#include "cublas_v2.h"

// LBM defines

#define EPSILON 0.000000001

#define CFL 0.45
#define ALPHA 0.9
#define BETA 0.9


// HPC defines

#define DIMENSIONS_NUMBER 3


#define MAX_SUBGRIDS 256

#define PROBLEM_SIZE_X (64)
#define PROBLEM_SIZE_Y (64)
#define PROBLEM_SIZE_Z (64)

typedef struct Grid
{
    // Difference between the logical size, the true size, and the owned size in 1D:
    // Subgrid:
    // OOOAAAAAAAAASOOO

    // Terminology:
    // O = overlap (= ghost cells = halo)
    // A = owned
    // S = shared
    // true size = O + A + S
    // logical size = A + S
    // owned size = A
    // In this case (overlap=3) and with a basic 2-direction scheme (left/right):
    // We can perform 3 consecutive steps (because we have 3 ghost cells)
    // After the 3 steps, A and S are set to the right values, but O is not.
    // The overlap must, therefore, be synchronized with the neighbors at this point in time.

    // Conceptually, the grid is a 5D array: (x, y, z, c, d) of size (size[0], size[1], size[2], conservativesNumber, directionsNumber)
	int size[DIMENSIONS_NUMBER];
    int conservativesNumber;
    int directionsNumber;

	int subgridNumber[DIMENSIONS_NUMBER];
	int subgridLogicalSizeCompressed[DIMENSIONS_NUMBER];
	int subgridOwnedSize[DIMENSIONS_NUMBER]; // cells owned uniquely by this subgrid
	int subgridLogicalSize[DIMENSIONS_NUMBER]; // cells that need computation on this subgrid (includes cells that share the same computation)
	int subgridTrueSize[DIMENSIONS_NUMBER]; // all the cells of the subgrid (including ghost cells = halo = overlap)
    int sharedLayers[DIMENSIONS_NUMBER][2]; // number of values that are shared with the neighbor [dim][dir] (shared meaning that the values are computed twice and not exchanged)
	int overlapSize[DIMENSIONS_NUMBER]; // depth of the overlap on each dimension


    // Conceptually, the grid is a 5D array: (x, y, z, c, d)

	size_t subgridCompressedSize[MAX_SUBGRIDS];
	size_t subgridTempSize[MAX_SUBGRIDS];
	int cellsPerSubgrid;
	int subgridsNumber;
	int currentSubgrid;

	double mass[MAX_SUBGRIDS];

    // physical coords of the problem, used by the physical model
	double physicalMinCoords[DIMENSIONS_NUMBER];
	double physicalSize[DIMENSIONS_NUMBER];

    parsec_multidimensional_grid_t *desc;
} Grid;

Grid newGrid(int rank, int nodes)
{
	Grid grid;

	grid.size[0] = PROBLEM_SIZE_X;
	grid.size[1] = PROBLEM_SIZE_Y;
	grid.size[2] = PROBLEM_SIZE_Z;
    grid.conservativesNumber = 3;
    grid.directionsNumber = 2;

	grid.subgridNumber[0] = 2;
	grid.subgridNumber[1] = 2;
	grid.subgridNumber[2] = 2;
	grid.subgridOwnedSize[0] = grid.size[0] / grid.subgridNumber[0];
	grid.subgridOwnedSize[1] = grid.size[1] / grid.subgridNumber[1];
	grid.subgridOwnedSize[2] = grid.size[2] / grid.subgridNumber[2];

    grid.sharedLayers[0][0] = 0;
    grid.sharedLayers[0][1] = 1;
    grid.sharedLayers[1][0] = 0;
    grid.sharedLayers[1][1] = 1;
    grid.sharedLayers[2][0] = 0;
    grid.sharedLayers[2][1] = 1;

    for(int d=0;d<DIMENSIONS_NUMBER;++d)
    {
        grid.subgridLogicalSize[d] = grid.subgridOwnedSize[d] + grid.sharedLayers[d][0] + grid.sharedLayers[d][1];
    }


	grid.overlapSize[0] = 1;
	grid.overlapSize[1] = 3;
	grid.overlapSize[2] = 4;

	grid.currentSubgrid = 0;

	grid.physicalMinCoords[0] = -1;
	grid.physicalMinCoords[1] = -1;
	grid.physicalMinCoords[2] = -1;
	grid.physicalSize[0] = 2;
	grid.physicalSize[1] = 2;
	grid.physicalSize[2] = 2;

	grid.subgridLogicalSizeCompressed[0] = grid.subgridLogicalSize[0];
	grid.subgridLogicalSizeCompressed[1] = grid.subgridLogicalSize[1];
	grid.subgridLogicalSizeCompressed[2] = grid.subgridLogicalSize[2];
	grid.subgridTrueSize[0] = grid.subgridLogicalSize[0] + 2 * grid.overlapSize[0];
	grid.subgridTrueSize[1] = grid.subgridLogicalSize[1] + 2 * grid.overlapSize[1];
	grid.subgridTrueSize[2] = grid.subgridLogicalSize[2] + 2 * grid.overlapSize[2];
	grid.cellsPerSubgrid = (grid.subgridTrueSize[0]) * (grid.subgridTrueSize[1]) * (grid.subgridTrueSize[2]);
	grid.subgridsNumber = grid.subgridNumber[0] * grid.subgridNumber[1] * grid.subgridNumber[2];

    int numberOfSharedLayers = 1; // TODO !
    for(int d=0;d<DIMENSIONS_NUMBER;++d)
    {
        assert(grid.size[d] % grid.subgridOwnedSize[d] == 0);
        assert(grid.subgridNumber[d]*grid.subgridOwnedSize[d] == grid.size[d]);
        assert(grid.subgridOwnedSize[d] <= grid.subgridLogicalSize[d]);
        assert(grid.subgridLogicalSize[d] <= grid.subgridTrueSize[d]);
    }

	for (int i = 0; i < grid.subgridsNumber; ++i)
	{
		// init to a random value
		grid.subgridCompressedSize[i] = 0;
		grid.subgridTempSize[i] = 0;
	}

    grid.desc = (parsec_multidimensional_grid_t*)malloc(sizeof(parsec_multidimensional_grid_t)*grid.conservativesNumber*grid.directionsNumber);
    assert(grid.desc != NULL);

    for(int i=0;i<grid.conservativesNumber*grid.directionsNumber;++i)
    {
        parsec_multidimensional_grid_init(&grid.desc[i],
                               PARSEC_MATRIX_DOUBLE,
                               nodes, rank,
                               5, 3,
                               grid.subgridNumber[0], grid.subgridNumber[1], grid.subgridNumber[2], grid.conservativesNumber, grid.directionsNumber,
                               grid.subgridTrueSize[0], grid.subgridTrueSize[1], grid.subgridTrueSize[2],
                               1, 1, 1, 1, 1);
        grid.desc[i].grid = parsec_data_allocate((size_t)grid.desc[i].nb_local_tiles *
                                        (size_t)grid.desc[i].bsiz *
                                        (size_t)parsec_datadist_getsizeoftype(grid.desc[i].mtype));
        assert(grid.desc[i].grid != NULL);

        parsec_data_collection_set_key((parsec_data_collection_t*)&grid.desc[i], "grid.desc[");
    }

	return grid;
}


typedef cublasStatus_t (*cublas_dgemm_v2_t) ( cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const double *alpha,
                            const double *A, int lda,
                            const double *B, int ldb,
                            const double *beta,
                            double       *C, int ldc);


#if defined(PARSEC_HAVE_CUDA)
static void destruct_cublas_handle(void *p)
{
    cublasHandle_t handle = (cublasHandle_t)p;
    cublasStatus_t status;
    if(NULL != handle) {
        status = cublasDestroy(handle);
        assert(status == CUBLAS_STATUS_SUCCESS);
        (void)status;
    }
}

static void *create_cublas_handle(void *obj, void *p)
{
    cublasHandle_t handle;
    cublasStatus_t status;
    parsec_cuda_exec_stream_t *stream = (parsec_cuda_exec_stream_t *)obj;
    (void)p;
    /* No need to call cudaSetDevice, as this has been done by PaRSEC before calling the task body */
    status = cublasCreate(&handle);
    assert(CUBLAS_STATUS_SUCCESS == status);
    status = cublasSetStream(handle, stream->cuda_stream);
    assert(CUBLAS_STATUS_SUCCESS == status);
    (void)status;
    return (void*)handle;
}
#endif

static void destroy_cublas_handle(void *_h, void *_n)
{
#if defined(PARSEC_HAVE_CUDA)
    cublasHandle_t cublas_handle = (cublasHandle_t)_h;
    cublasDestroy_v2(cublas_handle);
#endif
    (void)_n;
    (void)_h;
}

int cd;

#line 286 "LBM.c"
#define PARSEC_LBM_DEFAULT_ADT    (&__parsec_tp->super.arenas_datatypes[PARSEC_LBM_DEFAULT_ADT_IDX])
#include "LBM.h"

struct __parsec_LBM_internal_taskpool_s {
 parsec_LBM_taskpool_t super;
 volatile int32_t sync_point;
 volatile int32_t  initial_number_tasks;
 parsec_task_t* startup_queue;
  /* The ranges to compute the hash key */
  int WriteBack_x_range;
  int WriteBack_y_range;
  int WriteBack_z_range;
  int WriteBack_c_range;
  int WriteBack_d_range;
  int Exchange_x_range;
  int Exchange_y_range;
  int Exchange_z_range;
  int Exchange_s_range;
  int Exchange_conservative_range;
  int Exchange_direction_range;
  int Exchange_dimension_range;
  int Exchange_side_range;
  int LBM_STEP_x_range;
  int LBM_STEP_y_range;
  int LBM_STEP_z_range;
  int LBM_STEP_s_range;
  int FillGrid_x_range;
  int FillGrid_y_range;
  int FillGrid_z_range;
  int FillGrid_c_range;
  int FillGrid_d_range;
  /* The list of data repositories  WriteBack  Exchange  LBM_STEP  FillGrid */
  data_repo_t* repositories[4];
};

#if defined(PARSEC_PROF_TRACE)
#  if defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)
static int LBM_profiling_array[4*PARSEC_LBM_NB_TASK_CLASSES] = {-1}; /* 2 pairs (begin, end) per task, times two because each task class has an internal_init task */
#  else /* defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */
static int LBM_profiling_array[2*PARSEC_LBM_NB_TASK_CLASSES] = {-1}; /* 2 pairs (begin, end) per task */
#  endif /* defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */
#endif  /* defined(PARSEC_PROF_TRACE) */
/* Globals */
#define descGridDC __parsec_tp->super._g_descGridDC
#define gridParameters (__parsec_tp->super._g_gridParameters)
#define rank (__parsec_tp->super._g_rank)
#define nodes (__parsec_tp->super._g_nodes)
#define subgrid_number_x (__parsec_tp->super._g_subgrid_number_x)
#define subgrid_number_y (__parsec_tp->super._g_subgrid_number_y)
#define subgrid_number_z (__parsec_tp->super._g_subgrid_number_z)
#define tile_size_x (__parsec_tp->super._g_tile_size_x)
#define tile_size_y (__parsec_tp->super._g_tile_size_y)
#define tile_size_z (__parsec_tp->super._g_tile_size_z)
#define conservatives_number (__parsec_tp->super._g_conservatives_number)
#define directions_number (__parsec_tp->super._g_directions_number)
#define overlap_x (__parsec_tp->super._g_overlap_x)
#define overlap_y (__parsec_tp->super._g_overlap_y)
#define overlap_z (__parsec_tp->super._g_overlap_z)
#define number_of_steps (__parsec_tp->super._g_number_of_steps)
#define CuHI (__parsec_tp->super._g_CuHI)

static inline int parsec_imin(int a, int b) { return (a <= b) ? a : b; };

static inline int parsec_imax(int a, int b) { return (a >= b) ? a : b; };

/* Data Access Macros */
#define data_of_descGridDC(descGridDC_d0, descGridDC_d1, descGridDC_d2, descGridDC_d3, descGridDC_d4)  (((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->data_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, (descGridDC_d0), (descGridDC_d1), (descGridDC_d2), (descGridDC_d3), (descGridDC_d4)))

#define rank_of_descGridDC(descGridDC_d0, descGridDC_d1, descGridDC_d2, descGridDC_d3, descGridDC_d4)  (((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->rank_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, (descGridDC_d0), (descGridDC_d1), (descGridDC_d2), (descGridDC_d3), (descGridDC_d4)))

/* Functions Predicates */
#define WriteBack_pred(x, y, z, c, d) (((parsec_data_collection_t*)(__parsec_tp->super._g_descGridDC))->myrank == rank_of_descGridDC(x, y, z, c, d))
#define Exchange_pred(x, y, z, s, conservative, direction, dimension, side) (((parsec_data_collection_t*)(__parsec_tp->super._g_descGridDC))->myrank == rank_of_descGridDC(x, y, z, conservative, direction))
#define LBM_STEP_pred(x, y, z, s) (((parsec_data_collection_t*)(__parsec_tp->super._g_descGridDC))->myrank == rank_of_descGridDC(x, y, z, 0, 0))
#define FillGrid_pred(x, y, z, c, d) (((parsec_data_collection_t*)(__parsec_tp->super._g_descGridDC))->myrank == rank_of_descGridDC(x, y, z, c, d))

/* Data Repositories */
#define WriteBack_repo (__parsec_tp->repositories[3])
#define Exchange_repo (__parsec_tp->repositories[2])
#define LBM_STEP_repo (__parsec_tp->repositories[1])
#define FillGrid_repo (__parsec_tp->repositories[0])
/* Release dependencies output macro */
#if defined(PARSEC_DEBUG_NOISIER)
#define RELEASE_DEP_OUTPUT(ES, DEPO, TASKO, DEPI, TASKI, RSRC, RDST, DATA)\
  do { \
    char tmp1[128], tmp2[128]; (void)tmp1; (void)tmp2;\
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "thread %d VP %d explore deps from %s:%s to %s:%s (from rank %d to %d) base ptr %p",\
           (NULL != (ES) ? (ES)->th_id : -1), (NULL != (ES) ? (ES)->virtual_process->vp_id : -1),\
           DEPO, parsec_task_snprintf(tmp1, 128, (parsec_task_t*)(TASKO)),\
           DEPI, parsec_task_snprintf(tmp2, 128, (parsec_task_t*)(TASKI)), (RSRC), (RDST), (DATA));\
  } while(0)
#define ACQUIRE_FLOW(TASKI, DEPI, FUNO, DEPO, LOCALS, PTR)\
  do { \
    char tmp1[128], tmp2[128]; (void)tmp1; (void)tmp2;\
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "task %s acquires flow %s from %s %s data ptr %p",\
           parsec_task_snprintf(tmp1, 128, (parsec_task_t*)(TASKI)), (DEPI),\
           (DEPO), parsec_snprintf_assignments(tmp2, 128, (FUNO), (parsec_assignment_t*)(LOCALS)), (PTR));\
  } while(0)
#else
#define RELEASE_DEP_OUTPUT(ES, DEPO, TASKO, DEPI, TASKI, RSRC, RDST, DATA)
#define ACQUIRE_FLOW(TASKI, DEPI, TASKO, DEPO, LOCALS, PTR)
#endif
static inline parsec_key_t __jdf2c_make_key_WriteBack(const parsec_taskpool_t *tp, const parsec_assignment_t *as)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t *)tp;
  __parsec_LBM_WriteBack_parsec_assignment_t ascopy, *assignment = &ascopy;
  uintptr_t __parsec_id = 0;
  memcpy(assignment, as, sizeof(__parsec_LBM_WriteBack_parsec_assignment_t));
  const int x = assignment->x.value;
  int __jdf2c_x_min = 0;
  const int y = assignment->y.value;
  int __jdf2c_y_min = 0;
  const int z = assignment->z.value;
  int __jdf2c_z_min = 0;
  const int c = assignment->c.value;
  int __jdf2c_c_min = 0;
  const int d = assignment->d.value;
  int __jdf2c_d_min = 0;
  __parsec_id += (x - __jdf2c_x_min);
  __parsec_id += (y - __jdf2c_y_min) * __parsec_tp->WriteBack_x_range;
  __parsec_id += (z - __jdf2c_z_min) * __parsec_tp->WriteBack_x_range * __parsec_tp->WriteBack_y_range;
  __parsec_id += (c - __jdf2c_c_min) * __parsec_tp->WriteBack_x_range * __parsec_tp->WriteBack_y_range * __parsec_tp->WriteBack_z_range;
  __parsec_id += (d - __jdf2c_d_min) * __parsec_tp->WriteBack_x_range * __parsec_tp->WriteBack_y_range * __parsec_tp->WriteBack_z_range * __parsec_tp->WriteBack_c_range;
  (void)__parsec_tp;
  return (parsec_key_t)__parsec_id;
}
static char *__jdf2c_key_fns_WriteBack_key_print(char *buffer, size_t buffer_size, parsec_key_t __parsec_key_, void *user_data)
{
  uint64_t __parsec_key = (uint64_t)(uintptr_t)__parsec_key_;
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)user_data;
  int __jdf2c_x_min = 0;
  int x = (__parsec_key) % __parsec_tp->WriteBack_x_range + __jdf2c_x_min;
  __parsec_key = __parsec_key / __parsec_tp->WriteBack_x_range;
  int __jdf2c_y_min = 0;
  int y = (__parsec_key) % __parsec_tp->WriteBack_y_range + __jdf2c_y_min;
  __parsec_key = __parsec_key / __parsec_tp->WriteBack_y_range;
  int __jdf2c_z_min = 0;
  int z = (__parsec_key) % __parsec_tp->WriteBack_z_range + __jdf2c_z_min;
  __parsec_key = __parsec_key / __parsec_tp->WriteBack_z_range;
  int __jdf2c_c_min = 0;
  int c = (__parsec_key) % __parsec_tp->WriteBack_c_range + __jdf2c_c_min;
  __parsec_key = __parsec_key / __parsec_tp->WriteBack_c_range;
  int __jdf2c_d_min = 0;
  int d = (__parsec_key) % __parsec_tp->WriteBack_d_range + __jdf2c_d_min;
  __parsec_key = __parsec_key / __parsec_tp->WriteBack_d_range;
  snprintf(buffer, buffer_size, "WriteBack(%d, %d, %d, %d, %d)", x, y, z, c, d);
  return buffer;
}

static parsec_key_fn_t __jdf2c_key_fns_WriteBack = {
   .key_equal = parsec_hash_table_generic_64bits_key_equal,
   .key_print = __jdf2c_key_fns_WriteBack_key_print,
   .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

static inline parsec_key_t __jdf2c_make_key_Exchange(const parsec_taskpool_t *tp, const parsec_assignment_t *as)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t *)tp;
  __parsec_LBM_Exchange_parsec_assignment_t ascopy, *assignment = &ascopy;
  uintptr_t __parsec_id = 0;
  memcpy(assignment, as, sizeof(__parsec_LBM_Exchange_parsec_assignment_t));
  const int x = assignment->x.value;
  int __jdf2c_x_min = 0;
  const int y = assignment->y.value;
  int __jdf2c_y_min = 0;
  const int z = assignment->z.value;
  int __jdf2c_z_min = 0;
  const int s = assignment->s.value;
  int __jdf2c_s_min = 0;
  const int conservative = assignment->conservative.value;
  int __jdf2c_conservative_min = 0;
  const int direction = assignment->direction.value;
  int __jdf2c_direction_min = 0;
  const int dimension = assignment->dimension.value;
  int __jdf2c_dimension_min = 0;
  const int side = assignment->side.value;
  int __jdf2c_side_min = 0;
  __parsec_id += (x - __jdf2c_x_min);
  __parsec_id += (y - __jdf2c_y_min) * __parsec_tp->Exchange_x_range;
  __parsec_id += (z - __jdf2c_z_min) * __parsec_tp->Exchange_x_range * __parsec_tp->Exchange_y_range;
  __parsec_id += (s - __jdf2c_s_min) * __parsec_tp->Exchange_x_range * __parsec_tp->Exchange_y_range * __parsec_tp->Exchange_z_range;
  __parsec_id += (conservative - __jdf2c_conservative_min) * __parsec_tp->Exchange_x_range * __parsec_tp->Exchange_y_range * __parsec_tp->Exchange_z_range * __parsec_tp->Exchange_s_range;
  __parsec_id += (direction - __jdf2c_direction_min) * __parsec_tp->Exchange_x_range * __parsec_tp->Exchange_y_range * __parsec_tp->Exchange_z_range * __parsec_tp->Exchange_s_range * __parsec_tp->Exchange_conservative_range;
  __parsec_id += (dimension - __jdf2c_dimension_min) * __parsec_tp->Exchange_x_range * __parsec_tp->Exchange_y_range * __parsec_tp->Exchange_z_range * __parsec_tp->Exchange_s_range * __parsec_tp->Exchange_conservative_range * __parsec_tp->Exchange_direction_range;
  __parsec_id += (side - __jdf2c_side_min) * __parsec_tp->Exchange_x_range * __parsec_tp->Exchange_y_range * __parsec_tp->Exchange_z_range * __parsec_tp->Exchange_s_range * __parsec_tp->Exchange_conservative_range * __parsec_tp->Exchange_direction_range * __parsec_tp->Exchange_dimension_range;
  (void)__parsec_tp;
  return (parsec_key_t)__parsec_id;
}
static char *__jdf2c_key_fns_Exchange_key_print(char *buffer, size_t buffer_size, parsec_key_t __parsec_key_, void *user_data)
{
  uint64_t __parsec_key = (uint64_t)(uintptr_t)__parsec_key_;
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)user_data;
  int __jdf2c_x_min = 0;
  int x = (__parsec_key) % __parsec_tp->Exchange_x_range + __jdf2c_x_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_x_range;
  int __jdf2c_y_min = 0;
  int y = (__parsec_key) % __parsec_tp->Exchange_y_range + __jdf2c_y_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_y_range;
  int __jdf2c_z_min = 0;
  int z = (__parsec_key) % __parsec_tp->Exchange_z_range + __jdf2c_z_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_z_range;
  int __jdf2c_s_min = 0;
  int s = (__parsec_key) % __parsec_tp->Exchange_s_range + __jdf2c_s_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_s_range;
  int __jdf2c_conservative_min = 0;
  int conservative = (__parsec_key) % __parsec_tp->Exchange_conservative_range + __jdf2c_conservative_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_conservative_range;
  int __jdf2c_direction_min = 0;
  int direction = (__parsec_key) % __parsec_tp->Exchange_direction_range + __jdf2c_direction_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_direction_range;
  int __jdf2c_dimension_min = 0;
  int dimension = (__parsec_key) % __parsec_tp->Exchange_dimension_range + __jdf2c_dimension_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_dimension_range;
  int __jdf2c_side_min = 0;
  int side = (__parsec_key) % __parsec_tp->Exchange_side_range + __jdf2c_side_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_side_range;
  snprintf(buffer, buffer_size, "Exchange(%d, %d, %d, %d, %d, %d, %d, %d)", x, y, z, s, conservative, direction, dimension, side);
  return buffer;
}

static parsec_key_fn_t __jdf2c_key_fns_Exchange = {
   .key_equal = parsec_hash_table_generic_64bits_key_equal,
   .key_print = __jdf2c_key_fns_Exchange_key_print,
   .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

static inline parsec_key_t __jdf2c_make_key_LBM_STEP(const parsec_taskpool_t *tp, const parsec_assignment_t *as)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t *)tp;
  __parsec_LBM_LBM_STEP_parsec_assignment_t ascopy, *assignment = &ascopy;
  uintptr_t __parsec_id = 0;
  memcpy(assignment, as, sizeof(__parsec_LBM_LBM_STEP_parsec_assignment_t));
  const int x = assignment->x.value;
  int __jdf2c_x_min = 0;
  const int y = assignment->y.value;
  int __jdf2c_y_min = 0;
  const int z = assignment->z.value;
  int __jdf2c_z_min = 0;
  const int s = assignment->s.value;
  int __jdf2c_s_min = 0;
  __parsec_id += (x - __jdf2c_x_min);
  __parsec_id += (y - __jdf2c_y_min) * __parsec_tp->LBM_STEP_x_range;
  __parsec_id += (z - __jdf2c_z_min) * __parsec_tp->LBM_STEP_x_range * __parsec_tp->LBM_STEP_y_range;
  __parsec_id += (s - __jdf2c_s_min) * __parsec_tp->LBM_STEP_x_range * __parsec_tp->LBM_STEP_y_range * __parsec_tp->LBM_STEP_z_range;
  (void)__parsec_tp;
  return (parsec_key_t)__parsec_id;
}
static char *__jdf2c_key_fns_LBM_STEP_key_print(char *buffer, size_t buffer_size, parsec_key_t __parsec_key_, void *user_data)
{
  uint64_t __parsec_key = (uint64_t)(uintptr_t)__parsec_key_;
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)user_data;
  int __jdf2c_x_min = 0;
  int x = (__parsec_key) % __parsec_tp->LBM_STEP_x_range + __jdf2c_x_min;
  __parsec_key = __parsec_key / __parsec_tp->LBM_STEP_x_range;
  int __jdf2c_y_min = 0;
  int y = (__parsec_key) % __parsec_tp->LBM_STEP_y_range + __jdf2c_y_min;
  __parsec_key = __parsec_key / __parsec_tp->LBM_STEP_y_range;
  int __jdf2c_z_min = 0;
  int z = (__parsec_key) % __parsec_tp->LBM_STEP_z_range + __jdf2c_z_min;
  __parsec_key = __parsec_key / __parsec_tp->LBM_STEP_z_range;
  int __jdf2c_s_min = 0;
  int s = (__parsec_key) % __parsec_tp->LBM_STEP_s_range + __jdf2c_s_min;
  __parsec_key = __parsec_key / __parsec_tp->LBM_STEP_s_range;
  snprintf(buffer, buffer_size, "LBM_STEP(%d, %d, %d, %d)", x, y, z, s);
  return buffer;
}

static parsec_key_fn_t __jdf2c_key_fns_LBM_STEP = {
   .key_equal = parsec_hash_table_generic_64bits_key_equal,
   .key_print = __jdf2c_key_fns_LBM_STEP_key_print,
   .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

static inline parsec_key_t __jdf2c_make_key_FillGrid(const parsec_taskpool_t *tp, const parsec_assignment_t *as)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t *)tp;
  __parsec_LBM_FillGrid_parsec_assignment_t ascopy, *assignment = &ascopy;
  uintptr_t __parsec_id = 0;
  memcpy(assignment, as, sizeof(__parsec_LBM_FillGrid_parsec_assignment_t));
  const int x = assignment->x.value;
  int __jdf2c_x_min = 0;
  const int y = assignment->y.value;
  int __jdf2c_y_min = 0;
  const int z = assignment->z.value;
  int __jdf2c_z_min = 0;
  const int c = assignment->c.value;
  int __jdf2c_c_min = 0;
  const int d = assignment->d.value;
  int __jdf2c_d_min = 0;
  __parsec_id += (x - __jdf2c_x_min);
  __parsec_id += (y - __jdf2c_y_min) * __parsec_tp->FillGrid_x_range;
  __parsec_id += (z - __jdf2c_z_min) * __parsec_tp->FillGrid_x_range * __parsec_tp->FillGrid_y_range;
  __parsec_id += (c - __jdf2c_c_min) * __parsec_tp->FillGrid_x_range * __parsec_tp->FillGrid_y_range * __parsec_tp->FillGrid_z_range;
  __parsec_id += (d - __jdf2c_d_min) * __parsec_tp->FillGrid_x_range * __parsec_tp->FillGrid_y_range * __parsec_tp->FillGrid_z_range * __parsec_tp->FillGrid_c_range;
  (void)__parsec_tp;
  return (parsec_key_t)__parsec_id;
}
static char *__jdf2c_key_fns_FillGrid_key_print(char *buffer, size_t buffer_size, parsec_key_t __parsec_key_, void *user_data)
{
  uint64_t __parsec_key = (uint64_t)(uintptr_t)__parsec_key_;
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)user_data;
  int __jdf2c_x_min = 0;
  int x = (__parsec_key) % __parsec_tp->FillGrid_x_range + __jdf2c_x_min;
  __parsec_key = __parsec_key / __parsec_tp->FillGrid_x_range;
  int __jdf2c_y_min = 0;
  int y = (__parsec_key) % __parsec_tp->FillGrid_y_range + __jdf2c_y_min;
  __parsec_key = __parsec_key / __parsec_tp->FillGrid_y_range;
  int __jdf2c_z_min = 0;
  int z = (__parsec_key) % __parsec_tp->FillGrid_z_range + __jdf2c_z_min;
  __parsec_key = __parsec_key / __parsec_tp->FillGrid_z_range;
  int __jdf2c_c_min = 0;
  int c = (__parsec_key) % __parsec_tp->FillGrid_c_range + __jdf2c_c_min;
  __parsec_key = __parsec_key / __parsec_tp->FillGrid_c_range;
  int __jdf2c_d_min = 0;
  int d = (__parsec_key) % __parsec_tp->FillGrid_d_range + __jdf2c_d_min;
  __parsec_key = __parsec_key / __parsec_tp->FillGrid_d_range;
  snprintf(buffer, buffer_size, "FillGrid(%d, %d, %d, %d, %d)", x, y, z, c, d);
  return buffer;
}

static parsec_key_fn_t __jdf2c_key_fns_FillGrid = {
   .key_equal = parsec_hash_table_generic_64bits_key_equal,
   .key_print = __jdf2c_key_fns_FillGrid_key_print,
   .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

/******                                  WriteBack                                    ******/

static inline int32_t minexpr_of_symb_LBM_WriteBack_x_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_WriteBack_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_WriteBack_x = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_WriteBack_x_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_WriteBack_x_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_WriteBack_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return (subgrid_number_x - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_WriteBack_x = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_WriteBack_x_fct }
                   }
};
static const parsec_symbol_t symb_LBM_WriteBack_x = { .name = "x", .context_index = 0, .min = &minexpr_of_symb_LBM_WriteBack_x, .max = &maxexpr_of_symb_LBM_WriteBack_x, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_WriteBack_y_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_WriteBack_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_WriteBack_y = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_WriteBack_y_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_WriteBack_y_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_WriteBack_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return (subgrid_number_y - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_WriteBack_y = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_WriteBack_y_fct }
                   }
};
static const parsec_symbol_t symb_LBM_WriteBack_y = { .name = "y", .context_index = 1, .min = &minexpr_of_symb_LBM_WriteBack_y, .max = &maxexpr_of_symb_LBM_WriteBack_y, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_WriteBack_z_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_WriteBack_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_WriteBack_z = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_WriteBack_z_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_WriteBack_z_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_WriteBack_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return (subgrid_number_z - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_WriteBack_z = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_WriteBack_z_fct }
                   }
};
static const parsec_symbol_t symb_LBM_WriteBack_z = { .name = "z", .context_index = 2, .min = &minexpr_of_symb_LBM_WriteBack_z, .max = &maxexpr_of_symb_LBM_WriteBack_z, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_WriteBack_c_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_WriteBack_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_WriteBack_c = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_WriteBack_c_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_WriteBack_c_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_WriteBack_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return (conservatives_number - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_WriteBack_c = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_WriteBack_c_fct }
                   }
};
static const parsec_symbol_t symb_LBM_WriteBack_c = { .name = "c", .context_index = 3, .min = &minexpr_of_symb_LBM_WriteBack_c, .max = &maxexpr_of_symb_LBM_WriteBack_c, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_WriteBack_d_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_WriteBack_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_WriteBack_d = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_WriteBack_d_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_WriteBack_d_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_WriteBack_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return (directions_number - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_WriteBack_d = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_WriteBack_d_fct }
                   }
};
static const parsec_symbol_t symb_LBM_WriteBack_d = { .name = "d", .context_index = 4, .min = &minexpr_of_symb_LBM_WriteBack_d, .max = &maxexpr_of_symb_LBM_WriteBack_d, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int affinity_of_LBM_WriteBack(__parsec_LBM_WriteBack_task_t *this_task,
                     parsec_data_ref_t *ref)
{
    const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t*)this_task->taskpool;
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int c = this_task->locals.c.value; (void)c;
  const int d = this_task->locals.d.value; (void)d;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  ref->dc = (parsec_data_collection_t *)__parsec_tp->super._g_descGridDC;
  /* Compute data key */
  ref->key = ref->dc->data_key(ref->dc, x, y, z, c, d);
  return 1;
}
static const parsec_property_t properties_of_LBM_WriteBack[1] = {
  {.name = NULL, .expr = NULL}
};
static inline int expr_of_cond_for_flow_of_LBM_WriteBack_for_FINAL_GRID_dep1_atline_421_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_WriteBack_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return ((c) == (0) && (d) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_WriteBack_for_FINAL_GRID_dep1_atline_421 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_WriteBack_for_FINAL_GRID_dep1_atline_421_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_WriteBack_for_FINAL_GRID_dep1_atline_421 = {
  .cond = &expr_of_cond_for_flow_of_LBM_WriteBack_for_FINAL_GRID_dep1_atline_421,  /* ((c) == (0) && (d) == (0)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_WriteBack_for_FINAL_GRID,
};
static parsec_data_t *flow_of_LBM_WriteBack_for_FINAL_GRID_dep2_atline_423_direct_access(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_WriteBack_parsec_assignment_t *assignments)
{
  const int x = assignments->x.value; (void)x;
  const int y = assignments->y.value; (void)y;
  const int z = assignments->z.value; (void)z;
  const int c = assignments->c.value; (void)c;
  const int d = assignments->d.value; (void)d;

  /* Silence Warnings: should look into parameters to know what variables are useful */
  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  if( __parsec_tp->super.super.context->my_rank == (int32_t)rank_of_descGridDC(x, y, z, c, d) )
    return data_of_descGridDC(x, y, z, c, d);
  return NULL;
}

static const parsec_dep_t flow_of_LBM_WriteBack_for_FINAL_GRID_dep2_atline_423 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .task_class_id = PARSEC_LOCAL_DATA_TASK_CLASS_ID, /* LBM_descGridDC */
  .direct_data = (parsec_data_lookup_func_t)&flow_of_LBM_WriteBack_for_FINAL_GRID_dep2_atline_423_direct_access,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_WriteBack_for_FINAL_GRID,
};
#if MAX_DEP_IN_COUNT < 1  /* number of input dependencies */
    #error Too many input dependencies (supports up to MAX_DEP_IN_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_IN_COUNT.
#endif
#if MAX_DEP_OUT_COUNT < 1  /* number of output dependencies */
    #error Too many output dependencies (supports up to MAX_DEP_OUT_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_OUT_COUNT.
#endif

static const parsec_flow_t flow_of_LBM_WriteBack_for_FINAL_GRID = {
  .name               = "FINAL_GRID",
  .sym_type           = PARSEC_SYM_INOUT,
  .flow_flags         = PARSEC_FLOW_ACCESS_RW,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_LBM_WriteBack_for_FINAL_GRID_dep1_atline_421 },
  .dep_out    = { &flow_of_LBM_WriteBack_for_FINAL_GRID_dep2_atline_423 }
};

static void
iterate_predecessors_of_LBM_WriteBack(parsec_execution_stream_t *es, const __parsec_LBM_WriteBack_task_t *this_task,
               uint32_t action_mask, parsec_ontask_function_t *ontask, void *ontask_arg)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t*)this_task->taskpool;
  parsec_task_t nc;  /* generic placeholder for locals */
  parsec_dep_data_description_t data;
  __parsec_LBM_WriteBack_parsec_assignment_t __jdf2c__tmp_locals = *(__parsec_LBM_WriteBack_parsec_assignment_t*)&this_task->locals;   /* copy of this_task locals in R/W mode to manage local definitions */
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int x = __jdf2c__tmp_locals.x.value; (void)x;
  const int y = __jdf2c__tmp_locals.y.value; (void)y;
  const int z = __jdf2c__tmp_locals.z.value; (void)z;
  const int c = __jdf2c__tmp_locals.c.value; (void)c;
  const int d = __jdf2c__tmp_locals.d.value; (void)d;
  (void)rank_src; (void)rank_dst; (void)__parsec_tp; (void)vpid_dst;
   data_repo_t *successor_repo; parsec_key_t successor_repo_key;  (void)x;  (void)y;  (void)z;  (void)c;  (void)d;
  nc.taskpool  = this_task->taskpool;
  nc.priority  = this_task->priority;
  nc.chore_mask  = PARSEC_DEV_ALL;
#if defined(DISTRIBUTED)
  rank_src = rank_of_descGridDC(x, y, z, c, d);
#endif
  if( action_mask & 0x1 ) {  /* Flow of data FINAL_GRID [0] */
    data.data   = this_task->data._f_FINAL_GRID.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
  if( action_mask & 0x1 ) {
        if( ((c) == (0) && (d) == (0)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = (number_of_steps - 1);
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "FINAL_GRID", this_task, "GRID_CD_0_0", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_WriteBack_for_FINAL_GRID_dep1_atline_421, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
  }
  (void)data;(void)nc;(void)es;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_LBM_WriteBack(parsec_execution_stream_t *es, __parsec_LBM_WriteBack_task_t *this_task, uint32_t action_mask, parsec_remote_deps_t *deps)
{
PARSEC_PINS(es, RELEASE_DEPS_BEGIN, (parsec_task_t *)this_task);{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  parsec_release_dep_fct_arg_t arg;
  int __vp_id;
  int consume_local_repo = 0;
  arg.action_mask = action_mask;
  arg.output_entry = NULL;
  arg.output_repo = NULL;
#if defined(DISTRIBUTED)
  arg.remote_deps = deps;
#endif  /* defined(DISTRIBUTED) */
  assert(NULL != es);
  arg.ready_lists = alloca(sizeof(parsec_task_t *) * es->virtual_process->parsec_context->nb_vp);
  for( __vp_id = 0; __vp_id < es->virtual_process->parsec_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );
  (void)__parsec_tp; (void)deps;
  if( action_mask & PARSEC_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data._f_FINAL_GRID.source_repo_entry ) {
        data_repo_entry_used_once( this_task->data._f_FINAL_GRID.source_repo, this_task->data._f_FINAL_GRID.source_repo_entry->ht_item.key );
    }
  }
  consume_local_repo = (this_task->repo_entry != NULL);
  /* No successors, don't call iterate_successors and don't release any local deps */
      if (consume_local_repo) {
         data_repo_entry_used_once( WriteBack_repo, this_task->repo_entry->ht_item.key );
      }
  if( action_mask & PARSEC_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data._f_FINAL_GRID.data_in ) {
        PARSEC_DATA_COPY_RELEASE(this_task->data._f_FINAL_GRID.data_in);
    }
  }
PARSEC_PINS(es, RELEASE_DEPS_END, (parsec_task_t *)this_task);}
  return 0;
}

static int data_lookup_of_LBM_WriteBack(parsec_execution_stream_t *es, __parsec_LBM_WriteBack_task_t *this_task)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  parsec_assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__parsec_tp; (void)generic_locals; (void)es;
  parsec_data_copy_t *chunk = NULL;
  data_repo_t        *reshape_repo  = NULL, *consumed_repo  = NULL;
  data_repo_entry_t  *reshape_entry = NULL, *consumed_entry = NULL;
  parsec_key_t        reshape_entry_key = 0, consumed_entry_key = 0;
  uint8_t             consumed_flow_index;
  parsec_dep_data_description_t data;
  int ret;
  (void)reshape_repo; (void)reshape_entry; (void)reshape_entry_key;
  (void)consumed_repo; (void)consumed_entry; (void)consumed_entry_key;
  (void)consumed_flow_index;
  (void)chunk; (void)data; (void)ret;
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int c = this_task->locals.c.value; (void)c;
  const int d = this_task->locals.d.value; (void)d;
  if( NULL == this_task->repo_entry ){
    this_task->repo_entry = data_repo_lookup_entry_and_create(es, WriteBack_repo,                                       __jdf2c_make_key_WriteBack((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals));
    data_repo_entry_addto_usage_limit(WriteBack_repo, this_task->repo_entry->ht_item.key, 1);    this_task->repo_entry ->generator = (void*)this_task;  /* for AYU */
#if defined(PARSEC_SIM)
    assert(this_task->repo_entry ->sim_exec_date == 0);
    this_task->repo_entry ->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  /* The reshape repo is the current task repo. */  reshape_repo = WriteBack_repo;
  reshape_entry_key = __jdf2c_make_key_WriteBack((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals) ;
  reshape_entry = this_task->repo_entry;
  /* Lookup the input data, and store them in the context if any */

if(! this_task->data._f_FINAL_GRID.fulfill ){ /* Flow FINAL_GRID */
  consumed_repo = NULL;
  consumed_entry_key = 0;
  consumed_entry = NULL;
  chunk = NULL;

    this_task->data._f_FINAL_GRID.data_out = NULL;  /* By default, if nothing matches */
    if( ((c) == (0) && (d) == (0)) ) {
    /* Flow FINAL_GRID [0] dependency [0] from predecessor LBM_STEP */
    if( NULL == (chunk = this_task->data._f_FINAL_GRID.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_LBM_STEP_parsec_assignment_t *target_locals = (__parsec_LBM_LBM_STEP_parsec_assignment_t*)&generic_locals;
      const int LBM_STEPx = target_locals->x.value = x; (void)LBM_STEPx;
      const int LBM_STEPy = target_locals->y.value = y; (void)LBM_STEPy;
      const int LBM_STEPz = target_locals->z.value = z; (void)LBM_STEPz;
      const int LBM_STEPs = target_locals->s.value = (number_of_steps - 1); (void)LBM_STEPs;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 0;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = LBM_STEP_repo;
          consumed_entry_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "FINAL_GRID", &LBM_LBM_STEP, "GRID_CD_0_0", target_locals, chunk);
      this_task->data._f_FINAL_GRID.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_FINAL_GRID.source_repo;
      consumed_entry = this_task->data._f_FINAL_GRID.source_repo_entry;
      consumed_entry_key = this_task->data._f_FINAL_GRID.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 0;
          assert( (this_task->data._f_FINAL_GRID.source_repo == reshape_repo)
               && (this_task->data._f_FINAL_GRID.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_FINAL_GRID.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_WriteBack_for_FINAL_GRID, 0);
#endif
    }
    }
    this_task->data._f_FINAL_GRID.data_in     = chunk;
    this_task->data._f_FINAL_GRID.source_repo       = consumed_repo;
    this_task->data._f_FINAL_GRID.source_repo_entry = consumed_entry;
    this_task->data._f_FINAL_GRID.fulfill = 1;
}

  /** Generate profiling information */
#if defined(PARSEC_PROF_TRACE)
  this_task->prof_info.desc = (parsec_data_collection_t*)__parsec_tp->super._g_descGridDC;
  this_task->prof_info.priority = this_task->priority;
  this_task->prof_info.data_id   = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->data_key((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, x, y, z, c, d);
  this_task->prof_info.task_class_id = this_task->task_class->task_class_id;
  this_task->prof_info.task_return_code = -1;
#endif  /* defined(PARSEC_PROF_TRACE) */
  return PARSEC_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_LBM_WriteBack(parsec_execution_stream_t *es, const __parsec_LBM_WriteBack_task_t *this_task,
              uint32_t* flow_mask, parsec_dep_data_description_t* data)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  (void)__parsec_tp; (void)es; (void)this_task; (void)data;
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int c = this_task->locals.c.value; (void)c;
  const int d = this_task->locals.d.value; (void)d;
  data->local.arena = data->remote.arena = NULL;
  data->local.src_datatype  = data->local.dst_datatype = PARSEC_DATATYPE_NULL;
  data->local.src_count     = data->local.dst_count = 0;
  data->local.src_displ     = data->local.dst_displ = 0;
  data->data_future  = NULL;
  if( (*flow_mask) & 0x80000000U ) { /* these are the input dependencies remote datatypes  */
if( (*flow_mask) & 0x1U ) {  /* Flow FINAL_GRID */
    if( ((*flow_mask) & 0x1U)
 && (((c) == (0) && (d) == (0))) ) {
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x1U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x1U) */
    goto no_mask_match;
  }

  /* these are the output dependencies remote datatypes */
if( (*flow_mask) & 0x1U ) {  /* Flow FINAL_GRID */
    if( ((*flow_mask) & 0x1U) ) {  /* Have unconditional! */
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x1U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x1U) */
 no_mask_match:
  data->data                             = NULL;
  data->local.arena = data->remote.arena = NULL;
  data->local.src_datatype  = data->local.dst_datatype = PARSEC_DATATYPE_NULL;
  data->remote.src_datatype = data->remote.dst_datatype = PARSEC_DATATYPE_NULL;
  data->local.src_count     = data->local.dst_count = 0;
  data->remote.src_count    = data->remote.dst_count = 0;
  data->local.src_displ     = data->local.dst_displ = 0;
  data->remote.src_displ    = data->remote.dst_displ = 0;
  data->data_future  = NULL;
  (*flow_mask) = 0;  /* nothing left */
  (void)x;  (void)y;  (void)z;  (void)c;  (void)d;
  return PARSEC_HOOK_RETURN_DONE;
}
#if defined(PARSEC_HAVE_CUDA)
struct parsec_body_cuda_LBM_WriteBack_s {
  uint8_t      index;
  cudaStream_t stream;
  void*           dyld_fn;
};

static int cuda_kernel_submit_LBM_WriteBack(parsec_device_gpu_module_t  *gpu_device,
                                    parsec_gpu_task_t           *gpu_task,
                                    parsec_gpu_exec_stream_t    *gpu_stream )
{
  __parsec_LBM_WriteBack_task_t *this_task = (__parsec_LBM_WriteBack_task_t *)gpu_task->ec;
  parsec_device_cuda_module_t *cuda_device = (parsec_device_cuda_module_t*)gpu_device;
  parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t*)gpu_stream;
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  struct parsec_body_cuda_LBM_WriteBack_s parsec_body = { cuda_device->cuda_index, cuda_stream->cuda_stream, NULL };
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int c = this_task->locals.c.value; (void)c;
  const int d = this_task->locals.d.value; (void)d;
  (void)x;  (void)y;  (void)z;  (void)c;  (void)d;

  (void)gpu_device; (void)gpu_stream; (void)__parsec_tp; (void)parsec_body; (void)cuda_device; (void)cuda_stream;
  /** Declare the variables that will hold the data, and all the accounting for each */
    parsec_data_copy_t *_f_FINAL_GRID = this_task->data._f_FINAL_GRID.data_out;
    void *FINAL_GRID = PARSEC_DATA_COPY_GET_PTR(_f_FINAL_GRID); (void)FINAL_GRID;

  /** Update starting simulation date */
#if defined(PARSEC_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eFINAL_GRID = this_task->data._f_FINAL_GRID.source_repo_entry;
    if( (NULL != eFINAL_GRID) && (eFINAL_GRID->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eFINAL_GRID->sim_exec_date;
    if( this_task->task_class->sim_cost_fct != NULL ) {
      this_task->sim_exec_date += this_task->task_class->sim_cost_fct(this_task);
    }
    if( es->largest_simulation_date < this_task->sim_exec_date )
      es->largest_simulation_date = this_task->sim_exec_date;
  }
#endif
  /** Cache Awareness Accounting */
#if defined(PARSEC_CACHE_AWARENESS)
  cache_buf_referenced(es->closest_cache, FINAL_GRID);
#endif /* PARSEC_CACHE_AWARENESS */
#if defined(PARSEC_DEBUG_NOISIER)
  {
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream, "GPU[%s]:\tEnqueue on device %s priority %d", gpu_device->super.name, 
           parsec_task_snprintf(tmp, MAX_TASK_STRLEN, (parsec_task_t *)this_task),
           this_task->priority );
  }
#endif /* defined(PARSEC_DEBUG_NOISIER) */


#if !defined(PARSEC_PROF_DRY_BODY)

/*-----                                WriteBack BODY                                -----*/

#if defined(PARSEC_PROF_TRACE)
  if(gpu_stream->prof_event_track_enable) {
    PARSEC_TASK_PROF_TRACE(gpu_stream->profiling,
                           PARSEC_PROF_FUNC_KEY_START(this_task->taskpool,
                                     this_task->task_class->task_class_id),
                           (parsec_task_t*)this_task);
    gpu_task->prof_key_end = PARSEC_PROF_FUNC_KEY_END(this_task->taskpool,
                                   this_task->task_class->task_class_id);
    gpu_task->prof_event_id = this_task->task_class->key_functions->
           key_hash(this_task->task_class->make_key(this_task->taskpool, ((parsec_task_t*)this_task)->locals), NULL);
    gpu_task->prof_tp_id = this_task->taskpool->taskpool_id;
  }
#endif /* PARSEC_PROF_TRACE */
#line 426 "LBM.jdf"
    printf("[Process %d] kernel WRITE_BACK (%d %d %d %d %d)\n", rank, x, y, z, c, d);

#line 1274 "LBM.c"
/*-----                            END OF WriteBack BODY                              -----*/



#endif /*!defined(PARSEC_PROF_DRY_BODY)*/

  return PARSEC_HOOK_RETURN_DONE;
}

static int hook_of_LBM_WriteBack_CUDA(parsec_execution_stream_t *es, __parsec_LBM_WriteBack_task_t *this_task)
{
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  parsec_gpu_task_t *gpu_task;
  double ratio;
  int dev_index;
    const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int c = this_task->locals.c.value; (void)c;
  const int d = this_task->locals.d.value; (void)d;
  (void)x;  (void)y;  (void)z;  (void)c;  (void)d;

  (void)es; (void)__parsec_tp;

  ratio = 1.;
  dev_index = parsec_get_best_device((parsec_task_t*)this_task, ratio);
  assert(dev_index >= 0);
  if( dev_index < 2 ) {
    return PARSEC_HOOK_RETURN_NEXT;  /* Fall back */
  }

  gpu_task = (parsec_gpu_task_t*)calloc(1, sizeof(parsec_gpu_task_t));
  PARSEC_OBJ_CONSTRUCT(gpu_task, parsec_list_item_t);
  gpu_task->ec = (parsec_task_t*)this_task;
  gpu_task->submit = &cuda_kernel_submit_LBM_WriteBack;
  gpu_task->task_type = 0;
  gpu_task->load = ratio * parsec_device_sweight[dev_index];
  gpu_task->last_data_check_epoch = -1;  /* force at least one validation for the task */
  gpu_task->stage_in  = parsec_default_cuda_stage_in;
  gpu_task->stage_out = parsec_default_cuda_stage_out;
  gpu_task->pushout = 0;
  gpu_task->flow[0]         = &flow_of_LBM_WriteBack_for_FINAL_GRID;
  gpu_task->flow_dc[0] = NULL;
  gpu_task->flow_nb_elts[0] = gpu_task->ec->data[0].data_in->original->nb_elts;
  gpu_task->pushout |= (1 << 0);
  parsec_device_load[dev_index] += gpu_task->load;

  return parsec_cuda_kernel_scheduler( es, gpu_task, dev_index );
}

#endif  /*  defined(PARSEC_HAVE_CUDA) */
static int complete_hook_of_LBM_WriteBack(parsec_execution_stream_t *es, __parsec_LBM_WriteBack_task_t *this_task)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
#if defined(DISTRIBUTED)
    const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int c = this_task->locals.c.value; (void)c;
  const int d = this_task->locals.d.value; (void)d;
#endif  /* defined(DISTRIBUTED) */
  (void)es; (void)__parsec_tp;
parsec_data_t* data_t_desc = NULL; (void)data_t_desc;
  if ( NULL != this_task->data._f_FINAL_GRID.data_out ) {
#if defined(PARSEC_DEBUG_NOISIER)
     char tmp[128];
#endif
     this_task->data._f_FINAL_GRID.data_out->version++;  /* FINAL_GRID */
     PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                          "Complete hook of %s: change Data copy %p to version %d at %s:%d",
                          parsec_task_snprintf(tmp, 128, (parsec_task_t*)(this_task)),
                          this_task->data._f_FINAL_GRID.data_out, this_task->data._f_FINAL_GRID.data_out->version, __FILE__, __LINE__);
  }
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  data_t_desc = data_of_descGridDC(x, y, z, c, d);
  if( (NULL != this_task->data._f_FINAL_GRID.data_out) && (this_task->data._f_FINAL_GRID.data_out->original != data_t_desc) ) {
    /* Writting back using remote_type */;
    parsec_dep_data_description_t data;
    data.data         = this_task->data._f_FINAL_GRID.data_out;
    data.local.arena        = NULL;
    data.local.src_datatype = data.data->dtt;
    data.local.src_count    = 1;
    data.local.src_displ    = 0;
    data.local.dst_datatype = parsec_data_get_copy(data_t_desc, 0)->dtt;
    data.local.dst_count    = 1;
    data.local.dst_displ    = 0;
    data.data_future  = NULL;
    assert( data.local.src_count > 0 );
    parsec_remote_dep_memcpy(es,
                            this_task->taskpool,
                            parsec_data_get_copy(data_of_descGridDC(x, y, z, c, d), 0),
                            this_task->data._f_FINAL_GRID.data_out, &data);
  }
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
  parsec_prof_grapher_data_output((parsec_task_t*)this_task, data_of_descGridDC(x, y, z, c, d), &flow_of_LBM_WriteBack_for_FINAL_GRID);
#endif
  (void)x;  (void)y;  (void)z;  (void)c;  (void)d;

#endif /* DISTRIBUTED */
#if defined(PARSEC_PROF_GRAPHER)
  parsec_prof_grapher_task((parsec_task_t*)this_task, es->th_id, es->virtual_process->vp_id,
     __jdf2c_key_fns_WriteBack.key_hash(this_task->task_class->make_key( (parsec_taskpool_t*)this_task->taskpool, ((parsec_task_t*)this_task)->locals), NULL));
#endif  /* defined(PARSEC_PROF_GRAPHER) */
  release_deps_of_LBM_WriteBack(es, this_task,
      PARSEC_ACTION_RELEASE_REMOTE_DEPS |
      PARSEC_ACTION_RELEASE_LOCAL_DEPS |
      PARSEC_ACTION_RELEASE_LOCAL_REFS |
      PARSEC_ACTION_RESHAPE_ON_RELEASE |
      0x1,  /* mask of all dep_index */ 
      NULL);
  return PARSEC_HOOK_RETURN_DONE;
}

static parsec_hook_return_t release_task_of_LBM_WriteBack(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    const __parsec_LBM_internal_taskpool_t *__parsec_tp =
        (const __parsec_LBM_internal_taskpool_t *)this_task->taskpool;
    parsec_hash_table_t *ht = (parsec_hash_table_t*)__parsec_tp->super.super.dependencies_array[3];
    parsec_key_t key = this_task->task_class->make_key((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals);
    parsec_hashable_dependency_t *hash_dep = (parsec_hashable_dependency_t *)parsec_hash_table_remove(ht, key);
    parsec_thread_mempool_free(hash_dep->mempool_owner, hash_dep);
    return parsec_release_task_to_mempool_update_nbtasks(es, this_task);
}

static char *LBM_LBM_WriteBack_internal_init_deps_key_functions_key_print(char *buffer, size_t buffer_size, parsec_key_t __parsec_key_, void *user_data)
{
  uint64_t __parsec_key = (uint64_t)(uintptr_t)__parsec_key_;
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)user_data;
  int __jdf2c_x_min = 0;
  int x = (__parsec_key) % __parsec_tp->WriteBack_x_range + __jdf2c_x_min;
  __parsec_key = __parsec_key / __parsec_tp->WriteBack_x_range;
  int __jdf2c_y_min = 0;
  int y = (__parsec_key) % __parsec_tp->WriteBack_y_range + __jdf2c_y_min;
  __parsec_key = __parsec_key / __parsec_tp->WriteBack_y_range;
  int __jdf2c_z_min = 0;
  int z = (__parsec_key) % __parsec_tp->WriteBack_z_range + __jdf2c_z_min;
  __parsec_key = __parsec_key / __parsec_tp->WriteBack_z_range;
  int __jdf2c_c_min = 0;
  int c = (__parsec_key) % __parsec_tp->WriteBack_c_range + __jdf2c_c_min;
  __parsec_key = __parsec_key / __parsec_tp->WriteBack_c_range;
  int __jdf2c_d_min = 0;
  int d = (__parsec_key) % __parsec_tp->WriteBack_d_range + __jdf2c_d_min;
  __parsec_key = __parsec_key / __parsec_tp->WriteBack_d_range;
  snprintf(buffer, buffer_size, "WriteBack(%d, %d, %d, %d, %d)", x, y, z, c, d);
  return buffer;
}

static parsec_key_fn_t LBM_LBM_WriteBack_internal_init_deps_key_functions = {
   .key_equal = parsec_hash_table_generic_64bits_key_equal,
   .key_print = LBM_LBM_WriteBack_internal_init_deps_key_functions_key_print,
   .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

/* Needs: min-max count-tasks iterate */
static int LBM_WriteBack_internal_init(parsec_execution_stream_t * es, __parsec_LBM_WriteBack_task_t * this_task)
{
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t*)this_task->taskpool;
  int32_t nb_tasks = 0, saved_nb_tasks = 0;
int32_t __x_min = 0x7fffffff, __x_max = 0;int32_t __jdf2c_x_min = 0x7fffffff, __jdf2c_x_max = 0;int32_t __y_min = 0x7fffffff, __y_max = 0;int32_t __jdf2c_y_min = 0x7fffffff, __jdf2c_y_max = 0;int32_t __z_min = 0x7fffffff, __z_max = 0;int32_t __jdf2c_z_min = 0x7fffffff, __jdf2c_z_max = 0;int32_t __c_min = 0x7fffffff, __c_max = 0;int32_t __jdf2c_c_min = 0x7fffffff, __jdf2c_c_max = 0;int32_t __d_min = 0x7fffffff, __d_max = 0;int32_t __jdf2c_d_min = 0x7fffffff, __jdf2c_d_max = 0;  __parsec_LBM_WriteBack_parsec_assignment_t assignments = {  .x.value = 0, .y.value = 0, .z.value = 0, .c.value = 0, .d.value = 0 };
  int32_t  x,  y,  z,  c,  d;
  int32_t __jdf2c_x_start, __jdf2c_x_end, __jdf2c_x_inc;
  int32_t __jdf2c_y_start, __jdf2c_y_end, __jdf2c_y_inc;
  int32_t __jdf2c_z_start, __jdf2c_z_end, __jdf2c_z_inc;
  int32_t __jdf2c_c_start, __jdf2c_c_end, __jdf2c_c_inc;
  int32_t __jdf2c_d_start, __jdf2c_d_end, __jdf2c_d_inc;
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)
  PARSEC_PROFILING_TRACE(es->es_profile,
                         this_task->taskpool->profiling_array[2 * this_task->task_class->task_class_id],
                         0,
                         this_task->taskpool->taskpool_id, NULL);
#endif /* defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */
    __jdf2c_x_start = 0;
    __jdf2c_x_end = (subgrid_number_x - 1);
    __jdf2c_x_inc = 1;
    __x_min = parsec_imin(__jdf2c_x_start, __jdf2c_x_end);
    __x_max = parsec_imax(__jdf2c_x_start, __jdf2c_x_end);
    __jdf2c_x_min = parsec_imin(__jdf2c_x_min, __x_min);
    __jdf2c_x_max = parsec_imax(__jdf2c_x_max, __x_max);
    for(x =  __jdf2c_x_start;
        x <= __jdf2c_x_end;
        x += __jdf2c_x_inc) {
    assignments.x.value = x;
      __jdf2c_y_start = 0;
      __jdf2c_y_end = (subgrid_number_y - 1);
      __jdf2c_y_inc = 1;
      __y_min = parsec_imin(__jdf2c_y_start, __jdf2c_y_end);
      __y_max = parsec_imax(__jdf2c_y_start, __jdf2c_y_end);
      __jdf2c_y_min = parsec_imin(__jdf2c_y_min, __y_min);
      __jdf2c_y_max = parsec_imax(__jdf2c_y_max, __y_max);
      for(y =  __jdf2c_y_start;
          y <= __jdf2c_y_end;
          y += __jdf2c_y_inc) {
      assignments.y.value = y;
        __jdf2c_z_start = 0;
        __jdf2c_z_end = (subgrid_number_z - 1);
        __jdf2c_z_inc = 1;
        __z_min = parsec_imin(__jdf2c_z_start, __jdf2c_z_end);
        __z_max = parsec_imax(__jdf2c_z_start, __jdf2c_z_end);
        __jdf2c_z_min = parsec_imin(__jdf2c_z_min, __z_min);
        __jdf2c_z_max = parsec_imax(__jdf2c_z_max, __z_max);
        for(z =  __jdf2c_z_start;
            z <= __jdf2c_z_end;
            z += __jdf2c_z_inc) {
        assignments.z.value = z;
          __jdf2c_c_start = 0;
          __jdf2c_c_end = (conservatives_number - 1);
          __jdf2c_c_inc = 1;
          __c_min = parsec_imin(__jdf2c_c_start, __jdf2c_c_end);
          __c_max = parsec_imax(__jdf2c_c_start, __jdf2c_c_end);
          __jdf2c_c_min = parsec_imin(__jdf2c_c_min, __c_min);
          __jdf2c_c_max = parsec_imax(__jdf2c_c_max, __c_max);
          for(c =  __jdf2c_c_start;
              c <= __jdf2c_c_end;
              c += __jdf2c_c_inc) {
          assignments.c.value = c;
            __jdf2c_d_start = 0;
            __jdf2c_d_end = (directions_number - 1);
            __jdf2c_d_inc = 1;
            __d_min = parsec_imin(__jdf2c_d_start, __jdf2c_d_end);
            __d_max = parsec_imax(__jdf2c_d_start, __jdf2c_d_end);
            __jdf2c_d_min = parsec_imin(__jdf2c_d_min, __d_min);
            __jdf2c_d_max = parsec_imax(__jdf2c_d_max, __d_max);
            for(d =  __jdf2c_d_start;
                d <= __jdf2c_d_end;
                d += __jdf2c_d_inc) {
            assignments.d.value = d;
            if( !WriteBack_pred(x, y, z, c, d) ) continue;
            nb_tasks++;
          } /* Loop on normal range d */
        } /* For loop of c */ 
      } /* For loop of z */ 
    } /* For loop of y */ 
  } /* For loop of x */ 
   if( 0 != nb_tasks ) {
     (void)parsec_atomic_fetch_add_int32(&__parsec_tp->initial_number_tasks, nb_tasks);
   }
  /* Set the range variables for the collision-free hash-computation */
  __parsec_tp->WriteBack_x_range = (__jdf2c_x_max - __jdf2c_x_min) + 1;
  __parsec_tp->WriteBack_y_range = (__jdf2c_y_max - __jdf2c_y_min) + 1;
  __parsec_tp->WriteBack_z_range = (__jdf2c_z_max - __jdf2c_z_min) + 1;
  __parsec_tp->WriteBack_c_range = (__jdf2c_c_max - __jdf2c_c_min) + 1;
  __parsec_tp->WriteBack_d_range = (__jdf2c_d_max - __jdf2c_d_min) + 1;
  this_task->status = PARSEC_TASK_STATUS_COMPLETE;

  PARSEC_AYU_REGISTER_TASK(&LBM_WriteBack);
  __parsec_tp->super.super.dependencies_array[3] = PARSEC_OBJ_NEW(parsec_hash_table_t);
  parsec_hash_table_init(__parsec_tp->super.super.dependencies_array[3], offsetof(parsec_hashable_dependency_t, ht_item), 10, LBM_LBM_WriteBack_internal_init_deps_key_functions, this_task->taskpool);
  __parsec_tp->repositories[3] = data_repo_create_nothreadsafe(nb_tasks, __jdf2c_key_fns_WriteBack, (parsec_taskpool_t*)__parsec_tp, 1);
(void)saved_nb_tasks;
(void)__x_min; (void)__x_max;(void)__y_min; (void)__y_max;(void)__z_min; (void)__z_max;(void)__c_min; (void)__c_max;(void)__d_min; (void)__d_max;  (void)__jdf2c_x_start; (void)__jdf2c_x_end; (void)__jdf2c_x_inc;  (void)__jdf2c_y_start; (void)__jdf2c_y_end; (void)__jdf2c_y_inc;  (void)__jdf2c_z_start; (void)__jdf2c_z_end; (void)__jdf2c_z_inc;  (void)__jdf2c_c_start; (void)__jdf2c_c_end; (void)__jdf2c_c_inc;  (void)__jdf2c_d_start; (void)__jdf2c_d_end; (void)__jdf2c_d_inc;  (void)assignments; (void)__parsec_tp; (void)es;
  if(1 == parsec_atomic_fetch_dec_int32(&__parsec_tp->sync_point)) {
    /* Last initialization task complete. Update the number of tasks. */
    __parsec_tp->super.super.tdm.module->taskpool_addto_nb_tasks(&__parsec_tp->super.super, __parsec_tp->initial_number_tasks);
    parsec_mfence();  /* write memory barrier to guarantee that the scheduler gets the correct number of tasks */
    parsec_taskpool_enable((parsec_taskpool_t*)__parsec_tp, &__parsec_tp->startup_queue,
                           (parsec_task_t*)this_task, es, __parsec_tp->super.super.nb_pending_actions);
    __parsec_tp->super.super.tdm.module->taskpool_ready(&__parsec_tp->super.super);
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)
    PARSEC_PROFILING_TRACE(es->es_profile,
                           this_task->taskpool->profiling_array[2 * this_task->task_class->task_class_id + 1],
                           0,
                           this_task->taskpool->taskpool_id, NULL);
#endif /* defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */
    return PARSEC_HOOK_RETURN_DONE;
  }
  return PARSEC_HOOK_RETURN_DONE;
}

static const __parsec_chore_t __LBM_WriteBack_chores[] ={
#if defined(PARSEC_HAVE_CUDA)
    { .type     = PARSEC_DEV_CUDA,
      .dyld     = NULL,
      .evaluate = NULL,
      .hook     = (parsec_hook_t*)hook_of_LBM_WriteBack_CUDA },
#endif  /* defined(PARSEC_HAVE_CUDA) */
    { .type     = PARSEC_DEV_NONE,
      .evaluate = NULL,
      .hook     = (parsec_hook_t*)NULL },  /* End marker */
};

static const parsec_task_class_t LBM_WriteBack = {
  .name = "WriteBack",
  .task_class_id = 3,
  .nb_flows = 1,
  .nb_parameters = 5,
  .nb_locals = 5,
  .task_class_type = PARSEC_TASK_CLASS_TYPE_PTG,
  .params = { &symb_LBM_WriteBack_x, &symb_LBM_WriteBack_y, &symb_LBM_WriteBack_z, &symb_LBM_WriteBack_c, &symb_LBM_WriteBack_d, NULL },
  .locals = { &symb_LBM_WriteBack_x, &symb_LBM_WriteBack_y, &symb_LBM_WriteBack_z, &symb_LBM_WriteBack_c, &symb_LBM_WriteBack_d, NULL },
  .data_affinity = (parsec_data_ref_fn_t*)affinity_of_LBM_WriteBack,
  .initial_data = (parsec_data_ref_fn_t*)affinity_of_LBM_WriteBack,
  .final_data = (parsec_data_ref_fn_t*)affinity_of_LBM_WriteBack,
  .priority = NULL,
  .properties = properties_of_LBM_WriteBack,
#if MAX_PARAM_COUNT < 1  /* number of read flows of WriteBack */
  #error Too many read flows for task WriteBack
#endif  /* MAX_PARAM_COUNT */
#if MAX_PARAM_COUNT < 1  /* number of write flows of WriteBack */
  #error Too many write flows for task WriteBack
#endif  /* MAX_PARAM_COUNT */
  .in = { &flow_of_LBM_WriteBack_for_FINAL_GRID, NULL },
  .out = { &flow_of_LBM_WriteBack_for_FINAL_GRID, NULL },
  .flags = 0x0 | PARSEC_USE_DEPS_MASK,
  .dependencies_goal = 0x1,
  .make_key = __jdf2c_make_key_WriteBack,
  .task_snprintf = parsec_task_snprintf,
  .key_functions = &__jdf2c_key_fns_WriteBack,
  .fini = (parsec_hook_t*)NULL,
  .incarnations = __LBM_WriteBack_chores,
  .find_deps = parsec_hash_find_deps,
  .update_deps = parsec_update_deps_with_mask,
  .iterate_successors = (parsec_traverse_function_t*)NULL,
  .iterate_predecessors = (parsec_traverse_function_t*)iterate_predecessors_of_LBM_WriteBack,
  .release_deps = (parsec_release_deps_t*)release_deps_of_LBM_WriteBack,
  .prepare_output = (parsec_hook_t*)NULL,
  .prepare_input = (parsec_hook_t*)data_lookup_of_LBM_WriteBack,
  .get_datatype = (parsec_datatype_lookup_t*)datatype_lookup_of_LBM_WriteBack,
  .complete_execution = (parsec_hook_t*)complete_hook_of_LBM_WriteBack,
  .release_task = &release_task_of_LBM_WriteBack,
#if defined(PARSEC_SIM)
  .sim_cost_fct = (parsec_sim_cost_fct_t*)NULL,
#endif
};


/******                                    Exchange                                    ******/

static inline int32_t minexpr_of_symb_LBM_Exchange_x_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_Exchange_x = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_Exchange_x_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_Exchange_x_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (subgrid_number_x - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_Exchange_x = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_Exchange_x_fct }
                   }
};
static const parsec_symbol_t symb_LBM_Exchange_x = { .name = "x", .context_index = 0, .min = &minexpr_of_symb_LBM_Exchange_x, .max = &maxexpr_of_symb_LBM_Exchange_x, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_Exchange_y_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_Exchange_y = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_Exchange_y_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_Exchange_y_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (subgrid_number_y - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_Exchange_y = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_Exchange_y_fct }
                   }
};
static const parsec_symbol_t symb_LBM_Exchange_y = { .name = "y", .context_index = 1, .min = &minexpr_of_symb_LBM_Exchange_y, .max = &maxexpr_of_symb_LBM_Exchange_y, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_Exchange_z_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_Exchange_z = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_Exchange_z_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_Exchange_z_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (subgrid_number_z - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_Exchange_z = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_Exchange_z_fct }
                   }
};
static const parsec_symbol_t symb_LBM_Exchange_z = { .name = "z", .context_index = 2, .min = &minexpr_of_symb_LBM_Exchange_z, .max = &maxexpr_of_symb_LBM_Exchange_z, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_Exchange_s_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_Exchange_s = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_Exchange_s_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_Exchange_s_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (number_of_steps - 2);
}
static const parsec_expr_t maxexpr_of_symb_LBM_Exchange_s = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_Exchange_s_fct }
                   }
};
static const parsec_symbol_t symb_LBM_Exchange_s = { .name = "s", .context_index = 3, .min = &minexpr_of_symb_LBM_Exchange_s, .max = &maxexpr_of_symb_LBM_Exchange_s, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_Exchange_conservative_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_Exchange_conservative = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_Exchange_conservative_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_Exchange_conservative_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (conservatives_number - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_Exchange_conservative = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_Exchange_conservative_fct }
                   }
};
static const parsec_symbol_t symb_LBM_Exchange_conservative = { .name = "conservative", .context_index = 4, .min = &minexpr_of_symb_LBM_Exchange_conservative, .max = &maxexpr_of_symb_LBM_Exchange_conservative, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_Exchange_direction_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_Exchange_direction = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_Exchange_direction_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_Exchange_direction_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (directions_number - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_Exchange_direction = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_Exchange_direction_fct }
                   }
};
static const parsec_symbol_t symb_LBM_Exchange_direction = { .name = "direction", .context_index = 5, .min = &minexpr_of_symb_LBM_Exchange_direction, .max = &maxexpr_of_symb_LBM_Exchange_direction, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_Exchange_dimension_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_Exchange_dimension = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_Exchange_dimension_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_Exchange_dimension_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{


;
  (void)__parsec_tp; (void)locals;
  return (3 - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_Exchange_dimension = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_Exchange_dimension_fct }
                   }
};
static const parsec_symbol_t symb_LBM_Exchange_dimension = { .name = "dimension", .context_index = 6, .min = &minexpr_of_symb_LBM_Exchange_dimension, .max = &maxexpr_of_symb_LBM_Exchange_dimension, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_Exchange_side_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_Exchange_side = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_Exchange_side_fct }
                   }
};
static inline int32_t maxexpr_of_symb_LBM_Exchange_side_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 1;
}
static const parsec_expr_t maxexpr_of_symb_LBM_Exchange_side = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_Exchange_side_fct }
                   }
};
static const parsec_symbol_t symb_LBM_Exchange_side = { .name = "side", .context_index = 7, .min = &minexpr_of_symb_LBM_Exchange_side, .max = &maxexpr_of_symb_LBM_Exchange_side, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int affinity_of_LBM_Exchange(__parsec_LBM_Exchange_task_t *this_task,
                     parsec_data_ref_t *ref)
{
    const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t*)this_task->taskpool;
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int s = this_task->locals.s.value; (void)s;
  const int conservative = this_task->locals.conservative.value; (void)conservative;
  const int direction = this_task->locals.direction.value; (void)direction;
  const int dimension = this_task->locals.dimension.value; (void)dimension;
  const int side = this_task->locals.side.value; (void)side;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  ref->dc = (parsec_data_collection_t *)__parsec_tp->super._g_descGridDC;
  /* Compute data key */
  ref->key = ref->dc->data_key(ref->dc, x, y, z, conservative, direction);
  return 1;
}
static const parsec_property_t properties_of_LBM_Exchange[1] = {
  {.name = NULL, .expr = NULL}
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep1_atline_371_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return ((((conservative) == (0) && (direction) == (0)) && (dimension) == (0)) && (side) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep1_atline_371 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep1_atline_371_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep1_atline_371 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep1_atline_371,  /* ((((conservative) == (0) && (direction) == (0)) && (dimension) == (0)) && (side) == (0)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep2_atline_372_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (((conservative) == (0) && (direction) == (0)) && !(((dimension) == (0) && (side) == (0))));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep2_atline_372 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep2_atline_372_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep2_atline_372 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep2_atline_372,  /* (((conservative) == (0) && (direction) == (0)) && !(((dimension) == (0) && (side) == (0)))) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 1,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep3_atline_373_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return ((((conservative) == (1) && (direction) == (0)) && (dimension) == (0)) && (side) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep3_atline_373 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep3_atline_373_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep3_atline_373 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep3_atline_373,  /* ((((conservative) == (1) && (direction) == (0)) && (dimension) == (0)) && (side) == (0)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0,
  .dep_index = 2,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep4_atline_374_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (((conservative) == (1) && (direction) == (0)) && !(((dimension) == (0) && (side) == (0))));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep4_atline_374 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep4_atline_374_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep4_atline_374 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep4_atline_374,  /* (((conservative) == (1) && (direction) == (0)) && !(((dimension) == (0) && (side) == (0)))) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 3,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep5_atline_375_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return ((((conservative) == (2) && (direction) == (0)) && (dimension) == (0)) && (side) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep5_atline_375 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep5_atline_375_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep5_atline_375 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep5_atline_375,  /* ((((conservative) == (2) && (direction) == (0)) && (dimension) == (0)) && (side) == (0)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0,
  .dep_index = 4,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep6_atline_376_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (((conservative) == (2) && (direction) == (0)) && !(((dimension) == (0) && (side) == (0))));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep6_atline_376 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep6_atline_376_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep6_atline_376 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep6_atline_376,  /* (((conservative) == (2) && (direction) == (0)) && !(((dimension) == (0) && (side) == (0)))) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 5,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep7_atline_377_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return ((((conservative) == (0) && (direction) == (1)) && (dimension) == (0)) && (side) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep7_atline_377 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep7_atline_377_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep7_atline_377 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep7_atline_377,  /* ((((conservative) == (0) && (direction) == (1)) && (dimension) == (0)) && (side) == (0)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1,
  .dep_index = 6,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep8_atline_378_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (((conservative) == (0) && (direction) == (1)) && !(((dimension) == (0) && (side) == (0))));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep8_atline_378 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep8_atline_378_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep8_atline_378 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep8_atline_378,  /* (((conservative) == (0) && (direction) == (1)) && !(((dimension) == (0) && (side) == (0)))) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 7,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep9_atline_379_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return ((((conservative) == (1) && (direction) == (1)) && (dimension) == (0)) && (side) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep9_atline_379 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep9_atline_379_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep9_atline_379 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep9_atline_379,  /* ((((conservative) == (1) && (direction) == (1)) && (dimension) == (0)) && (side) == (0)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1,
  .dep_index = 8,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep10_atline_380_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (((conservative) == (1) && (direction) == (1)) && !(((dimension) == (0) && (side) == (0))));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep10_atline_380 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep10_atline_380_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep10_atline_380 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep10_atline_380,  /* (((conservative) == (1) && (direction) == (1)) && !(((dimension) == (0) && (side) == (0)))) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 9,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep11_atline_381_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return ((((conservative) == (2) && (direction) == (1)) && (dimension) == (0)) && (side) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep11_atline_381 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep11_atline_381_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep11_atline_381 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep11_atline_381,  /* ((((conservative) == (2) && (direction) == (1)) && (dimension) == (0)) && (side) == (0)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1,
  .dep_index = 10,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep12_atline_382_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (((conservative) == (2) && (direction) == (1)) && !(((dimension) == (0) && (side) == (0))));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep12_atline_382 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep12_atline_382_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep12_atline_382 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep12_atline_382,  /* (((conservative) == (2) && (direction) == (1)) && !(((dimension) == (0) && (side) == (0)))) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 11,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep13_atline_384_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return ((((conservative) == (0) && (direction) == (0)) && (dimension) == (2)) && (side) == (1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep13_atline_384 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep13_atline_384_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep13_atline_384 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep13_atline_384,  /* ((((conservative) == (0) && (direction) == (0)) && (dimension) == (2)) && (side) == (1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep14_atline_385_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (((conservative) == (0) && (direction) == (0)) && !(((dimension) == (2) && (side) == (1))));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep14_atline_385 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep14_atline_385_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep14_atline_385 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep14_atline_385,  /* (((conservative) == (0) && (direction) == (0)) && !(((dimension) == (2) && (side) == (1)))) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 1,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep15_atline_386_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return ((((conservative) == (1) && (direction) == (0)) && (dimension) == (2)) && (side) == (1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep15_atline_386 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep15_atline_386_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep15_atline_386 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep15_atline_386,  /* ((((conservative) == (1) && (direction) == (0)) && (dimension) == (2)) && (side) == (1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0,
  .dep_index = 2,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep16_atline_387_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (((conservative) == (1) && (direction) == (0)) && !(((dimension) == (2) && (side) == (1))));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep16_atline_387 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep16_atline_387_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep16_atline_387 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep16_atline_387,  /* (((conservative) == (1) && (direction) == (0)) && !(((dimension) == (2) && (side) == (1)))) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 3,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep17_atline_388_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return ((((conservative) == (2) && (direction) == (0)) && (dimension) == (2)) && (side) == (1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep17_atline_388 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep17_atline_388_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep17_atline_388 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep17_atline_388,  /* ((((conservative) == (2) && (direction) == (0)) && (dimension) == (2)) && (side) == (1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0,
  .dep_index = 4,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep18_atline_389_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (((conservative) == (2) && (direction) == (0)) && !(((dimension) == (2) && (side) == (1))));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep18_atline_389 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep18_atline_389_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep18_atline_389 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep18_atline_389,  /* (((conservative) == (2) && (direction) == (0)) && !(((dimension) == (2) && (side) == (1)))) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 5,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep19_atline_390_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return ((((conservative) == (0) && (direction) == (1)) && (dimension) == (2)) && (side) == (1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep19_atline_390 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep19_atline_390_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep19_atline_390 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep19_atline_390,  /* ((((conservative) == (0) && (direction) == (1)) && (dimension) == (2)) && (side) == (1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1,
  .dep_index = 6,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep20_atline_391_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (((conservative) == (0) && (direction) == (1)) && !(((dimension) == (2) && (side) == (1))));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep20_atline_391 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep20_atline_391_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep20_atline_391 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep20_atline_391,  /* (((conservative) == (0) && (direction) == (1)) && !(((dimension) == (2) && (side) == (1)))) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 7,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep21_atline_392_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return ((((conservative) == (1) && (direction) == (1)) && (dimension) == (2)) && (side) == (1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep21_atline_392 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep21_atline_392_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep21_atline_392 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep21_atline_392,  /* ((((conservative) == (1) && (direction) == (1)) && (dimension) == (2)) && (side) == (1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1,
  .dep_index = 8,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep22_atline_393_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (((conservative) == (1) && (direction) == (1)) && !(((dimension) == (2) && (side) == (1))));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep22_atline_393 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep22_atline_393_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep22_atline_393 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep22_atline_393,  /* (((conservative) == (1) && (direction) == (1)) && !(((dimension) == (2) && (side) == (1)))) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 9,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep23_atline_394_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return ((((conservative) == (2) && (direction) == (1)) && (dimension) == (2)) && (side) == (1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep23_atline_394 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep23_atline_394_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep23_atline_394 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep23_atline_394,  /* ((((conservative) == (2) && (direction) == (1)) && (dimension) == (2)) && (side) == (1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1,
  .dep_index = 10,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
static inline int expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep24_atline_395_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_Exchange_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;
  const int conservative = locals->conservative.value; (void)conservative;
  const int direction = locals->direction.value; (void)direction;
  const int dimension = locals->dimension.value; (void)dimension;
  const int side = locals->side.value; (void)side;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)conservative;
  (void)direction;
  (void)dimension;
  (void)side;
  (void)__parsec_tp; (void)locals;
  return (((conservative) == (2) && (direction) == (1)) && !(((dimension) == (2) && (side) == (1))));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep24_atline_395 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep24_atline_395_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_Exchange_for_GRID_TO_dep24_atline_395 = {
  .cond = &expr_of_cond_for_flow_of_LBM_Exchange_for_GRID_TO_dep24_atline_395,  /* (((conservative) == (2) && (direction) == (1)) && !(((dimension) == (2) && (side) == (1)))) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 11,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_Exchange_for_GRID_TO,
};
#if MAX_DEP_IN_COUNT < 12  /* number of input dependencies */
    #error Too many input dependencies (supports up to MAX_DEP_IN_COUNT [=24] but found 12). Fix the code or recompile PaRSEC with a larger MAX_DEP_IN_COUNT.
#endif
#if MAX_DEP_OUT_COUNT < 12  /* number of output dependencies */
    #error Too many output dependencies (supports up to MAX_DEP_OUT_COUNT [=24] but found 12). Fix the code or recompile PaRSEC with a larger MAX_DEP_OUT_COUNT.
#endif

static const parsec_flow_t flow_of_LBM_Exchange_for_GRID_TO = {
  .name               = "GRID_TO",
  .sym_type           = PARSEC_SYM_INOUT,
  .flow_flags         = PARSEC_FLOW_ACCESS_RW,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_LBM_Exchange_for_GRID_TO_dep1_atline_371,
 &flow_of_LBM_Exchange_for_GRID_TO_dep2_atline_372,
 &flow_of_LBM_Exchange_for_GRID_TO_dep3_atline_373,
 &flow_of_LBM_Exchange_for_GRID_TO_dep4_atline_374,
 &flow_of_LBM_Exchange_for_GRID_TO_dep5_atline_375,
 &flow_of_LBM_Exchange_for_GRID_TO_dep6_atline_376,
 &flow_of_LBM_Exchange_for_GRID_TO_dep7_atline_377,
 &flow_of_LBM_Exchange_for_GRID_TO_dep8_atline_378,
 &flow_of_LBM_Exchange_for_GRID_TO_dep9_atline_379,
 &flow_of_LBM_Exchange_for_GRID_TO_dep10_atline_380,
 &flow_of_LBM_Exchange_for_GRID_TO_dep11_atline_381,
 &flow_of_LBM_Exchange_for_GRID_TO_dep12_atline_382 },
  .dep_out    = { &flow_of_LBM_Exchange_for_GRID_TO_dep13_atline_384,
 &flow_of_LBM_Exchange_for_GRID_TO_dep14_atline_385,
 &flow_of_LBM_Exchange_for_GRID_TO_dep15_atline_386,
 &flow_of_LBM_Exchange_for_GRID_TO_dep16_atline_387,
 &flow_of_LBM_Exchange_for_GRID_TO_dep17_atline_388,
 &flow_of_LBM_Exchange_for_GRID_TO_dep18_atline_389,
 &flow_of_LBM_Exchange_for_GRID_TO_dep19_atline_390,
 &flow_of_LBM_Exchange_for_GRID_TO_dep20_atline_391,
 &flow_of_LBM_Exchange_for_GRID_TO_dep21_atline_392,
 &flow_of_LBM_Exchange_for_GRID_TO_dep22_atline_393,
 &flow_of_LBM_Exchange_for_GRID_TO_dep23_atline_394,
 &flow_of_LBM_Exchange_for_GRID_TO_dep24_atline_395 }
};

static void
iterate_successors_of_LBM_Exchange(parsec_execution_stream_t *es, const __parsec_LBM_Exchange_task_t *this_task,
               uint32_t action_mask, parsec_ontask_function_t *ontask, void *ontask_arg)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t*)this_task->taskpool;
  parsec_task_t nc;  /* generic placeholder for locals */
  parsec_dep_data_description_t data;
  __parsec_LBM_Exchange_parsec_assignment_t __jdf2c__tmp_locals = *(__parsec_LBM_Exchange_parsec_assignment_t*)&this_task->locals;   /* copy of this_task locals in R/W mode to manage local definitions */
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int x = __jdf2c__tmp_locals.x.value; (void)x;
  const int y = __jdf2c__tmp_locals.y.value; (void)y;
  const int z = __jdf2c__tmp_locals.z.value; (void)z;
  const int s = __jdf2c__tmp_locals.s.value; (void)s;
  const int conservative = __jdf2c__tmp_locals.conservative.value; (void)conservative;
  const int direction = __jdf2c__tmp_locals.direction.value; (void)direction;
  const int dimension = __jdf2c__tmp_locals.dimension.value; (void)dimension;
  const int side = __jdf2c__tmp_locals.side.value; (void)side;
  (void)rank_src; (void)rank_dst; (void)__parsec_tp; (void)vpid_dst;
   data_repo_t *successor_repo; parsec_key_t successor_repo_key;  (void)x;  (void)y;  (void)z;  (void)s;  (void)conservative;  (void)direction;  (void)dimension;  (void)side;
  nc.taskpool  = this_task->taskpool;
  nc.priority  = this_task->priority;
  nc.chore_mask  = PARSEC_DEV_ALL;
#if defined(DISTRIBUTED)
  rank_src = rank_of_descGridDC(x, y, z, conservative, direction);
#endif
  if( action_mask & 0xfff ) {  /* Flow of data GRID_TO [0] */
    data.data   = this_task->data._f_GRID_TO.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
  if( action_mask & 0x1 ) {
        if( ((((conservative) == (0) && (direction) == (0)) && (dimension) == (2)) && (side) == (1)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = (s + 1);
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_CD_0_0", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep13_atline_384, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x2 ) {
        if( (((conservative) == (0) && (direction) == (0)) && !(((dimension) == (2) && (side) == (1)))) ) {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = conservative;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = direction;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = (((side) == (1)) ? ((dimension + 1)) : (dimension));
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = ((side + 1) % 2);
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep14_atline_385, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x4 ) {
        if( ((((conservative) == (1) && (direction) == (0)) && (dimension) == (2)) && (side) == (1)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = (s + 1);
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_CD_1_0", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep15_atline_386, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x8 ) {
        if( (((conservative) == (1) && (direction) == (0)) && !(((dimension) == (2) && (side) == (1)))) ) {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = conservative;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = direction;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = (((side) == (1)) ? ((dimension + 1)) : (dimension));
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = ((side + 1) % 2);
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep16_atline_387, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x10 ) {
        if( ((((conservative) == (2) && (direction) == (0)) && (dimension) == (2)) && (side) == (1)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = (s + 1);
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_CD_2_0", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep17_atline_388, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x20 ) {
        if( (((conservative) == (2) && (direction) == (0)) && !(((dimension) == (2) && (side) == (1)))) ) {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = conservative;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = direction;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = (((side) == (1)) ? ((dimension + 1)) : (dimension));
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = ((side + 1) % 2);
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep18_atline_389, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x40 ) {
        if( ((((conservative) == (0) && (direction) == (1)) && (dimension) == (2)) && (side) == (1)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = (s + 1);
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_CD_0_1", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep19_atline_390, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x80 ) {
        if( (((conservative) == (0) && (direction) == (1)) && !(((dimension) == (2) && (side) == (1)))) ) {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = conservative;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = direction;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = (((side) == (1)) ? ((dimension + 1)) : (dimension));
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = ((side + 1) % 2);
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep20_atline_391, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x100 ) {
        if( ((((conservative) == (1) && (direction) == (1)) && (dimension) == (2)) && (side) == (1)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = (s + 1);
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_CD_1_1", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep21_atline_392, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x200 ) {
        if( (((conservative) == (1) && (direction) == (1)) && !(((dimension) == (2) && (side) == (1)))) ) {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = conservative;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = direction;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = (((side) == (1)) ? ((dimension + 1)) : (dimension));
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = ((side + 1) % 2);
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep22_atline_393, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x400 ) {
        if( ((((conservative) == (2) && (direction) == (1)) && (dimension) == (2)) && (side) == (1)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = (s + 1);
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_CD_2_1", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep23_atline_394, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x800 ) {
        if( (((conservative) == (2) && (direction) == (1)) && !(((dimension) == (2) && (side) == (1)))) ) {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = conservative;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = direction;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = (((side) == (1)) ? ((dimension + 1)) : (dimension));
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = ((side + 1) % 2);
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep24_atline_395, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
  }
  (void)data;(void)nc;(void)es;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static void
iterate_predecessors_of_LBM_Exchange(parsec_execution_stream_t *es, const __parsec_LBM_Exchange_task_t *this_task,
               uint32_t action_mask, parsec_ontask_function_t *ontask, void *ontask_arg)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t*)this_task->taskpool;
  parsec_task_t nc;  /* generic placeholder for locals */
  parsec_dep_data_description_t data;
  __parsec_LBM_Exchange_parsec_assignment_t __jdf2c__tmp_locals = *(__parsec_LBM_Exchange_parsec_assignment_t*)&this_task->locals;   /* copy of this_task locals in R/W mode to manage local definitions */
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int x = __jdf2c__tmp_locals.x.value; (void)x;
  const int y = __jdf2c__tmp_locals.y.value; (void)y;
  const int z = __jdf2c__tmp_locals.z.value; (void)z;
  const int s = __jdf2c__tmp_locals.s.value; (void)s;
  const int conservative = __jdf2c__tmp_locals.conservative.value; (void)conservative;
  const int direction = __jdf2c__tmp_locals.direction.value; (void)direction;
  const int dimension = __jdf2c__tmp_locals.dimension.value; (void)dimension;
  const int side = __jdf2c__tmp_locals.side.value; (void)side;
  (void)rank_src; (void)rank_dst; (void)__parsec_tp; (void)vpid_dst;
   data_repo_t *successor_repo; parsec_key_t successor_repo_key;  (void)x;  (void)y;  (void)z;  (void)s;  (void)conservative;  (void)direction;  (void)dimension;  (void)side;
  nc.taskpool  = this_task->taskpool;
  nc.priority  = this_task->priority;
  nc.chore_mask  = PARSEC_DEV_ALL;
#if defined(DISTRIBUTED)
  rank_src = rank_of_descGridDC(x, y, z, conservative, direction);
#endif
  if( action_mask & 0xfff ) {  /* Flow of data GRID_TO [0] */
    data.data   = this_task->data._f_GRID_TO.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
  if( action_mask & 0x1 ) {
        if( ((((conservative) == (0) && (direction) == (0)) && (dimension) == (0)) && (side) == (0)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = s;
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_CD_0_0", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep1_atline_371, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x2 ) {
        if( (((conservative) == (0) && (direction) == (0)) && !(((dimension) == (0) && (side) == (0)))) ) {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = conservative;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = direction;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = (((side) == (0)) ? ((dimension - 1)) : (dimension));
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = ((side + 1) % 2);
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep2_atline_372, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x4 ) {
        if( ((((conservative) == (1) && (direction) == (0)) && (dimension) == (0)) && (side) == (0)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = s;
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_CD_1_0", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep3_atline_373, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x8 ) {
        if( (((conservative) == (1) && (direction) == (0)) && !(((dimension) == (0) && (side) == (0)))) ) {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = conservative;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = direction;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = (((side) == (0)) ? ((dimension - 1)) : (dimension));
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = ((side + 1) % 2);
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep4_atline_374, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x10 ) {
        if( ((((conservative) == (2) && (direction) == (0)) && (dimension) == (0)) && (side) == (0)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = s;
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_CD_2_0", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep5_atline_375, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x20 ) {
        if( (((conservative) == (2) && (direction) == (0)) && !(((dimension) == (0) && (side) == (0)))) ) {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = conservative;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = direction;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = (((side) == (0)) ? ((dimension - 1)) : (dimension));
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = ((side + 1) % 2);
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep6_atline_376, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x40 ) {
        if( ((((conservative) == (0) && (direction) == (1)) && (dimension) == (0)) && (side) == (0)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = s;
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_CD_0_1", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep7_atline_377, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x80 ) {
        if( (((conservative) == (0) && (direction) == (1)) && !(((dimension) == (0) && (side) == (0)))) ) {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = conservative;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = direction;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = (((side) == (0)) ? ((dimension - 1)) : (dimension));
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = ((side + 1) % 2);
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep8_atline_378, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x100 ) {
        if( ((((conservative) == (1) && (direction) == (1)) && (dimension) == (0)) && (side) == (0)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = s;
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_CD_1_1", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep9_atline_379, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x200 ) {
        if( (((conservative) == (1) && (direction) == (1)) && !(((dimension) == (0) && (side) == (0)))) ) {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = conservative;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = direction;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = (((side) == (0)) ? ((dimension - 1)) : (dimension));
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = ((side + 1) % 2);
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep10_atline_380, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x400 ) {
        if( ((((conservative) == (2) && (direction) == (1)) && (dimension) == (0)) && (side) == (0)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = s;
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_CD_2_1", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep11_atline_381, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x800 ) {
        if( (((conservative) == (2) && (direction) == (1)) && !(((dimension) == (0) && (side) == (0)))) ) {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = conservative;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = direction;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = (((side) == (0)) ? ((dimension - 1)) : (dimension));
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = ((side + 1) % 2);
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_TO", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_Exchange_for_GRID_TO_dep12_atline_382, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
  }
  (void)data;(void)nc;(void)es;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_LBM_Exchange(parsec_execution_stream_t *es, __parsec_LBM_Exchange_task_t *this_task, uint32_t action_mask, parsec_remote_deps_t *deps)
{
PARSEC_PINS(es, RELEASE_DEPS_BEGIN, (parsec_task_t *)this_task);{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  parsec_release_dep_fct_arg_t arg;
  int __vp_id;
  int consume_local_repo = 0;
  arg.action_mask = action_mask;
  arg.output_entry = NULL;
  arg.output_repo = NULL;
#if defined(DISTRIBUTED)
  arg.remote_deps = deps;
#endif  /* defined(DISTRIBUTED) */
  assert(NULL != es);
  arg.ready_lists = alloca(sizeof(parsec_task_t *) * es->virtual_process->parsec_context->nb_vp);
  for( __vp_id = 0; __vp_id < es->virtual_process->parsec_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );
  (void)__parsec_tp; (void)deps;
  if( action_mask & PARSEC_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data._f_GRID_TO.source_repo_entry ) {
        data_repo_entry_used_once( this_task->data._f_GRID_TO.source_repo, this_task->data._f_GRID_TO.source_repo_entry->ht_item.key );
    }
  }
  consume_local_repo = (this_task->repo_entry != NULL);
  arg.output_repo = Exchange_repo;
  arg.output_entry = this_task->repo_entry;
  arg.output_usage = 0;
  if( action_mask & (PARSEC_ACTION_RELEASE_LOCAL_DEPS | PARSEC_ACTION_GET_REPO_ENTRY) ) {
    arg.output_entry = data_repo_lookup_entry_and_create( es, arg.output_repo, __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals));
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(PARSEC_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  if(action_mask & ( PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE ) ){
    /* Generate the reshape promise for thet outputs that need it */
    iterate_successors_of_LBM_Exchange(es, this_task, action_mask, parsec_set_up_reshape_promise, &arg);
   }
  iterate_successors_of_LBM_Exchange(es, this_task, action_mask, parsec_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & PARSEC_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    parsec_remote_dep_activate(es, (parsec_task_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {
    data_repo_entry_addto_usage_limit(Exchange_repo, arg.output_entry->ht_item.key, arg.output_usage);
    __parsec_schedule_vp(es, arg.ready_lists, 0);
  }
      if (consume_local_repo) {
         data_repo_entry_used_once( Exchange_repo, this_task->repo_entry->ht_item.key );
      }
  if( action_mask & PARSEC_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data._f_GRID_TO.data_in ) {
        PARSEC_DATA_COPY_RELEASE(this_task->data._f_GRID_TO.data_in);
    }
  }
PARSEC_PINS(es, RELEASE_DEPS_END, (parsec_task_t *)this_task);}
  return 0;
}

static int data_lookup_of_LBM_Exchange(parsec_execution_stream_t *es, __parsec_LBM_Exchange_task_t *this_task)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  parsec_assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__parsec_tp; (void)generic_locals; (void)es;
  parsec_data_copy_t *chunk = NULL;
  data_repo_t        *reshape_repo  = NULL, *consumed_repo  = NULL;
  data_repo_entry_t  *reshape_entry = NULL, *consumed_entry = NULL;
  parsec_key_t        reshape_entry_key = 0, consumed_entry_key = 0;
  uint8_t             consumed_flow_index;
  parsec_dep_data_description_t data;
  int ret;
  (void)reshape_repo; (void)reshape_entry; (void)reshape_entry_key;
  (void)consumed_repo; (void)consumed_entry; (void)consumed_entry_key;
  (void)consumed_flow_index;
  (void)chunk; (void)data; (void)ret;
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int s = this_task->locals.s.value; (void)s;
  const int conservative = this_task->locals.conservative.value; (void)conservative;
  const int direction = this_task->locals.direction.value; (void)direction;
  const int dimension = this_task->locals.dimension.value; (void)dimension;
  const int side = this_task->locals.side.value; (void)side;
  if( NULL == this_task->repo_entry ){
    this_task->repo_entry = data_repo_lookup_entry_and_create(es, Exchange_repo,                                       __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals));
    data_repo_entry_addto_usage_limit(Exchange_repo, this_task->repo_entry->ht_item.key, 1);    this_task->repo_entry ->generator = (void*)this_task;  /* for AYU */
#if defined(PARSEC_SIM)
    assert(this_task->repo_entry ->sim_exec_date == 0);
    this_task->repo_entry ->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  /* The reshape repo is the current task repo. */  reshape_repo = Exchange_repo;
  reshape_entry_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals) ;
  reshape_entry = this_task->repo_entry;
  /* Lookup the input data, and store them in the context if any */

if(! this_task->data._f_GRID_TO.fulfill ){ /* Flow GRID_TO */
  consumed_repo = NULL;
  consumed_entry_key = 0;
  consumed_entry = NULL;
  chunk = NULL;

    this_task->data._f_GRID_TO.data_out = NULL;  /* By default, if nothing matches */
    if( ((((conservative) == (0) && (direction) == (0)) && (dimension) == (0)) && (side) == (0)) ) {
    /* Flow GRID_TO [0] dependency [0] from predecessor LBM_STEP */
    if( NULL == (chunk = this_task->data._f_GRID_TO.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_LBM_STEP_parsec_assignment_t *target_locals = (__parsec_LBM_LBM_STEP_parsec_assignment_t*)&generic_locals;
      const int LBM_STEPx = target_locals->x.value = x; (void)LBM_STEPx;
      const int LBM_STEPy = target_locals->y.value = y; (void)LBM_STEPy;
      const int LBM_STEPz = target_locals->z.value = z; (void)LBM_STEPz;
      const int LBM_STEPs = target_locals->s.value = s; (void)LBM_STEPs;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 0;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = LBM_STEP_repo;
          consumed_entry_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_TO", &LBM_LBM_STEP, "GRID_CD_0_0", target_locals, chunk);
      this_task->data._f_GRID_TO.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_TO.source_repo;
      consumed_entry = this_task->data._f_GRID_TO.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_TO.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 0;
          assert( (this_task->data._f_GRID_TO.source_repo == reshape_repo)
               && (this_task->data._f_GRID_TO.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_TO.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_Exchange_for_GRID_TO, 0);
#endif
    }
    }
    else if( (((conservative) == (0) && (direction) == (0)) && !(((dimension) == (0) && (side) == (0)))) ) {
    /* Flow GRID_TO [0] dependency [1] from predecessor Exchange */
    if( NULL == (chunk = this_task->data._f_GRID_TO.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_Exchange_parsec_assignment_t *target_locals = (__parsec_LBM_Exchange_parsec_assignment_t*)&generic_locals;
      const int Exchangex = target_locals->x.value = x; (void)Exchangex;
      const int Exchangey = target_locals->y.value = y; (void)Exchangey;
      const int Exchangez = target_locals->z.value = z; (void)Exchangez;
      const int Exchanges = target_locals->s.value = s; (void)Exchanges;
      const int Exchangeconservative = target_locals->conservative.value = conservative; (void)Exchangeconservative;
      const int Exchangedirection = target_locals->direction.value = direction; (void)Exchangedirection;
      const int Exchangedimension = target_locals->dimension.value = (((side) == (0)) ? ((dimension - 1)) : (dimension)); (void)Exchangedimension;
      const int Exchangeside = target_locals->side.value = ((side + 1) % 2); (void)Exchangeside;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 0;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = Exchange_repo;
          consumed_entry_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_TO", &LBM_Exchange, "GRID_TO", target_locals, chunk);
      this_task->data._f_GRID_TO.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_TO.source_repo;
      consumed_entry = this_task->data._f_GRID_TO.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_TO.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 0;
          assert( (this_task->data._f_GRID_TO.source_repo == reshape_repo)
               && (this_task->data._f_GRID_TO.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_TO.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_Exchange_for_GRID_TO, 0);
#endif
    }
    }
    else if( ((((conservative) == (1) && (direction) == (0)) && (dimension) == (0)) && (side) == (0)) ) {
    /* Flow GRID_TO [0] dependency [2] from predecessor LBM_STEP */
    if( NULL == (chunk = this_task->data._f_GRID_TO.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_LBM_STEP_parsec_assignment_t *target_locals = (__parsec_LBM_LBM_STEP_parsec_assignment_t*)&generic_locals;
      const int LBM_STEPx = target_locals->x.value = x; (void)LBM_STEPx;
      const int LBM_STEPy = target_locals->y.value = y; (void)LBM_STEPy;
      const int LBM_STEPz = target_locals->z.value = z; (void)LBM_STEPz;
      const int LBM_STEPs = target_locals->s.value = s; (void)LBM_STEPs;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 0;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = LBM_STEP_repo;
          consumed_entry_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 1;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_TO", &LBM_LBM_STEP, "GRID_CD_1_0", target_locals, chunk);
      this_task->data._f_GRID_TO.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_TO.source_repo;
      consumed_entry = this_task->data._f_GRID_TO.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_TO.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 0;
          assert( (this_task->data._f_GRID_TO.source_repo == reshape_repo)
               && (this_task->data._f_GRID_TO.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 1;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_TO.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_Exchange_for_GRID_TO, 0);
#endif
    }
    }
    else if( (((conservative) == (1) && (direction) == (0)) && !(((dimension) == (0) && (side) == (0)))) ) {
    /* Flow GRID_TO [0] dependency [3] from predecessor Exchange */
    if( NULL == (chunk = this_task->data._f_GRID_TO.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_Exchange_parsec_assignment_t *target_locals = (__parsec_LBM_Exchange_parsec_assignment_t*)&generic_locals;
      const int Exchangex = target_locals->x.value = x; (void)Exchangex;
      const int Exchangey = target_locals->y.value = y; (void)Exchangey;
      const int Exchangez = target_locals->z.value = z; (void)Exchangez;
      const int Exchanges = target_locals->s.value = s; (void)Exchanges;
      const int Exchangeconservative = target_locals->conservative.value = conservative; (void)Exchangeconservative;
      const int Exchangedirection = target_locals->direction.value = direction; (void)Exchangedirection;
      const int Exchangedimension = target_locals->dimension.value = (((side) == (0)) ? ((dimension - 1)) : (dimension)); (void)Exchangedimension;
      const int Exchangeside = target_locals->side.value = ((side + 1) % 2); (void)Exchangeside;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 0;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = Exchange_repo;
          consumed_entry_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_TO", &LBM_Exchange, "GRID_TO", target_locals, chunk);
      this_task->data._f_GRID_TO.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_TO.source_repo;
      consumed_entry = this_task->data._f_GRID_TO.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_TO.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 0;
          assert( (this_task->data._f_GRID_TO.source_repo == reshape_repo)
               && (this_task->data._f_GRID_TO.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_TO.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_Exchange_for_GRID_TO, 0);
#endif
    }
    }
    else if( ((((conservative) == (2) && (direction) == (0)) && (dimension) == (0)) && (side) == (0)) ) {
    /* Flow GRID_TO [0] dependency [4] from predecessor LBM_STEP */
    if( NULL == (chunk = this_task->data._f_GRID_TO.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_LBM_STEP_parsec_assignment_t *target_locals = (__parsec_LBM_LBM_STEP_parsec_assignment_t*)&generic_locals;
      const int LBM_STEPx = target_locals->x.value = x; (void)LBM_STEPx;
      const int LBM_STEPy = target_locals->y.value = y; (void)LBM_STEPy;
      const int LBM_STEPz = target_locals->z.value = z; (void)LBM_STEPz;
      const int LBM_STEPs = target_locals->s.value = s; (void)LBM_STEPs;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 0;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = LBM_STEP_repo;
          consumed_entry_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 2;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_TO", &LBM_LBM_STEP, "GRID_CD_2_0", target_locals, chunk);
      this_task->data._f_GRID_TO.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_TO.source_repo;
      consumed_entry = this_task->data._f_GRID_TO.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_TO.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 0;
          assert( (this_task->data._f_GRID_TO.source_repo == reshape_repo)
               && (this_task->data._f_GRID_TO.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 2;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_TO.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_Exchange_for_GRID_TO, 0);
#endif
    }
    }
    else if( (((conservative) == (2) && (direction) == (0)) && !(((dimension) == (0) && (side) == (0)))) ) {
    /* Flow GRID_TO [0] dependency [5] from predecessor Exchange */
    if( NULL == (chunk = this_task->data._f_GRID_TO.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_Exchange_parsec_assignment_t *target_locals = (__parsec_LBM_Exchange_parsec_assignment_t*)&generic_locals;
      const int Exchangex = target_locals->x.value = x; (void)Exchangex;
      const int Exchangey = target_locals->y.value = y; (void)Exchangey;
      const int Exchangez = target_locals->z.value = z; (void)Exchangez;
      const int Exchanges = target_locals->s.value = s; (void)Exchanges;
      const int Exchangeconservative = target_locals->conservative.value = conservative; (void)Exchangeconservative;
      const int Exchangedirection = target_locals->direction.value = direction; (void)Exchangedirection;
      const int Exchangedimension = target_locals->dimension.value = (((side) == (0)) ? ((dimension - 1)) : (dimension)); (void)Exchangedimension;
      const int Exchangeside = target_locals->side.value = ((side + 1) % 2); (void)Exchangeside;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 0;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = Exchange_repo;
          consumed_entry_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_TO", &LBM_Exchange, "GRID_TO", target_locals, chunk);
      this_task->data._f_GRID_TO.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_TO.source_repo;
      consumed_entry = this_task->data._f_GRID_TO.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_TO.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 0;
          assert( (this_task->data._f_GRID_TO.source_repo == reshape_repo)
               && (this_task->data._f_GRID_TO.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_TO.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_Exchange_for_GRID_TO, 0);
#endif
    }
    }
    else if( ((((conservative) == (0) && (direction) == (1)) && (dimension) == (0)) && (side) == (0)) ) {
    /* Flow GRID_TO [0] dependency [6] from predecessor LBM_STEP */
    if( NULL == (chunk = this_task->data._f_GRID_TO.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_LBM_STEP_parsec_assignment_t *target_locals = (__parsec_LBM_LBM_STEP_parsec_assignment_t*)&generic_locals;
      const int LBM_STEPx = target_locals->x.value = x; (void)LBM_STEPx;
      const int LBM_STEPy = target_locals->y.value = y; (void)LBM_STEPy;
      const int LBM_STEPz = target_locals->z.value = z; (void)LBM_STEPz;
      const int LBM_STEPs = target_locals->s.value = s; (void)LBM_STEPs;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 0;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = LBM_STEP_repo;
          consumed_entry_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 3;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_TO", &LBM_LBM_STEP, "GRID_CD_0_1", target_locals, chunk);
      this_task->data._f_GRID_TO.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_TO.source_repo;
      consumed_entry = this_task->data._f_GRID_TO.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_TO.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 0;
          assert( (this_task->data._f_GRID_TO.source_repo == reshape_repo)
               && (this_task->data._f_GRID_TO.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 3;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_TO.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_Exchange_for_GRID_TO, 0);
#endif
    }
    }
    else if( (((conservative) == (0) && (direction) == (1)) && !(((dimension) == (0) && (side) == (0)))) ) {
    /* Flow GRID_TO [0] dependency [7] from predecessor Exchange */
    if( NULL == (chunk = this_task->data._f_GRID_TO.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_Exchange_parsec_assignment_t *target_locals = (__parsec_LBM_Exchange_parsec_assignment_t*)&generic_locals;
      const int Exchangex = target_locals->x.value = x; (void)Exchangex;
      const int Exchangey = target_locals->y.value = y; (void)Exchangey;
      const int Exchangez = target_locals->z.value = z; (void)Exchangez;
      const int Exchanges = target_locals->s.value = s; (void)Exchanges;
      const int Exchangeconservative = target_locals->conservative.value = conservative; (void)Exchangeconservative;
      const int Exchangedirection = target_locals->direction.value = direction; (void)Exchangedirection;
      const int Exchangedimension = target_locals->dimension.value = (((side) == (0)) ? ((dimension - 1)) : (dimension)); (void)Exchangedimension;
      const int Exchangeside = target_locals->side.value = ((side + 1) % 2); (void)Exchangeside;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 0;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = Exchange_repo;
          consumed_entry_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_TO", &LBM_Exchange, "GRID_TO", target_locals, chunk);
      this_task->data._f_GRID_TO.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_TO.source_repo;
      consumed_entry = this_task->data._f_GRID_TO.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_TO.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 0;
          assert( (this_task->data._f_GRID_TO.source_repo == reshape_repo)
               && (this_task->data._f_GRID_TO.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_TO.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_Exchange_for_GRID_TO, 0);
#endif
    }
    }
    else if( ((((conservative) == (1) && (direction) == (1)) && (dimension) == (0)) && (side) == (0)) ) {
    /* Flow GRID_TO [0] dependency [8] from predecessor LBM_STEP */
    if( NULL == (chunk = this_task->data._f_GRID_TO.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_LBM_STEP_parsec_assignment_t *target_locals = (__parsec_LBM_LBM_STEP_parsec_assignment_t*)&generic_locals;
      const int LBM_STEPx = target_locals->x.value = x; (void)LBM_STEPx;
      const int LBM_STEPy = target_locals->y.value = y; (void)LBM_STEPy;
      const int LBM_STEPz = target_locals->z.value = z; (void)LBM_STEPz;
      const int LBM_STEPs = target_locals->s.value = s; (void)LBM_STEPs;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 0;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = LBM_STEP_repo;
          consumed_entry_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 4;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_TO", &LBM_LBM_STEP, "GRID_CD_1_1", target_locals, chunk);
      this_task->data._f_GRID_TO.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_TO.source_repo;
      consumed_entry = this_task->data._f_GRID_TO.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_TO.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 0;
          assert( (this_task->data._f_GRID_TO.source_repo == reshape_repo)
               && (this_task->data._f_GRID_TO.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 4;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_TO.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_Exchange_for_GRID_TO, 0);
#endif
    }
    }
    else if( (((conservative) == (1) && (direction) == (1)) && !(((dimension) == (0) && (side) == (0)))) ) {
    /* Flow GRID_TO [0] dependency [9] from predecessor Exchange */
    if( NULL == (chunk = this_task->data._f_GRID_TO.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_Exchange_parsec_assignment_t *target_locals = (__parsec_LBM_Exchange_parsec_assignment_t*)&generic_locals;
      const int Exchangex = target_locals->x.value = x; (void)Exchangex;
      const int Exchangey = target_locals->y.value = y; (void)Exchangey;
      const int Exchangez = target_locals->z.value = z; (void)Exchangez;
      const int Exchanges = target_locals->s.value = s; (void)Exchanges;
      const int Exchangeconservative = target_locals->conservative.value = conservative; (void)Exchangeconservative;
      const int Exchangedirection = target_locals->direction.value = direction; (void)Exchangedirection;
      const int Exchangedimension = target_locals->dimension.value = (((side) == (0)) ? ((dimension - 1)) : (dimension)); (void)Exchangedimension;
      const int Exchangeside = target_locals->side.value = ((side + 1) % 2); (void)Exchangeside;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 0;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = Exchange_repo;
          consumed_entry_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_TO", &LBM_Exchange, "GRID_TO", target_locals, chunk);
      this_task->data._f_GRID_TO.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_TO.source_repo;
      consumed_entry = this_task->data._f_GRID_TO.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_TO.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 0;
          assert( (this_task->data._f_GRID_TO.source_repo == reshape_repo)
               && (this_task->data._f_GRID_TO.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_TO.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_Exchange_for_GRID_TO, 0);
#endif
    }
    }
    else if( ((((conservative) == (2) && (direction) == (1)) && (dimension) == (0)) && (side) == (0)) ) {
    /* Flow GRID_TO [0] dependency [10] from predecessor LBM_STEP */
    if( NULL == (chunk = this_task->data._f_GRID_TO.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_LBM_STEP_parsec_assignment_t *target_locals = (__parsec_LBM_LBM_STEP_parsec_assignment_t*)&generic_locals;
      const int LBM_STEPx = target_locals->x.value = x; (void)LBM_STEPx;
      const int LBM_STEPy = target_locals->y.value = y; (void)LBM_STEPy;
      const int LBM_STEPz = target_locals->z.value = z; (void)LBM_STEPz;
      const int LBM_STEPs = target_locals->s.value = s; (void)LBM_STEPs;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 0;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = LBM_STEP_repo;
          consumed_entry_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 5;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_TO", &LBM_LBM_STEP, "GRID_CD_2_1", target_locals, chunk);
      this_task->data._f_GRID_TO.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_TO.source_repo;
      consumed_entry = this_task->data._f_GRID_TO.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_TO.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 0;
          assert( (this_task->data._f_GRID_TO.source_repo == reshape_repo)
               && (this_task->data._f_GRID_TO.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 5;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_TO.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_Exchange_for_GRID_TO, 0);
#endif
    }
    }
    else if( (((conservative) == (2) && (direction) == (1)) && !(((dimension) == (0) && (side) == (0)))) ) {
    /* Flow GRID_TO [0] dependency [11] from predecessor Exchange */
    if( NULL == (chunk = this_task->data._f_GRID_TO.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_Exchange_parsec_assignment_t *target_locals = (__parsec_LBM_Exchange_parsec_assignment_t*)&generic_locals;
      const int Exchangex = target_locals->x.value = x; (void)Exchangex;
      const int Exchangey = target_locals->y.value = y; (void)Exchangey;
      const int Exchangez = target_locals->z.value = z; (void)Exchangez;
      const int Exchanges = target_locals->s.value = s; (void)Exchanges;
      const int Exchangeconservative = target_locals->conservative.value = conservative; (void)Exchangeconservative;
      const int Exchangedirection = target_locals->direction.value = direction; (void)Exchangedirection;
      const int Exchangedimension = target_locals->dimension.value = (((side) == (0)) ? ((dimension - 1)) : (dimension)); (void)Exchangedimension;
      const int Exchangeside = target_locals->side.value = ((side + 1) % 2); (void)Exchangeside;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 0;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = Exchange_repo;
          consumed_entry_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_TO", &LBM_Exchange, "GRID_TO", target_locals, chunk);
      this_task->data._f_GRID_TO.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_TO.source_repo;
      consumed_entry = this_task->data._f_GRID_TO.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_TO.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 0;
          assert( (this_task->data._f_GRID_TO.source_repo == reshape_repo)
               && (this_task->data._f_GRID_TO.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_TO.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_Exchange_for_GRID_TO, 0);
#endif
    }
    }
    this_task->data._f_GRID_TO.data_in     = chunk;
    this_task->data._f_GRID_TO.source_repo       = consumed_repo;
    this_task->data._f_GRID_TO.source_repo_entry = consumed_entry;
    if( this_task->data._f_GRID_TO.source_repo_entry == this_task->repo_entry ){
      /* in case we have consume from this task repo entry for the flow,
       * it is cleaned up, avoiding having old stuff during release_deps_of
       */
      this_task->repo_entry->data[0] = NULL;
    }
    this_task->data._f_GRID_TO.fulfill = 1;
}

  /** Generate profiling information */
#if defined(PARSEC_PROF_TRACE)
  this_task->prof_info.desc = (parsec_data_collection_t*)__parsec_tp->super._g_descGridDC;
  this_task->prof_info.priority = this_task->priority;
  this_task->prof_info.data_id   = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->data_key((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, x, y, z, conservative, direction);
  this_task->prof_info.task_class_id = this_task->task_class->task_class_id;
  this_task->prof_info.task_return_code = -1;
#endif  /* defined(PARSEC_PROF_TRACE) */
  return PARSEC_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_LBM_Exchange(parsec_execution_stream_t *es, const __parsec_LBM_Exchange_task_t *this_task,
              uint32_t* flow_mask, parsec_dep_data_description_t* data)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  (void)__parsec_tp; (void)es; (void)this_task; (void)data;
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int s = this_task->locals.s.value; (void)s;
  const int conservative = this_task->locals.conservative.value; (void)conservative;
  const int direction = this_task->locals.direction.value; (void)direction;
  const int dimension = this_task->locals.dimension.value; (void)dimension;
  const int side = this_task->locals.side.value; (void)side;
  data->local.arena = data->remote.arena = NULL;
  data->local.src_datatype  = data->local.dst_datatype = PARSEC_DATATYPE_NULL;
  data->local.src_count     = data->local.dst_count = 0;
  data->local.src_displ     = data->local.dst_displ = 0;
  data->data_future  = NULL;
  if( (*flow_mask) & 0x80000000U ) { /* these are the input dependencies remote datatypes  */
if( (*flow_mask) & 0x1U ) {  /* Flow GRID_TO */
    if( ((*flow_mask) & 0x1U)
 && (((((conservative) == (0) && (direction) == (0)) && (dimension) == (0)) && (side) == (0)) || (((conservative) == (0) && (direction) == (0)) && !(((dimension) == (0) && (side) == (0)))) || ((((conservative) == (1) && (direction) == (0)) && (dimension) == (0)) && (side) == (0)) || (((conservative) == (1) && (direction) == (0)) && !(((dimension) == (0) && (side) == (0)))) || ((((conservative) == (2) && (direction) == (0)) && (dimension) == (0)) && (side) == (0)) || (((conservative) == (2) && (direction) == (0)) && !(((dimension) == (0) && (side) == (0)))) || ((((conservative) == (0) && (direction) == (1)) && (dimension) == (0)) && (side) == (0)) || (((conservative) == (0) && (direction) == (1)) && !(((dimension) == (0) && (side) == (0)))) || ((((conservative) == (1) && (direction) == (1)) && (dimension) == (0)) && (side) == (0)) || (((conservative) == (1) && (direction) == (1)) && !(((dimension) == (0) && (side) == (0)))) || ((((conservative) == (2) && (direction) == (1)) && (dimension) == (0)) && (side) == (0)) || (((conservative) == (2) && (direction) == (1)) && !(((dimension) == (0) && (side) == (0))))) ) {
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x1U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x1U) */
    goto no_mask_match;
  }

  /* these are the output dependencies remote datatypes */
if( (*flow_mask) & 0xfffU ) {  /* Flow GRID_TO */
    if( ((*flow_mask) & 0xfffU)
 && (((((conservative) == (0) && (direction) == (0)) && (dimension) == (2)) && (side) == (1)) || (((conservative) == (0) && (direction) == (0)) && !(((dimension) == (2) && (side) == (1)))) || ((((conservative) == (1) && (direction) == (0)) && (dimension) == (2)) && (side) == (1)) || (((conservative) == (1) && (direction) == (0)) && !(((dimension) == (2) && (side) == (1)))) || ((((conservative) == (2) && (direction) == (0)) && (dimension) == (2)) && (side) == (1)) || (((conservative) == (2) && (direction) == (0)) && !(((dimension) == (2) && (side) == (1)))) || ((((conservative) == (0) && (direction) == (1)) && (dimension) == (2)) && (side) == (1)) || (((conservative) == (0) && (direction) == (1)) && !(((dimension) == (2) && (side) == (1)))) || ((((conservative) == (1) && (direction) == (1)) && (dimension) == (2)) && (side) == (1)) || (((conservative) == (1) && (direction) == (1)) && !(((dimension) == (2) && (side) == (1)))) || ((((conservative) == (2) && (direction) == (1)) && (dimension) == (2)) && (side) == (1)) || (((conservative) == (2) && (direction) == (1)) && !(((dimension) == (2) && (side) == (1))))) ) {
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0xfffU;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0xfffU) */
 no_mask_match:
  data->data                             = NULL;
  data->local.arena = data->remote.arena = NULL;
  data->local.src_datatype  = data->local.dst_datatype = PARSEC_DATATYPE_NULL;
  data->remote.src_datatype = data->remote.dst_datatype = PARSEC_DATATYPE_NULL;
  data->local.src_count     = data->local.dst_count = 0;
  data->remote.src_count    = data->remote.dst_count = 0;
  data->local.src_displ     = data->local.dst_displ = 0;
  data->remote.src_displ    = data->remote.dst_displ = 0;
  data->data_future  = NULL;
  (*flow_mask) = 0;  /* nothing left */
  (void)x;  (void)y;  (void)z;  (void)s;  (void)conservative;  (void)direction;  (void)dimension;  (void)side;
  return PARSEC_HOOK_RETURN_DONE;
}
#if defined(PARSEC_HAVE_CUDA)
struct parsec_body_cuda_LBM_Exchange_s {
  uint8_t      index;
  cudaStream_t stream;
  void*           dyld_fn;
};

static int cuda_kernel_submit_LBM_Exchange(parsec_device_gpu_module_t  *gpu_device,
                                    parsec_gpu_task_t           *gpu_task,
                                    parsec_gpu_exec_stream_t    *gpu_stream )
{
  __parsec_LBM_Exchange_task_t *this_task = (__parsec_LBM_Exchange_task_t *)gpu_task->ec;
  parsec_device_cuda_module_t *cuda_device = (parsec_device_cuda_module_t*)gpu_device;
  parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t*)gpu_stream;
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  struct parsec_body_cuda_LBM_Exchange_s parsec_body = { cuda_device->cuda_index, cuda_stream->cuda_stream, NULL };
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int s = this_task->locals.s.value; (void)s;
  const int conservative = this_task->locals.conservative.value; (void)conservative;
  const int direction = this_task->locals.direction.value; (void)direction;
  const int dimension = this_task->locals.dimension.value; (void)dimension;
  const int side = this_task->locals.side.value; (void)side;
  (void)x;  (void)y;  (void)z;  (void)s;  (void)conservative;  (void)direction;  (void)dimension;  (void)side;

  (void)gpu_device; (void)gpu_stream; (void)__parsec_tp; (void)parsec_body; (void)cuda_device; (void)cuda_stream;
  /** Declare the variables that will hold the data, and all the accounting for each */
    parsec_data_copy_t *_f_GRID_TO = this_task->data._f_GRID_TO.data_out;
    void *GRID_TO = PARSEC_DATA_COPY_GET_PTR(_f_GRID_TO); (void)GRID_TO;

  /** Update starting simulation date */
#if defined(PARSEC_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eGRID_TO = this_task->data._f_GRID_TO.source_repo_entry;
    if( (NULL != eGRID_TO) && (eGRID_TO->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eGRID_TO->sim_exec_date;
    if( this_task->task_class->sim_cost_fct != NULL ) {
      this_task->sim_exec_date += this_task->task_class->sim_cost_fct(this_task);
    }
    if( es->largest_simulation_date < this_task->sim_exec_date )
      es->largest_simulation_date = this_task->sim_exec_date;
  }
#endif
  /** Cache Awareness Accounting */
#if defined(PARSEC_CACHE_AWARENESS)
  cache_buf_referenced(es->closest_cache, GRID_TO);
#endif /* PARSEC_CACHE_AWARENESS */
#if defined(PARSEC_DEBUG_NOISIER)
  {
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream, "GPU[%s]:\tEnqueue on device %s priority %d", gpu_device->super.name, 
           parsec_task_snprintf(tmp, MAX_TASK_STRLEN, (parsec_task_t *)this_task),
           this_task->priority );
  }
#endif /* defined(PARSEC_DEBUG_NOISIER) */


#if !defined(PARSEC_PROF_DRY_BODY)

/*-----                                Exchange BODY                                  -----*/

#if defined(PARSEC_PROF_TRACE)
  if(gpu_stream->prof_event_track_enable) {
    PARSEC_TASK_PROF_TRACE(gpu_stream->profiling,
                           PARSEC_PROF_FUNC_KEY_START(this_task->taskpool,
                                     this_task->task_class->task_class_id),
                           (parsec_task_t*)this_task);
    gpu_task->prof_key_end = PARSEC_PROF_FUNC_KEY_END(this_task->taskpool,
                                   this_task->task_class->task_class_id);
    gpu_task->prof_event_id = this_task->task_class->key_functions->
           key_hash(this_task->task_class->make_key(this_task->taskpool, ((parsec_task_t*)this_task)->locals), NULL);
    gpu_task->prof_tp_id = this_task->taskpool->taskpool_id;
  }
#endif /* PARSEC_PROF_TRACE */
#line 399 "LBM.jdf"
    printf("[Process %d] kernel Exchange (%d %d %d %d %d %d %d %d) grid_to=%p\n",
                rank, x, y, z, s, conservative, direction, dimension, side, GRID_TO);

#line 5276 "LBM.c"
/*-----                              END OF Exchange BODY                              -----*/



#endif /*!defined(PARSEC_PROF_DRY_BODY)*/

  return PARSEC_HOOK_RETURN_DONE;
}

static int hook_of_LBM_Exchange_CUDA(parsec_execution_stream_t *es, __parsec_LBM_Exchange_task_t *this_task)
{
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  parsec_gpu_task_t *gpu_task;
  double ratio;
  int dev_index;
    const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int s = this_task->locals.s.value; (void)s;
  const int conservative = this_task->locals.conservative.value; (void)conservative;
  const int direction = this_task->locals.direction.value; (void)direction;
  const int dimension = this_task->locals.dimension.value; (void)dimension;
  const int side = this_task->locals.side.value; (void)side;
  (void)x;  (void)y;  (void)z;  (void)s;  (void)conservative;  (void)direction;  (void)dimension;  (void)side;

  (void)es; (void)__parsec_tp;

  ratio = 1.;
  dev_index = parsec_get_best_device((parsec_task_t*)this_task, ratio);
  assert(dev_index >= 0);
  if( dev_index < 2 ) {
    return PARSEC_HOOK_RETURN_NEXT;  /* Fall back */
  }

  gpu_task = (parsec_gpu_task_t*)calloc(1, sizeof(parsec_gpu_task_t));
  PARSEC_OBJ_CONSTRUCT(gpu_task, parsec_list_item_t);
  gpu_task->ec = (parsec_task_t*)this_task;
  gpu_task->submit = &cuda_kernel_submit_LBM_Exchange;
  gpu_task->task_type = 0;
  gpu_task->load = ratio * parsec_device_sweight[dev_index];
  gpu_task->last_data_check_epoch = -1;  /* force at least one validation for the task */
  gpu_task->stage_in  = parsec_default_cuda_stage_in;
  gpu_task->stage_out = parsec_default_cuda_stage_out;
  gpu_task->pushout = 0;
  gpu_task->flow[0]         = &flow_of_LBM_Exchange_for_GRID_TO;
  gpu_task->flow_dc[0] = NULL;
  gpu_task->flow_nb_elts[0] = gpu_task->ec->data[0].data_in->original->nb_elts;
  parsec_device_load[dev_index] += gpu_task->load;

  return parsec_cuda_kernel_scheduler( es, gpu_task, dev_index );
}

#endif  /*  defined(PARSEC_HAVE_CUDA) */
static int complete_hook_of_LBM_Exchange(parsec_execution_stream_t *es, __parsec_LBM_Exchange_task_t *this_task)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
#if defined(DISTRIBUTED)
    const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int s = this_task->locals.s.value; (void)s;
  const int conservative = this_task->locals.conservative.value; (void)conservative;
  const int direction = this_task->locals.direction.value; (void)direction;
  const int dimension = this_task->locals.dimension.value; (void)dimension;
  const int side = this_task->locals.side.value; (void)side;
#endif  /* defined(DISTRIBUTED) */
  (void)es; (void)__parsec_tp;
parsec_data_t* data_t_desc = NULL; (void)data_t_desc;
  if ( NULL != this_task->data._f_GRID_TO.data_out ) {
#if defined(PARSEC_DEBUG_NOISIER)
     char tmp[128];
#endif
     this_task->data._f_GRID_TO.data_out->version++;  /* GRID_TO */
     PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                          "Complete hook of %s: change Data copy %p to version %d at %s:%d",
                          parsec_task_snprintf(tmp, 128, (parsec_task_t*)(this_task)),
                          this_task->data._f_GRID_TO.data_out, this_task->data._f_GRID_TO.data_out->version, __FILE__, __LINE__);
  }
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  (void)x;  (void)y;  (void)z;  (void)s;  (void)conservative;  (void)direction;  (void)dimension;  (void)side;

#endif /* DISTRIBUTED */
#if defined(PARSEC_PROF_GRAPHER)
  parsec_prof_grapher_task((parsec_task_t*)this_task, es->th_id, es->virtual_process->vp_id,
     __jdf2c_key_fns_Exchange.key_hash(this_task->task_class->make_key( (parsec_taskpool_t*)this_task->taskpool, ((parsec_task_t*)this_task)->locals), NULL));
#endif  /* defined(PARSEC_PROF_GRAPHER) */
  release_deps_of_LBM_Exchange(es, this_task,
      PARSEC_ACTION_RELEASE_REMOTE_DEPS |
      PARSEC_ACTION_RELEASE_LOCAL_DEPS |
      PARSEC_ACTION_RELEASE_LOCAL_REFS |
      PARSEC_ACTION_RESHAPE_ON_RELEASE |
      0xfff,  /* mask of all dep_index */ 
      NULL);
  return PARSEC_HOOK_RETURN_DONE;
}

static parsec_hook_return_t release_task_of_LBM_Exchange(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    const __parsec_LBM_internal_taskpool_t *__parsec_tp =
        (const __parsec_LBM_internal_taskpool_t *)this_task->taskpool;
    parsec_hash_table_t *ht = (parsec_hash_table_t*)__parsec_tp->super.super.dependencies_array[2];
    parsec_key_t key = this_task->task_class->make_key((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals);
    parsec_hashable_dependency_t *hash_dep = (parsec_hashable_dependency_t *)parsec_hash_table_remove(ht, key);
    parsec_thread_mempool_free(hash_dep->mempool_owner, hash_dep);
    return parsec_release_task_to_mempool_update_nbtasks(es, this_task);
}

static char *LBM_LBM_Exchange_internal_init_deps_key_functions_key_print(char *buffer, size_t buffer_size, parsec_key_t __parsec_key_, void *user_data)
{
  uint64_t __parsec_key = (uint64_t)(uintptr_t)__parsec_key_;
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)user_data;
  int __jdf2c_x_min = 0;
  int x = (__parsec_key) % __parsec_tp->Exchange_x_range + __jdf2c_x_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_x_range;
  int __jdf2c_y_min = 0;
  int y = (__parsec_key) % __parsec_tp->Exchange_y_range + __jdf2c_y_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_y_range;
  int __jdf2c_z_min = 0;
  int z = (__parsec_key) % __parsec_tp->Exchange_z_range + __jdf2c_z_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_z_range;
  int __jdf2c_s_min = 0;
  int s = (__parsec_key) % __parsec_tp->Exchange_s_range + __jdf2c_s_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_s_range;
  int __jdf2c_conservative_min = 0;
  int conservative = (__parsec_key) % __parsec_tp->Exchange_conservative_range + __jdf2c_conservative_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_conservative_range;
  int __jdf2c_direction_min = 0;
  int direction = (__parsec_key) % __parsec_tp->Exchange_direction_range + __jdf2c_direction_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_direction_range;
  int __jdf2c_dimension_min = 0;
  int dimension = (__parsec_key) % __parsec_tp->Exchange_dimension_range + __jdf2c_dimension_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_dimension_range;
  int __jdf2c_side_min = 0;
  int side = (__parsec_key) % __parsec_tp->Exchange_side_range + __jdf2c_side_min;
  __parsec_key = __parsec_key / __parsec_tp->Exchange_side_range;
  snprintf(buffer, buffer_size, "Exchange(%d, %d, %d, %d, %d, %d, %d, %d)", x, y, z, s, conservative, direction, dimension, side);
  return buffer;
}

static parsec_key_fn_t LBM_LBM_Exchange_internal_init_deps_key_functions = {
   .key_equal = parsec_hash_table_generic_64bits_key_equal,
   .key_print = LBM_LBM_Exchange_internal_init_deps_key_functions_key_print,
   .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

/* Needs: min-max count-tasks iterate */
static int LBM_Exchange_internal_init(parsec_execution_stream_t * es, __parsec_LBM_Exchange_task_t * this_task)
{
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t*)this_task->taskpool;
  int32_t nb_tasks = 0, saved_nb_tasks = 0;
int32_t __x_min = 0x7fffffff, __x_max = 0;int32_t __jdf2c_x_min = 0x7fffffff, __jdf2c_x_max = 0;int32_t __y_min = 0x7fffffff, __y_max = 0;int32_t __jdf2c_y_min = 0x7fffffff, __jdf2c_y_max = 0;int32_t __z_min = 0x7fffffff, __z_max = 0;int32_t __jdf2c_z_min = 0x7fffffff, __jdf2c_z_max = 0;int32_t __s_min = 0x7fffffff, __s_max = 0;int32_t __jdf2c_s_min = 0x7fffffff, __jdf2c_s_max = 0;int32_t __conservative_min = 0x7fffffff, __conservative_max = 0;int32_t __jdf2c_conservative_min = 0x7fffffff, __jdf2c_conservative_max = 0;int32_t __direction_min = 0x7fffffff, __direction_max = 0;int32_t __jdf2c_direction_min = 0x7fffffff, __jdf2c_direction_max = 0;int32_t __dimension_min = 0x7fffffff, __dimension_max = 0;int32_t __jdf2c_dimension_min = 0x7fffffff, __jdf2c_dimension_max = 0;int32_t __side_min = 0x7fffffff, __side_max = 0;int32_t __jdf2c_side_min = 0x7fffffff, __jdf2c_side_max = 0;  __parsec_LBM_Exchange_parsec_assignment_t assignments = {  .x.value = 0, .y.value = 0, .z.value = 0, .s.value = 0, .conservative.value = 0, .direction.value = 0, .dimension.value = 0, .side.value = 0 };
  int32_t  x,  y,  z,  s,  conservative,  direction,  dimension,  side;
  int32_t __jdf2c_x_start, __jdf2c_x_end, __jdf2c_x_inc;
  int32_t __jdf2c_y_start, __jdf2c_y_end, __jdf2c_y_inc;
  int32_t __jdf2c_z_start, __jdf2c_z_end, __jdf2c_z_inc;
  int32_t __jdf2c_s_start, __jdf2c_s_end, __jdf2c_s_inc;
  int32_t __jdf2c_conservative_start, __jdf2c_conservative_end, __jdf2c_conservative_inc;
  int32_t __jdf2c_direction_start, __jdf2c_direction_end, __jdf2c_direction_inc;
  int32_t __jdf2c_dimension_start, __jdf2c_dimension_end, __jdf2c_dimension_inc;
  int32_t __jdf2c_side_start, __jdf2c_side_end, __jdf2c_side_inc;
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)
  PARSEC_PROFILING_TRACE(es->es_profile,
                         this_task->taskpool->profiling_array[2 * this_task->task_class->task_class_id],
                         0,
                         this_task->taskpool->taskpool_id, NULL);
#endif /* defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */
    __jdf2c_x_start = 0;
    __jdf2c_x_end = (subgrid_number_x - 1);
    __jdf2c_x_inc = 1;
    __x_min = parsec_imin(__jdf2c_x_start, __jdf2c_x_end);
    __x_max = parsec_imax(__jdf2c_x_start, __jdf2c_x_end);
    __jdf2c_x_min = parsec_imin(__jdf2c_x_min, __x_min);
    __jdf2c_x_max = parsec_imax(__jdf2c_x_max, __x_max);
    for(x =  __jdf2c_x_start;
        x <= __jdf2c_x_end;
        x += __jdf2c_x_inc) {
    assignments.x.value = x;
      __jdf2c_y_start = 0;
      __jdf2c_y_end = (subgrid_number_y - 1);
      __jdf2c_y_inc = 1;
      __y_min = parsec_imin(__jdf2c_y_start, __jdf2c_y_end);
      __y_max = parsec_imax(__jdf2c_y_start, __jdf2c_y_end);
      __jdf2c_y_min = parsec_imin(__jdf2c_y_min, __y_min);
      __jdf2c_y_max = parsec_imax(__jdf2c_y_max, __y_max);
      for(y =  __jdf2c_y_start;
          y <= __jdf2c_y_end;
          y += __jdf2c_y_inc) {
      assignments.y.value = y;
        __jdf2c_z_start = 0;
        __jdf2c_z_end = (subgrid_number_z - 1);
        __jdf2c_z_inc = 1;
        __z_min = parsec_imin(__jdf2c_z_start, __jdf2c_z_end);
        __z_max = parsec_imax(__jdf2c_z_start, __jdf2c_z_end);
        __jdf2c_z_min = parsec_imin(__jdf2c_z_min, __z_min);
        __jdf2c_z_max = parsec_imax(__jdf2c_z_max, __z_max);
        for(z =  __jdf2c_z_start;
            z <= __jdf2c_z_end;
            z += __jdf2c_z_inc) {
        assignments.z.value = z;
          __jdf2c_s_start = 0;
          __jdf2c_s_end = (number_of_steps - 2);
          __jdf2c_s_inc = 1;
          __s_min = parsec_imin(__jdf2c_s_start, __jdf2c_s_end);
          __s_max = parsec_imax(__jdf2c_s_start, __jdf2c_s_end);
          __jdf2c_s_min = parsec_imin(__jdf2c_s_min, __s_min);
          __jdf2c_s_max = parsec_imax(__jdf2c_s_max, __s_max);
          for(s =  __jdf2c_s_start;
              s <= __jdf2c_s_end;
              s += __jdf2c_s_inc) {
          assignments.s.value = s;
            __jdf2c_conservative_start = 0;
            __jdf2c_conservative_end = (conservatives_number - 1);
            __jdf2c_conservative_inc = 1;
            __conservative_min = parsec_imin(__jdf2c_conservative_start, __jdf2c_conservative_end);
            __conservative_max = parsec_imax(__jdf2c_conservative_start, __jdf2c_conservative_end);
            __jdf2c_conservative_min = parsec_imin(__jdf2c_conservative_min, __conservative_min);
            __jdf2c_conservative_max = parsec_imax(__jdf2c_conservative_max, __conservative_max);
            for(conservative =  __jdf2c_conservative_start;
                conservative <= __jdf2c_conservative_end;
                conservative += __jdf2c_conservative_inc) {
            assignments.conservative.value = conservative;
              __jdf2c_direction_start = 0;
              __jdf2c_direction_end = (directions_number - 1);
              __jdf2c_direction_inc = 1;
              __direction_min = parsec_imin(__jdf2c_direction_start, __jdf2c_direction_end);
              __direction_max = parsec_imax(__jdf2c_direction_start, __jdf2c_direction_end);
              __jdf2c_direction_min = parsec_imin(__jdf2c_direction_min, __direction_min);
              __jdf2c_direction_max = parsec_imax(__jdf2c_direction_max, __direction_max);
              for(direction =  __jdf2c_direction_start;
                  direction <= __jdf2c_direction_end;
                  direction += __jdf2c_direction_inc) {
              assignments.direction.value = direction;
                __jdf2c_dimension_start = 0;
                __jdf2c_dimension_end = (3 - 1);
                __jdf2c_dimension_inc = 1;
                __dimension_min = parsec_imin(__jdf2c_dimension_start, __jdf2c_dimension_end);
                __dimension_max = parsec_imax(__jdf2c_dimension_start, __jdf2c_dimension_end);
                __jdf2c_dimension_min = parsec_imin(__jdf2c_dimension_min, __dimension_min);
                __jdf2c_dimension_max = parsec_imax(__jdf2c_dimension_max, __dimension_max);
                for(dimension =  __jdf2c_dimension_start;
                    dimension <= __jdf2c_dimension_end;
                    dimension += __jdf2c_dimension_inc) {
                assignments.dimension.value = dimension;
                  __jdf2c_side_start = 0;
                  __jdf2c_side_end = 1;
                  __jdf2c_side_inc = 1;
                  __side_min = parsec_imin(__jdf2c_side_start, __jdf2c_side_end);
                  __side_max = parsec_imax(__jdf2c_side_start, __jdf2c_side_end);
                  __jdf2c_side_min = parsec_imin(__jdf2c_side_min, __side_min);
                  __jdf2c_side_max = parsec_imax(__jdf2c_side_max, __side_max);
                  for(side =  __jdf2c_side_start;
                      side <= __jdf2c_side_end;
                      side += __jdf2c_side_inc) {
                  assignments.side.value = side;
                  if( !Exchange_pred(x, y, z, s, conservative, direction, dimension, side) ) continue;
                  nb_tasks++;
                } /* Loop on normal range side */
              } /* For loop of dimension */ 
            } /* For loop of direction */ 
          } /* For loop of conservative */ 
        } /* For loop of s */ 
      } /* For loop of z */ 
    } /* For loop of y */ 
  } /* For loop of x */ 
   if( 0 != nb_tasks ) {
     (void)parsec_atomic_fetch_add_int32(&__parsec_tp->initial_number_tasks, nb_tasks);
   }
  /* Set the range variables for the collision-free hash-computation */
  __parsec_tp->Exchange_x_range = (__jdf2c_x_max - __jdf2c_x_min) + 1;
  __parsec_tp->Exchange_y_range = (__jdf2c_y_max - __jdf2c_y_min) + 1;
  __parsec_tp->Exchange_z_range = (__jdf2c_z_max - __jdf2c_z_min) + 1;
  __parsec_tp->Exchange_s_range = (__jdf2c_s_max - __jdf2c_s_min) + 1;
  __parsec_tp->Exchange_conservative_range = (__jdf2c_conservative_max - __jdf2c_conservative_min) + 1;
  __parsec_tp->Exchange_direction_range = (__jdf2c_direction_max - __jdf2c_direction_min) + 1;
  __parsec_tp->Exchange_dimension_range = (__jdf2c_dimension_max - __jdf2c_dimension_min) + 1;
  __parsec_tp->Exchange_side_range = (__jdf2c_side_max - __jdf2c_side_min) + 1;
  this_task->status = PARSEC_TASK_STATUS_COMPLETE;

  PARSEC_AYU_REGISTER_TASK(&LBM_Exchange);
  __parsec_tp->super.super.dependencies_array[2] = PARSEC_OBJ_NEW(parsec_hash_table_t);
  parsec_hash_table_init(__parsec_tp->super.super.dependencies_array[2], offsetof(parsec_hashable_dependency_t, ht_item), 10, LBM_LBM_Exchange_internal_init_deps_key_functions, this_task->taskpool);
  __parsec_tp->repositories[2] = data_repo_create_nothreadsafe(nb_tasks, __jdf2c_key_fns_Exchange, (parsec_taskpool_t*)__parsec_tp, 1);
(void)saved_nb_tasks;
(void)__x_min; (void)__x_max;(void)__y_min; (void)__y_max;(void)__z_min; (void)__z_max;(void)__s_min; (void)__s_max;(void)__conservative_min; (void)__conservative_max;(void)__direction_min; (void)__direction_max;(void)__dimension_min; (void)__dimension_max;(void)__side_min; (void)__side_max;  (void)__jdf2c_x_start; (void)__jdf2c_x_end; (void)__jdf2c_x_inc;  (void)__jdf2c_y_start; (void)__jdf2c_y_end; (void)__jdf2c_y_inc;  (void)__jdf2c_z_start; (void)__jdf2c_z_end; (void)__jdf2c_z_inc;  (void)__jdf2c_s_start; (void)__jdf2c_s_end; (void)__jdf2c_s_inc;  (void)__jdf2c_conservative_start; (void)__jdf2c_conservative_end; (void)__jdf2c_conservative_inc;  (void)__jdf2c_direction_start; (void)__jdf2c_direction_end; (void)__jdf2c_direction_inc;  (void)__jdf2c_dimension_start; (void)__jdf2c_dimension_end; (void)__jdf2c_dimension_inc;  (void)__jdf2c_side_start; (void)__jdf2c_side_end; (void)__jdf2c_side_inc;  (void)assignments; (void)__parsec_tp; (void)es;
  if(1 == parsec_atomic_fetch_dec_int32(&__parsec_tp->sync_point)) {
    /* Last initialization task complete. Update the number of tasks. */
    __parsec_tp->super.super.tdm.module->taskpool_addto_nb_tasks(&__parsec_tp->super.super, __parsec_tp->initial_number_tasks);
    parsec_mfence();  /* write memory barrier to guarantee that the scheduler gets the correct number of tasks */
    parsec_taskpool_enable((parsec_taskpool_t*)__parsec_tp, &__parsec_tp->startup_queue,
                           (parsec_task_t*)this_task, es, __parsec_tp->super.super.nb_pending_actions);
    __parsec_tp->super.super.tdm.module->taskpool_ready(&__parsec_tp->super.super);
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)
    PARSEC_PROFILING_TRACE(es->es_profile,
                           this_task->taskpool->profiling_array[2 * this_task->task_class->task_class_id + 1],
                           0,
                           this_task->taskpool->taskpool_id, NULL);
#endif /* defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */
    return PARSEC_HOOK_RETURN_DONE;
  }
  return PARSEC_HOOK_RETURN_DONE;
}

static const __parsec_chore_t __LBM_Exchange_chores[] ={
#if defined(PARSEC_HAVE_CUDA)
    { .type     = PARSEC_DEV_CUDA,
      .dyld     = NULL,
      .evaluate = NULL,
      .hook     = (parsec_hook_t*)hook_of_LBM_Exchange_CUDA },
#endif  /* defined(PARSEC_HAVE_CUDA) */
    { .type     = PARSEC_DEV_NONE,
      .evaluate = NULL,
      .hook     = (parsec_hook_t*)NULL },  /* End marker */
};

static const parsec_task_class_t LBM_Exchange = {
  .name = "Exchange",
  .task_class_id = 2,
  .nb_flows = 1,
  .nb_parameters = 8,
  .nb_locals = 8,
  .task_class_type = PARSEC_TASK_CLASS_TYPE_PTG,
  .params = { &symb_LBM_Exchange_x, &symb_LBM_Exchange_y, &symb_LBM_Exchange_z, &symb_LBM_Exchange_s, &symb_LBM_Exchange_conservative, &symb_LBM_Exchange_direction, &symb_LBM_Exchange_dimension, &symb_LBM_Exchange_side, NULL },
  .locals = { &symb_LBM_Exchange_x, &symb_LBM_Exchange_y, &symb_LBM_Exchange_z, &symb_LBM_Exchange_s, &symb_LBM_Exchange_conservative, &symb_LBM_Exchange_direction, &symb_LBM_Exchange_dimension, &symb_LBM_Exchange_side, NULL },
  .data_affinity = (parsec_data_ref_fn_t*)affinity_of_LBM_Exchange,
  .initial_data = (parsec_data_ref_fn_t*)affinity_of_LBM_Exchange,
  .final_data = (parsec_data_ref_fn_t*)affinity_of_LBM_Exchange,
  .priority = NULL,
  .properties = properties_of_LBM_Exchange,
#if MAX_PARAM_COUNT < 1  /* number of read flows of Exchange */
  #error Too many read flows for task Exchange
#endif  /* MAX_PARAM_COUNT */
#if MAX_PARAM_COUNT < 1  /* number of write flows of Exchange */
  #error Too many write flows for task Exchange
#endif  /* MAX_PARAM_COUNT */
  .in = { &flow_of_LBM_Exchange_for_GRID_TO, NULL },
  .out = { &flow_of_LBM_Exchange_for_GRID_TO, NULL },
  .flags = 0x0 | PARSEC_USE_DEPS_MASK,
  .dependencies_goal = 0x1,
  .make_key = __jdf2c_make_key_Exchange,
  .task_snprintf = parsec_task_snprintf,
  .key_functions = &__jdf2c_key_fns_Exchange,
  .fini = (parsec_hook_t*)NULL,
  .incarnations = __LBM_Exchange_chores,
  .find_deps = parsec_hash_find_deps,
  .update_deps = parsec_update_deps_with_mask,
  .iterate_successors = (parsec_traverse_function_t*)iterate_successors_of_LBM_Exchange,
  .iterate_predecessors = (parsec_traverse_function_t*)iterate_predecessors_of_LBM_Exchange,
  .release_deps = (parsec_release_deps_t*)release_deps_of_LBM_Exchange,
  .prepare_output = (parsec_hook_t*)NULL,
  .prepare_input = (parsec_hook_t*)data_lookup_of_LBM_Exchange,
  .get_datatype = (parsec_datatype_lookup_t*)datatype_lookup_of_LBM_Exchange,
  .complete_execution = (parsec_hook_t*)complete_hook_of_LBM_Exchange,
  .release_task = &release_task_of_LBM_Exchange,
#if defined(PARSEC_SIM)
  .sim_cost_fct = (parsec_sim_cost_fct_t*)NULL,
#endif
};


/******                                    LBM_STEP                                    ******/

static inline int32_t minexpr_of_symb_LBM_LBM_STEP_x_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_LBM_STEP_x = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_LBM_STEP_x_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_LBM_STEP_x_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (subgrid_number_x - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_LBM_STEP_x = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_LBM_STEP_x_fct }
                   }
};
static const parsec_symbol_t symb_LBM_LBM_STEP_x = { .name = "x", .context_index = 0, .min = &minexpr_of_symb_LBM_LBM_STEP_x, .max = &maxexpr_of_symb_LBM_LBM_STEP_x, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_LBM_STEP_y_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_LBM_STEP_y = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_LBM_STEP_y_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_LBM_STEP_y_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (subgrid_number_y - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_LBM_STEP_y = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_LBM_STEP_y_fct }
                   }
};
static const parsec_symbol_t symb_LBM_LBM_STEP_y = { .name = "y", .context_index = 1, .min = &minexpr_of_symb_LBM_LBM_STEP_y, .max = &maxexpr_of_symb_LBM_LBM_STEP_y, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_LBM_STEP_z_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_LBM_STEP_z = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_LBM_STEP_z_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_LBM_STEP_z_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (subgrid_number_z - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_LBM_STEP_z = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_LBM_STEP_z_fct }
                   }
};
static const parsec_symbol_t symb_LBM_LBM_STEP_z = { .name = "z", .context_index = 2, .min = &minexpr_of_symb_LBM_LBM_STEP_z, .max = &maxexpr_of_symb_LBM_LBM_STEP_z, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_LBM_STEP_s_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_LBM_STEP_s = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_LBM_STEP_s_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_LBM_STEP_s_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (number_of_steps - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_LBM_STEP_s = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_LBM_STEP_s_fct }
                   }
};
static const parsec_symbol_t symb_LBM_LBM_STEP_s = { .name = "s", .context_index = 3, .min = &minexpr_of_symb_LBM_LBM_STEP_s, .max = &maxexpr_of_symb_LBM_LBM_STEP_s, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int affinity_of_LBM_LBM_STEP(__parsec_LBM_LBM_STEP_task_t *this_task,
                     parsec_data_ref_t *ref)
{
    const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t*)this_task->taskpool;
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int s = this_task->locals.s.value; (void)s;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)x;
  (void)y;
  (void)z;
  (void)s;
  ref->dc = (parsec_data_collection_t *)__parsec_tp->super._g_descGridDC;
  /* Compute data key */
  ref->key = ref->dc->data_key(ref->dc, x, y, z, 0, 0);
  return 1;
}
static const parsec_property_t properties_of_LBM_LBM_STEP[1] = {
  {.name = NULL, .expr = NULL}
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep1_atline_314_iftrue_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (s) == (0);
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep1_atline_314_iftrue = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep1_atline_314_iftrue_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep1_atline_314_iftrue = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep1_atline_314_iftrue,  /* (s) == (0) */
  .ctl_gather_nb = NULL,
  .task_class_id = 0, /* LBM_FillGrid */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_FillGrid_for_INITIAL_GRID,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep1_atline_314_iffalse_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return !((s) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep1_atline_314_iffalse = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep1_atline_314_iffalse_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep1_atline_314_iffalse = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep1_atline_314_iffalse,  /* !(s) == (0) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep2_atline_315_iftrue_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (s) == ((number_of_steps - 1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep2_atline_315_iftrue = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep2_atline_315_iftrue_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep2_atline_315_iftrue = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep2_atline_315_iftrue,  /* (s) == ((number_of_steps - 1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 3, /* LBM_WriteBack */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_WriteBack_for_FINAL_GRID,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep2_atline_315_iffalse_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return !((s) == ((number_of_steps - 1)));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep2_atline_315_iffalse = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep2_atline_315_iffalse_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep2_atline_315_iffalse = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep2_atline_315_iffalse,  /* !(s) == ((number_of_steps - 1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0,
};
#if MAX_DEP_IN_COUNT < 1  /* number of input dependencies */
    #error Too many input dependencies (supports up to MAX_DEP_IN_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_IN_COUNT.
#endif
#if MAX_DEP_OUT_COUNT < 1  /* number of output dependencies */
    #error Too many output dependencies (supports up to MAX_DEP_OUT_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_OUT_COUNT.
#endif

static const parsec_flow_t flow_of_LBM_LBM_STEP_for_GRID_CD_0_0 = {
  .name               = "GRID_CD_0_0",
  .sym_type           = PARSEC_SYM_INOUT,
  .flow_flags         = PARSEC_FLOW_ACCESS_RW,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep1_atline_314_iftrue,
 &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep1_atline_314_iffalse },
  .dep_out    = { &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep2_atline_315_iftrue,
 &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep2_atline_315_iffalse }
};

static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep1_atline_317_iftrue_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (s) == (0);
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep1_atline_317_iftrue = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep1_atline_317_iftrue_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep1_atline_317_iftrue = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep1_atline_317_iftrue,  /* (s) == (0) */
  .ctl_gather_nb = NULL,
  .task_class_id = 0, /* LBM_FillGrid */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_FillGrid_for_INITIAL_GRID,
  .dep_index = 1,
  .dep_datatype_index = 1,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep1_atline_317_iffalse_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return !((s) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep1_atline_317_iffalse = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep1_atline_317_iffalse_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep1_atline_317_iffalse = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep1_atline_317_iffalse,  /* !(s) == (0) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 1,
  .dep_datatype_index = 1,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep2_atline_318_iftrue_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (s) == ((number_of_steps - 1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep2_atline_318_iftrue = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep2_atline_318_iftrue_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep2_atline_318_iftrue = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep2_atline_318_iftrue,  /* (s) == ((number_of_steps - 1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 3, /* LBM_WriteBack */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_WriteBack_for_FINAL_GRID,
  .dep_index = 1,
  .dep_datatype_index = 1,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep2_atline_318_iffalse_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return !((s) == ((number_of_steps - 1)));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep2_atline_318_iffalse = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep2_atline_318_iffalse_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep2_atline_318_iffalse = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep2_atline_318_iffalse,  /* !(s) == ((number_of_steps - 1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 1,
  .dep_datatype_index = 1,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0,
};
#if MAX_DEP_IN_COUNT < 1  /* number of input dependencies */
    #error Too many input dependencies (supports up to MAX_DEP_IN_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_IN_COUNT.
#endif
#if MAX_DEP_OUT_COUNT < 1  /* number of output dependencies */
    #error Too many output dependencies (supports up to MAX_DEP_OUT_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_OUT_COUNT.
#endif

static const parsec_flow_t flow_of_LBM_LBM_STEP_for_GRID_CD_1_0 = {
  .name               = "GRID_CD_1_0",
  .sym_type           = PARSEC_SYM_INOUT,
  .flow_flags         = PARSEC_FLOW_ACCESS_RW,
  .flow_index         = 1,
  .flow_datatype_mask = 0x2,
  .dep_in     = { &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep1_atline_317_iftrue,
 &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep1_atline_317_iffalse },
  .dep_out    = { &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep2_atline_318_iftrue,
 &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep2_atline_318_iffalse }
};

static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep1_atline_320_iftrue_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (s) == (0);
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep1_atline_320_iftrue = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep1_atline_320_iftrue_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep1_atline_320_iftrue = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep1_atline_320_iftrue,  /* (s) == (0) */
  .ctl_gather_nb = NULL,
  .task_class_id = 0, /* LBM_FillGrid */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_FillGrid_for_INITIAL_GRID,
  .dep_index = 2,
  .dep_datatype_index = 2,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep1_atline_320_iffalse_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return !((s) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep1_atline_320_iffalse = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep1_atline_320_iffalse_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep1_atline_320_iffalse = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep1_atline_320_iffalse,  /* !(s) == (0) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 2,
  .dep_datatype_index = 2,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep2_atline_321_iftrue_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (s) == ((number_of_steps - 1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep2_atline_321_iftrue = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep2_atline_321_iftrue_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep2_atline_321_iftrue = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep2_atline_321_iftrue,  /* (s) == ((number_of_steps - 1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 3, /* LBM_WriteBack */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_WriteBack_for_FINAL_GRID,
  .dep_index = 2,
  .dep_datatype_index = 2,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep2_atline_321_iffalse_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return !((s) == ((number_of_steps - 1)));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep2_atline_321_iffalse = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep2_atline_321_iffalse_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep2_atline_321_iffalse = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep2_atline_321_iffalse,  /* !(s) == ((number_of_steps - 1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 2,
  .dep_datatype_index = 2,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0,
};
#if MAX_DEP_IN_COUNT < 1  /* number of input dependencies */
    #error Too many input dependencies (supports up to MAX_DEP_IN_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_IN_COUNT.
#endif
#if MAX_DEP_OUT_COUNT < 1  /* number of output dependencies */
    #error Too many output dependencies (supports up to MAX_DEP_OUT_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_OUT_COUNT.
#endif

static const parsec_flow_t flow_of_LBM_LBM_STEP_for_GRID_CD_2_0 = {
  .name               = "GRID_CD_2_0",
  .sym_type           = PARSEC_SYM_INOUT,
  .flow_flags         = PARSEC_FLOW_ACCESS_RW,
  .flow_index         = 2,
  .flow_datatype_mask = 0x4,
  .dep_in     = { &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep1_atline_320_iftrue,
 &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep1_atline_320_iffalse },
  .dep_out    = { &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep2_atline_321_iftrue,
 &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep2_atline_321_iffalse }
};

static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep1_atline_323_iftrue_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (s) == (0);
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep1_atline_323_iftrue = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep1_atline_323_iftrue_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep1_atline_323_iftrue = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep1_atline_323_iftrue,  /* (s) == (0) */
  .ctl_gather_nb = NULL,
  .task_class_id = 0, /* LBM_FillGrid */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_FillGrid_for_INITIAL_GRID,
  .dep_index = 3,
  .dep_datatype_index = 3,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep1_atline_323_iffalse_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return !((s) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep1_atline_323_iffalse = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep1_atline_323_iffalse_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep1_atline_323_iffalse = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep1_atline_323_iffalse,  /* !(s) == (0) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 3,
  .dep_datatype_index = 3,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep2_atline_324_iftrue_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (s) == ((number_of_steps - 1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep2_atline_324_iftrue = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep2_atline_324_iftrue_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep2_atline_324_iftrue = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep2_atline_324_iftrue,  /* (s) == ((number_of_steps - 1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 3, /* LBM_WriteBack */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_WriteBack_for_FINAL_GRID,
  .dep_index = 3,
  .dep_datatype_index = 3,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep2_atline_324_iffalse_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return !((s) == ((number_of_steps - 1)));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep2_atline_324_iffalse = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep2_atline_324_iffalse_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep2_atline_324_iffalse = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep2_atline_324_iffalse,  /* !(s) == ((number_of_steps - 1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 3,
  .dep_datatype_index = 3,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1,
};
#if MAX_DEP_IN_COUNT < 1  /* number of input dependencies */
    #error Too many input dependencies (supports up to MAX_DEP_IN_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_IN_COUNT.
#endif
#if MAX_DEP_OUT_COUNT < 1  /* number of output dependencies */
    #error Too many output dependencies (supports up to MAX_DEP_OUT_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_OUT_COUNT.
#endif

static const parsec_flow_t flow_of_LBM_LBM_STEP_for_GRID_CD_0_1 = {
  .name               = "GRID_CD_0_1",
  .sym_type           = PARSEC_SYM_INOUT,
  .flow_flags         = PARSEC_FLOW_ACCESS_RW,
  .flow_index         = 3,
  .flow_datatype_mask = 0x8,
  .dep_in     = { &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep1_atline_323_iftrue,
 &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep1_atline_323_iffalse },
  .dep_out    = { &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep2_atline_324_iftrue,
 &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep2_atline_324_iffalse }
};

static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep1_atline_326_iftrue_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (s) == (0);
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep1_atline_326_iftrue = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep1_atline_326_iftrue_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep1_atline_326_iftrue = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep1_atline_326_iftrue,  /* (s) == (0) */
  .ctl_gather_nb = NULL,
  .task_class_id = 0, /* LBM_FillGrid */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_FillGrid_for_INITIAL_GRID,
  .dep_index = 4,
  .dep_datatype_index = 4,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep1_atline_326_iffalse_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return !((s) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep1_atline_326_iffalse = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep1_atline_326_iffalse_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep1_atline_326_iffalse = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep1_atline_326_iffalse,  /* !(s) == (0) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 4,
  .dep_datatype_index = 4,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep2_atline_327_iftrue_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (s) == ((number_of_steps - 1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep2_atline_327_iftrue = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep2_atline_327_iftrue_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep2_atline_327_iftrue = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep2_atline_327_iftrue,  /* (s) == ((number_of_steps - 1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 3, /* LBM_WriteBack */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_WriteBack_for_FINAL_GRID,
  .dep_index = 4,
  .dep_datatype_index = 4,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep2_atline_327_iffalse_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return !((s) == ((number_of_steps - 1)));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep2_atline_327_iffalse = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep2_atline_327_iffalse_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep2_atline_327_iffalse = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep2_atline_327_iffalse,  /* !(s) == ((number_of_steps - 1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 4,
  .dep_datatype_index = 4,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1,
};
#if MAX_DEP_IN_COUNT < 1  /* number of input dependencies */
    #error Too many input dependencies (supports up to MAX_DEP_IN_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_IN_COUNT.
#endif
#if MAX_DEP_OUT_COUNT < 1  /* number of output dependencies */
    #error Too many output dependencies (supports up to MAX_DEP_OUT_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_OUT_COUNT.
#endif

static const parsec_flow_t flow_of_LBM_LBM_STEP_for_GRID_CD_1_1 = {
  .name               = "GRID_CD_1_1",
  .sym_type           = PARSEC_SYM_INOUT,
  .flow_flags         = PARSEC_FLOW_ACCESS_RW,
  .flow_index         = 4,
  .flow_datatype_mask = 0x10,
  .dep_in     = { &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep1_atline_326_iftrue,
 &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep1_atline_326_iffalse },
  .dep_out    = { &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep2_atline_327_iftrue,
 &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep2_atline_327_iffalse }
};

static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep1_atline_329_iftrue_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (s) == (0);
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep1_atline_329_iftrue = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep1_atline_329_iftrue_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep1_atline_329_iftrue = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep1_atline_329_iftrue,  /* (s) == (0) */
  .ctl_gather_nb = NULL,
  .task_class_id = 0, /* LBM_FillGrid */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_FillGrid_for_INITIAL_GRID,
  .dep_index = 5,
  .dep_datatype_index = 5,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep1_atline_329_iffalse_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return !((s) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep1_atline_329_iffalse = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep1_atline_329_iffalse_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep1_atline_329_iffalse = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep1_atline_329_iffalse,  /* !(s) == (0) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 5,
  .dep_datatype_index = 5,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep2_atline_330_iftrue_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (s) == ((number_of_steps - 1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep2_atline_330_iftrue = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep2_atline_330_iftrue_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep2_atline_330_iftrue = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep2_atline_330_iftrue,  /* (s) == ((number_of_steps - 1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 3, /* LBM_WriteBack */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_WriteBack_for_FINAL_GRID,
  .dep_index = 5,
  .dep_datatype_index = 5,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep2_atline_330_iffalse_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return !((s) == ((number_of_steps - 1)));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep2_atline_330_iffalse = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep2_atline_330_iffalse_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep2_atline_330_iffalse = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep2_atline_330_iffalse,  /* !(s) == ((number_of_steps - 1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 2, /* LBM_Exchange */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_Exchange_for_GRID_TO,
  .dep_index = 5,
  .dep_datatype_index = 5,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1,
};
#if MAX_DEP_IN_COUNT < 1  /* number of input dependencies */
    #error Too many input dependencies (supports up to MAX_DEP_IN_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_IN_COUNT.
#endif
#if MAX_DEP_OUT_COUNT < 1  /* number of output dependencies */
    #error Too many output dependencies (supports up to MAX_DEP_OUT_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_OUT_COUNT.
#endif

static const parsec_flow_t flow_of_LBM_LBM_STEP_for_GRID_CD_2_1 = {
  .name               = "GRID_CD_2_1",
  .sym_type           = PARSEC_SYM_INOUT,
  .flow_flags         = PARSEC_FLOW_ACCESS_RW,
  .flow_index         = 5,
  .flow_datatype_mask = 0x20,
  .dep_in     = { &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep1_atline_329_iftrue,
 &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep1_atline_329_iffalse },
  .dep_out    = { &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep2_atline_330_iftrue,
 &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep2_atline_330_iffalse }
};

static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_X_dep1_atline_332_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (s) != (0);
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_X_dep1_atline_332 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_X_dep1_atline_332_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_X_dep1_atline_332 = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_X_dep1_atline_332,  /* (s) != (0) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_X,
  .dep_index = 6,
  .dep_datatype_index = 6,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_X,
};
static inline int expr_of_cond_for_flow_of_LBM_LBM_STEP_for_X_dep2_atline_333_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_LBM_STEP_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int s = locals->s.value; (void)s;

  (void)x;
  (void)y;
  (void)z;
  (void)s;
  (void)__parsec_tp; (void)locals;
  return (s) != ((number_of_steps - 1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_LBM_STEP_for_X_dep2_atline_333 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_LBM_STEP_for_X_dep2_atline_333_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_LBM_STEP_for_X_dep2_atline_333 = {
  .cond = &expr_of_cond_for_flow_of_LBM_LBM_STEP_for_X_dep2_atline_333,  /* (s) != ((number_of_steps - 1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_X,
  .dep_index = 6,
  .dep_datatype_index = 6,
  .belongs_to = &flow_of_LBM_LBM_STEP_for_X,
};
#if MAX_DEP_IN_COUNT < 1  /* number of input dependencies */
    #error Too many input dependencies (supports up to MAX_DEP_IN_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_IN_COUNT.
#endif
#if MAX_DEP_OUT_COUNT < 1  /* number of output dependencies */
    #error Too many output dependencies (supports up to MAX_DEP_OUT_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_OUT_COUNT.
#endif

static const parsec_flow_t flow_of_LBM_LBM_STEP_for_X = {
  .name               = "X",
  .sym_type           = PARSEC_SYM_INOUT,
  .flow_flags         = PARSEC_FLOW_ACCESS_NONE|PARSEC_FLOW_HAS_IN_DEPS,
  .flow_index         = 6,
  .flow_datatype_mask = 0x40,
  .dep_in     = { &flow_of_LBM_LBM_STEP_for_X_dep1_atline_332 },
  .dep_out    = { &flow_of_LBM_LBM_STEP_for_X_dep2_atline_333 }
};

static void
iterate_successors_of_LBM_LBM_STEP(parsec_execution_stream_t *es, const __parsec_LBM_LBM_STEP_task_t *this_task,
               uint32_t action_mask, parsec_ontask_function_t *ontask, void *ontask_arg)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t*)this_task->taskpool;
  parsec_task_t nc;  /* generic placeholder for locals */
  parsec_dep_data_description_t data;
  __parsec_LBM_LBM_STEP_parsec_assignment_t __jdf2c__tmp_locals = *(__parsec_LBM_LBM_STEP_parsec_assignment_t*)&this_task->locals;   /* copy of this_task locals in R/W mode to manage local definitions */
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int x = __jdf2c__tmp_locals.x.value; (void)x;
  const int y = __jdf2c__tmp_locals.y.value; (void)y;
  const int z = __jdf2c__tmp_locals.z.value; (void)z;
  const int s = __jdf2c__tmp_locals.s.value; (void)s;
  (void)rank_src; (void)rank_dst; (void)__parsec_tp; (void)vpid_dst;
   data_repo_t *successor_repo; parsec_key_t successor_repo_key;  (void)x;  (void)y;  (void)z;  (void)s;
  nc.taskpool  = this_task->taskpool;
  nc.priority  = this_task->priority;
  nc.chore_mask  = PARSEC_DEV_ALL;
#if defined(DISTRIBUTED)
  rank_src = rank_of_descGridDC(x, y, z, 0, 0);
#endif
  if( action_mask & 0x1 ) {  /* Flow of data GRID_CD_0_0 [0] */
    data.data   = this_task->data._f_GRID_CD_0_0.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
      if( (s) == ((number_of_steps - 1)) ) {
      __parsec_LBM_WriteBack_task_t* ncc = (__parsec_LBM_WriteBack_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_WriteBack.task_class_id];
        const int WriteBack_x = x;
        if( (WriteBack_x >= (0)) && (WriteBack_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = WriteBack_x;
          const int WriteBack_y = y;
          if( (WriteBack_y >= (0)) && (WriteBack_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = WriteBack_y;
            const int WriteBack_z = z;
            if( (WriteBack_z >= (0)) && (WriteBack_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = WriteBack_z;
              const int WriteBack_c = 0;
              if( (WriteBack_c >= (0)) && (WriteBack_c <= ((conservatives_number - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.c.value);
                ncc->locals.c.value = WriteBack_c;
                const int WriteBack_d = 0;
                if( (WriteBack_d >= (0)) && (WriteBack_d <= ((directions_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.d.value);
                  ncc->locals.d.value = WriteBack_d;
#if defined(DISTRIBUTED)
                  rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                    vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  nc.priority = __parsec_tp->super.super.priority;
                  successor_repo = WriteBack_repo;
                  successor_repo_key = __jdf2c_make_key_WriteBack((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                RELEASE_DEP_OUTPUT(es, "GRID_CD_0_0", this_task, "FINAL_GRID", &nc, rank_src, rank_dst, &data);
                if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep2_atline_315_iftrue, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                  }
                }
              }
            }
          }
    } else {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = 0;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = 0;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = 0;
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = 0;
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_CD_0_0", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep2_atline_315_iffalse, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
  if( action_mask & 0x2 ) {  /* Flow of data GRID_CD_1_0 [1] */
    data.data   = this_task->data._f_GRID_CD_1_0.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
      if( (s) == ((number_of_steps - 1)) ) {
      __parsec_LBM_WriteBack_task_t* ncc = (__parsec_LBM_WriteBack_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_WriteBack.task_class_id];
        const int WriteBack_x = x;
        if( (WriteBack_x >= (0)) && (WriteBack_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = WriteBack_x;
          const int WriteBack_y = y;
          if( (WriteBack_y >= (0)) && (WriteBack_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = WriteBack_y;
            const int WriteBack_z = z;
            if( (WriteBack_z >= (0)) && (WriteBack_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = WriteBack_z;
              const int WriteBack_c = 1;
              if( (WriteBack_c >= (0)) && (WriteBack_c <= ((conservatives_number - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.c.value);
                ncc->locals.c.value = WriteBack_c;
                const int WriteBack_d = 0;
                if( (WriteBack_d >= (0)) && (WriteBack_d <= ((directions_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.d.value);
                  ncc->locals.d.value = WriteBack_d;
#if defined(DISTRIBUTED)
                  rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                    vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  nc.priority = __parsec_tp->super.super.priority;
                  successor_repo = WriteBack_repo;
                  successor_repo_key = __jdf2c_make_key_WriteBack((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                RELEASE_DEP_OUTPUT(es, "GRID_CD_1_0", this_task, "FINAL_GRID", &nc, rank_src, rank_dst, &data);
                if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep2_atline_318_iftrue, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                  }
                }
              }
            }
          }
    } else {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = 1;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = 0;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = 0;
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = 0;
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_CD_1_0", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep2_atline_318_iffalse, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
  if( action_mask & 0x4 ) {  /* Flow of data GRID_CD_2_0 [2] */
    data.data   = this_task->data._f_GRID_CD_2_0.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
      if( (s) == ((number_of_steps - 1)) ) {
      __parsec_LBM_WriteBack_task_t* ncc = (__parsec_LBM_WriteBack_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_WriteBack.task_class_id];
        const int WriteBack_x = x;
        if( (WriteBack_x >= (0)) && (WriteBack_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = WriteBack_x;
          const int WriteBack_y = y;
          if( (WriteBack_y >= (0)) && (WriteBack_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = WriteBack_y;
            const int WriteBack_z = z;
            if( (WriteBack_z >= (0)) && (WriteBack_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = WriteBack_z;
              const int WriteBack_c = 2;
              if( (WriteBack_c >= (0)) && (WriteBack_c <= ((conservatives_number - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.c.value);
                ncc->locals.c.value = WriteBack_c;
                const int WriteBack_d = 0;
                if( (WriteBack_d >= (0)) && (WriteBack_d <= ((directions_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.d.value);
                  ncc->locals.d.value = WriteBack_d;
#if defined(DISTRIBUTED)
                  rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                    vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  nc.priority = __parsec_tp->super.super.priority;
                  successor_repo = WriteBack_repo;
                  successor_repo_key = __jdf2c_make_key_WriteBack((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                RELEASE_DEP_OUTPUT(es, "GRID_CD_2_0", this_task, "FINAL_GRID", &nc, rank_src, rank_dst, &data);
                if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep2_atline_321_iftrue, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                  }
                }
              }
            }
          }
    } else {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = 2;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = 0;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = 0;
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = 0;
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_CD_2_0", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep2_atline_321_iffalse, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
  if( action_mask & 0x8 ) {  /* Flow of data GRID_CD_0_1 [3] */
    data.data   = this_task->data._f_GRID_CD_0_1.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
      if( (s) == ((number_of_steps - 1)) ) {
      __parsec_LBM_WriteBack_task_t* ncc = (__parsec_LBM_WriteBack_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_WriteBack.task_class_id];
        const int WriteBack_x = x;
        if( (WriteBack_x >= (0)) && (WriteBack_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = WriteBack_x;
          const int WriteBack_y = y;
          if( (WriteBack_y >= (0)) && (WriteBack_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = WriteBack_y;
            const int WriteBack_z = z;
            if( (WriteBack_z >= (0)) && (WriteBack_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = WriteBack_z;
              const int WriteBack_c = 0;
              if( (WriteBack_c >= (0)) && (WriteBack_c <= ((conservatives_number - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.c.value);
                ncc->locals.c.value = WriteBack_c;
                const int WriteBack_d = 1;
                if( (WriteBack_d >= (0)) && (WriteBack_d <= ((directions_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.d.value);
                  ncc->locals.d.value = WriteBack_d;
#if defined(DISTRIBUTED)
                  rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                    vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  nc.priority = __parsec_tp->super.super.priority;
                  successor_repo = WriteBack_repo;
                  successor_repo_key = __jdf2c_make_key_WriteBack((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                RELEASE_DEP_OUTPUT(es, "GRID_CD_0_1", this_task, "FINAL_GRID", &nc, rank_src, rank_dst, &data);
                if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep2_atline_324_iftrue, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                  }
                }
              }
            }
          }
    } else {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = 0;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = 1;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = 0;
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = 0;
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_CD_0_1", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep2_atline_324_iffalse, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
  if( action_mask & 0x10 ) {  /* Flow of data GRID_CD_1_1 [4] */
    data.data   = this_task->data._f_GRID_CD_1_1.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
      if( (s) == ((number_of_steps - 1)) ) {
      __parsec_LBM_WriteBack_task_t* ncc = (__parsec_LBM_WriteBack_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_WriteBack.task_class_id];
        const int WriteBack_x = x;
        if( (WriteBack_x >= (0)) && (WriteBack_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = WriteBack_x;
          const int WriteBack_y = y;
          if( (WriteBack_y >= (0)) && (WriteBack_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = WriteBack_y;
            const int WriteBack_z = z;
            if( (WriteBack_z >= (0)) && (WriteBack_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = WriteBack_z;
              const int WriteBack_c = 1;
              if( (WriteBack_c >= (0)) && (WriteBack_c <= ((conservatives_number - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.c.value);
                ncc->locals.c.value = WriteBack_c;
                const int WriteBack_d = 1;
                if( (WriteBack_d >= (0)) && (WriteBack_d <= ((directions_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.d.value);
                  ncc->locals.d.value = WriteBack_d;
#if defined(DISTRIBUTED)
                  rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                    vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  nc.priority = __parsec_tp->super.super.priority;
                  successor_repo = WriteBack_repo;
                  successor_repo_key = __jdf2c_make_key_WriteBack((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                RELEASE_DEP_OUTPUT(es, "GRID_CD_1_1", this_task, "FINAL_GRID", &nc, rank_src, rank_dst, &data);
                if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep2_atline_327_iftrue, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                  }
                }
              }
            }
          }
    } else {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = 1;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = 1;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = 0;
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = 0;
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_CD_1_1", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep2_atline_327_iffalse, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
  if( action_mask & 0x20 ) {  /* Flow of data GRID_CD_2_1 [5] */
    data.data   = this_task->data._f_GRID_CD_2_1.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
      if( (s) == ((number_of_steps - 1)) ) {
      __parsec_LBM_WriteBack_task_t* ncc = (__parsec_LBM_WriteBack_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_WriteBack.task_class_id];
        const int WriteBack_x = x;
        if( (WriteBack_x >= (0)) && (WriteBack_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = WriteBack_x;
          const int WriteBack_y = y;
          if( (WriteBack_y >= (0)) && (WriteBack_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = WriteBack_y;
            const int WriteBack_z = z;
            if( (WriteBack_z >= (0)) && (WriteBack_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = WriteBack_z;
              const int WriteBack_c = 2;
              if( (WriteBack_c >= (0)) && (WriteBack_c <= ((conservatives_number - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.c.value);
                ncc->locals.c.value = WriteBack_c;
                const int WriteBack_d = 1;
                if( (WriteBack_d >= (0)) && (WriteBack_d <= ((directions_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.d.value);
                  ncc->locals.d.value = WriteBack_d;
#if defined(DISTRIBUTED)
                  rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                    vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  nc.priority = __parsec_tp->super.super.priority;
                  successor_repo = WriteBack_repo;
                  successor_repo_key = __jdf2c_make_key_WriteBack((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                RELEASE_DEP_OUTPUT(es, "GRID_CD_2_1", this_task, "FINAL_GRID", &nc, rank_src, rank_dst, &data);
                if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep2_atline_330_iftrue, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                  }
                }
              }
            }
          }
    } else {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = s;
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = 2;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = 1;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = 0;
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = 0;
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_CD_2_1", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep2_atline_330_iffalse, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
  if( action_mask & 0x40 ) {  /* Flow of data X [6] */
    data.data   = this_task->data._f_X.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  =   /* Control: always empty */ 0;
    data.local.dst_count  =   /* Control: always empty */ 0;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = NULL;
    data.remote.src_datatype = PARSEC_DATATYPE_NULL;
    data.remote.dst_datatype = PARSEC_DATATYPE_NULL;
    data.remote.src_count  =   /* Control: always empty */ 0;
    data.remote.dst_count  =   /* Control: always empty */ 0;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
      if( (s) != ((number_of_steps - 1)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = (s + 1);
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "X", this_task, "X", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_X_dep2_atline_333, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
  (void)data;(void)nc;(void)es;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static void
iterate_predecessors_of_LBM_LBM_STEP(parsec_execution_stream_t *es, const __parsec_LBM_LBM_STEP_task_t *this_task,
               uint32_t action_mask, parsec_ontask_function_t *ontask, void *ontask_arg)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t*)this_task->taskpool;
  parsec_task_t nc;  /* generic placeholder for locals */
  parsec_dep_data_description_t data;
  __parsec_LBM_LBM_STEP_parsec_assignment_t __jdf2c__tmp_locals = *(__parsec_LBM_LBM_STEP_parsec_assignment_t*)&this_task->locals;   /* copy of this_task locals in R/W mode to manage local definitions */
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int x = __jdf2c__tmp_locals.x.value; (void)x;
  const int y = __jdf2c__tmp_locals.y.value; (void)y;
  const int z = __jdf2c__tmp_locals.z.value; (void)z;
  const int s = __jdf2c__tmp_locals.s.value; (void)s;
  (void)rank_src; (void)rank_dst; (void)__parsec_tp; (void)vpid_dst;
   data_repo_t *successor_repo; parsec_key_t successor_repo_key;  (void)x;  (void)y;  (void)z;  (void)s;
  nc.taskpool  = this_task->taskpool;
  nc.priority  = this_task->priority;
  nc.chore_mask  = PARSEC_DEV_ALL;
#if defined(DISTRIBUTED)
  rank_src = rank_of_descGridDC(x, y, z, 0, 0);
#endif
  if( action_mask & 0x1 ) {  /* Flow of data GRID_CD_0_0 [0] */
    data.data   = this_task->data._f_GRID_CD_0_0.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
  if( action_mask & 0x1 ) {
        if( (s) == (0) ) {
      __parsec_LBM_FillGrid_task_t* ncc = (__parsec_LBM_FillGrid_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_FillGrid.task_class_id];
        const int FillGrid_x = x;
        if( (FillGrid_x >= (0)) && (FillGrid_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = FillGrid_x;
          const int FillGrid_y = y;
          if( (FillGrid_y >= (0)) && (FillGrid_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = FillGrid_y;
            const int FillGrid_z = z;
            if( (FillGrid_z >= (0)) && (FillGrid_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = FillGrid_z;
              const int FillGrid_c = 0;
              if( (FillGrid_c >= (0)) && (FillGrid_c <= ((conservatives_number - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.c.value);
                ncc->locals.c.value = FillGrid_c;
                const int FillGrid_d = 0;
                if( (FillGrid_d >= (0)) && (FillGrid_d <= ((directions_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.d.value);
                  ncc->locals.d.value = FillGrid_d;
#if defined(DISTRIBUTED)
                  rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                    vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  nc.priority = __parsec_tp->super.super.priority;
                  successor_repo = NULL;
                  successor_repo_key = 0;
                RELEASE_DEP_OUTPUT(es, "GRID_CD_0_0", this_task, "INITIAL_GRID", &nc, rank_src, rank_dst, &data);
                if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep1_atline_314_iftrue, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                  }
                }
              }
            }
          }
    } else {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = (s - 1);
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = 0;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = 0;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = 2;
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = 1;
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_CD_0_0", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0_dep1_atline_314_iffalse, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
  }
  if( action_mask & 0x2 ) {  /* Flow of data GRID_CD_1_0 [1] */
    data.data   = this_task->data._f_GRID_CD_1_0.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
  if( action_mask & 0x2 ) {
        if( (s) == (0) ) {
      __parsec_LBM_FillGrid_task_t* ncc = (__parsec_LBM_FillGrid_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_FillGrid.task_class_id];
        const int FillGrid_x = x;
        if( (FillGrid_x >= (0)) && (FillGrid_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = FillGrid_x;
          const int FillGrid_y = y;
          if( (FillGrid_y >= (0)) && (FillGrid_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = FillGrid_y;
            const int FillGrid_z = z;
            if( (FillGrid_z >= (0)) && (FillGrid_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = FillGrid_z;
              const int FillGrid_c = 1;
              if( (FillGrid_c >= (0)) && (FillGrid_c <= ((conservatives_number - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.c.value);
                ncc->locals.c.value = FillGrid_c;
                const int FillGrid_d = 0;
                if( (FillGrid_d >= (0)) && (FillGrid_d <= ((directions_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.d.value);
                  ncc->locals.d.value = FillGrid_d;
#if defined(DISTRIBUTED)
                  rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                    vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  nc.priority = __parsec_tp->super.super.priority;
                  successor_repo = NULL;
                  successor_repo_key = 0;
                RELEASE_DEP_OUTPUT(es, "GRID_CD_1_0", this_task, "INITIAL_GRID", &nc, rank_src, rank_dst, &data);
                if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep1_atline_317_iftrue, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                  }
                }
              }
            }
          }
    } else {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = (s - 1);
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = 1;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = 0;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = 2;
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = 1;
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_CD_1_0", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0_dep1_atline_317_iffalse, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
  }
  if( action_mask & 0x4 ) {  /* Flow of data GRID_CD_2_0 [2] */
    data.data   = this_task->data._f_GRID_CD_2_0.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
  if( action_mask & 0x4 ) {
        if( (s) == (0) ) {
      __parsec_LBM_FillGrid_task_t* ncc = (__parsec_LBM_FillGrid_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_FillGrid.task_class_id];
        const int FillGrid_x = x;
        if( (FillGrid_x >= (0)) && (FillGrid_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = FillGrid_x;
          const int FillGrid_y = y;
          if( (FillGrid_y >= (0)) && (FillGrid_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = FillGrid_y;
            const int FillGrid_z = z;
            if( (FillGrid_z >= (0)) && (FillGrid_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = FillGrid_z;
              const int FillGrid_c = 2;
              if( (FillGrid_c >= (0)) && (FillGrid_c <= ((conservatives_number - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.c.value);
                ncc->locals.c.value = FillGrid_c;
                const int FillGrid_d = 0;
                if( (FillGrid_d >= (0)) && (FillGrid_d <= ((directions_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.d.value);
                  ncc->locals.d.value = FillGrid_d;
#if defined(DISTRIBUTED)
                  rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                    vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  nc.priority = __parsec_tp->super.super.priority;
                  successor_repo = NULL;
                  successor_repo_key = 0;
                RELEASE_DEP_OUTPUT(es, "GRID_CD_2_0", this_task, "INITIAL_GRID", &nc, rank_src, rank_dst, &data);
                if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep1_atline_320_iftrue, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                  }
                }
              }
            }
          }
    } else {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = (s - 1);
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = 2;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = 0;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = 2;
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = 1;
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_CD_2_0", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0_dep1_atline_320_iffalse, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
  }
  if( action_mask & 0x8 ) {  /* Flow of data GRID_CD_0_1 [3] */
    data.data   = this_task->data._f_GRID_CD_0_1.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
  if( action_mask & 0x8 ) {
        if( (s) == (0) ) {
      __parsec_LBM_FillGrid_task_t* ncc = (__parsec_LBM_FillGrid_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_FillGrid.task_class_id];
        const int FillGrid_x = x;
        if( (FillGrid_x >= (0)) && (FillGrid_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = FillGrid_x;
          const int FillGrid_y = y;
          if( (FillGrid_y >= (0)) && (FillGrid_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = FillGrid_y;
            const int FillGrid_z = z;
            if( (FillGrid_z >= (0)) && (FillGrid_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = FillGrid_z;
              const int FillGrid_c = 0;
              if( (FillGrid_c >= (0)) && (FillGrid_c <= ((conservatives_number - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.c.value);
                ncc->locals.c.value = FillGrid_c;
                const int FillGrid_d = 1;
                if( (FillGrid_d >= (0)) && (FillGrid_d <= ((directions_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.d.value);
                  ncc->locals.d.value = FillGrid_d;
#if defined(DISTRIBUTED)
                  rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                    vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  nc.priority = __parsec_tp->super.super.priority;
                  successor_repo = NULL;
                  successor_repo_key = 0;
                RELEASE_DEP_OUTPUT(es, "GRID_CD_0_1", this_task, "INITIAL_GRID", &nc, rank_src, rank_dst, &data);
                if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep1_atline_323_iftrue, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                  }
                }
              }
            }
          }
    } else {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = (s - 1);
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = 0;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = 1;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = 2;
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = 1;
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_CD_0_1", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1_dep1_atline_323_iffalse, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
  }
  if( action_mask & 0x10 ) {  /* Flow of data GRID_CD_1_1 [4] */
    data.data   = this_task->data._f_GRID_CD_1_1.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
  if( action_mask & 0x10 ) {
        if( (s) == (0) ) {
      __parsec_LBM_FillGrid_task_t* ncc = (__parsec_LBM_FillGrid_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_FillGrid.task_class_id];
        const int FillGrid_x = x;
        if( (FillGrid_x >= (0)) && (FillGrid_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = FillGrid_x;
          const int FillGrid_y = y;
          if( (FillGrid_y >= (0)) && (FillGrid_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = FillGrid_y;
            const int FillGrid_z = z;
            if( (FillGrid_z >= (0)) && (FillGrid_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = FillGrid_z;
              const int FillGrid_c = 1;
              if( (FillGrid_c >= (0)) && (FillGrid_c <= ((conservatives_number - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.c.value);
                ncc->locals.c.value = FillGrid_c;
                const int FillGrid_d = 1;
                if( (FillGrid_d >= (0)) && (FillGrid_d <= ((directions_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.d.value);
                  ncc->locals.d.value = FillGrid_d;
#if defined(DISTRIBUTED)
                  rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                    vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  nc.priority = __parsec_tp->super.super.priority;
                  successor_repo = NULL;
                  successor_repo_key = 0;
                RELEASE_DEP_OUTPUT(es, "GRID_CD_1_1", this_task, "INITIAL_GRID", &nc, rank_src, rank_dst, &data);
                if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep1_atline_326_iftrue, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                  }
                }
              }
            }
          }
    } else {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = (s - 1);
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = 1;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = 1;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = 2;
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = 1;
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_CD_1_1", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1_dep1_atline_326_iffalse, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
  }
  if( action_mask & 0x20 ) {  /* Flow of data GRID_CD_2_1 [5] */
    data.data   = this_task->data._f_GRID_CD_2_1.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
  if( action_mask & 0x20 ) {
        if( (s) == (0) ) {
      __parsec_LBM_FillGrid_task_t* ncc = (__parsec_LBM_FillGrid_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_FillGrid.task_class_id];
        const int FillGrid_x = x;
        if( (FillGrid_x >= (0)) && (FillGrid_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = FillGrid_x;
          const int FillGrid_y = y;
          if( (FillGrid_y >= (0)) && (FillGrid_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = FillGrid_y;
            const int FillGrid_z = z;
            if( (FillGrid_z >= (0)) && (FillGrid_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = FillGrid_z;
              const int FillGrid_c = 2;
              if( (FillGrid_c >= (0)) && (FillGrid_c <= ((conservatives_number - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.c.value);
                ncc->locals.c.value = FillGrid_c;
                const int FillGrid_d = 1;
                if( (FillGrid_d >= (0)) && (FillGrid_d <= ((directions_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.d.value);
                  ncc->locals.d.value = FillGrid_d;
#if defined(DISTRIBUTED)
                  rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                    vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.c.value, ncc->locals.d.value);
                  nc.priority = __parsec_tp->super.super.priority;
                  successor_repo = NULL;
                  successor_repo_key = 0;
                RELEASE_DEP_OUTPUT(es, "GRID_CD_2_1", this_task, "INITIAL_GRID", &nc, rank_src, rank_dst, &data);
                if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep1_atline_329_iftrue, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                  }
                }
              }
            }
          }
    } else {
      __parsec_LBM_Exchange_task_t* ncc = (__parsec_LBM_Exchange_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_Exchange.task_class_id];
        const int Exchange_x = x;
        if( (Exchange_x >= (0)) && (Exchange_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = Exchange_x;
          const int Exchange_y = y;
          if( (Exchange_y >= (0)) && (Exchange_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = Exchange_y;
            const int Exchange_z = z;
            if( (Exchange_z >= (0)) && (Exchange_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = Exchange_z;
              const int Exchange_s = (s - 1);
              if( (Exchange_s >= (0)) && (Exchange_s <= ((number_of_steps - 2))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = Exchange_s;
                const int Exchange_conservative = 2;
                if( (Exchange_conservative >= (0)) && (Exchange_conservative <= ((conservatives_number - 1))) ) {
                  assert(&nc.locals[4].value == &ncc->locals.conservative.value);
                  ncc->locals.conservative.value = Exchange_conservative;
                  const int Exchange_direction = 1;
                  if( (Exchange_direction >= (0)) && (Exchange_direction <= ((directions_number - 1))) ) {
                    assert(&nc.locals[5].value == &ncc->locals.direction.value);
                    ncc->locals.direction.value = Exchange_direction;
                    const int Exchange_dimension = 2;
                    if( (Exchange_dimension >= (0)) && (Exchange_dimension <= ((3 - 1))) ) {
                      assert(&nc.locals[6].value == &ncc->locals.dimension.value);
                      ncc->locals.dimension.value = Exchange_dimension;
                      const int Exchange_side = 1;
                      if( (Exchange_side >= (0)) && (Exchange_side <= (1)) ) {
                        assert(&nc.locals[7].value == &ncc->locals.side.value);
                        ncc->locals.side.value = Exchange_side;
#if defined(DISTRIBUTED)
                        rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                          vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, ncc->locals.conservative.value, ncc->locals.direction.value);
                        nc.priority = __parsec_tp->super.super.priority;
                        successor_repo = Exchange_repo;
                        successor_repo_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
                      RELEASE_DEP_OUTPUT(es, "GRID_CD_2_1", this_task, "GRID_TO", &nc, rank_src, rank_dst, &data);
                      if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1_dep1_atline_329_iffalse, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
    }
  }
  }
  if( action_mask & 0x40 ) {  /* Flow of data X [6] */
    data.data   = this_task->data._f_X.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  =   /* Control: always empty */ 0;
    data.local.dst_count  =   /* Control: always empty */ 0;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = NULL;
    data.remote.src_datatype = PARSEC_DATATYPE_NULL;
    data.remote.dst_datatype = PARSEC_DATATYPE_NULL;
    data.remote.src_count  =   /* Control: always empty */ 0;
    data.remote.dst_count  =   /* Control: always empty */ 0;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
  if( action_mask & 0x40 ) {
        if( (s) != (0) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = (s - 1);
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "X", this_task, "X", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_LBM_STEP_for_X_dep1_atline_332, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
  }
  (void)data;(void)nc;(void)es;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_LBM_LBM_STEP(parsec_execution_stream_t *es, __parsec_LBM_LBM_STEP_task_t *this_task, uint32_t action_mask, parsec_remote_deps_t *deps)
{
PARSEC_PINS(es, RELEASE_DEPS_BEGIN, (parsec_task_t *)this_task);{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  parsec_release_dep_fct_arg_t arg;
  int __vp_id;
  int consume_local_repo = 0;
  arg.action_mask = action_mask;
  arg.output_entry = NULL;
  arg.output_repo = NULL;
#if defined(DISTRIBUTED)
  arg.remote_deps = deps;
#endif  /* defined(DISTRIBUTED) */
  assert(NULL != es);
  arg.ready_lists = alloca(sizeof(parsec_task_t *) * es->virtual_process->parsec_context->nb_vp);
  for( __vp_id = 0; __vp_id < es->virtual_process->parsec_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );
  (void)__parsec_tp; (void)deps;
  if( action_mask & PARSEC_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data._f_GRID_CD_0_0.source_repo_entry ) {
        data_repo_entry_used_once( this_task->data._f_GRID_CD_0_0.source_repo, this_task->data._f_GRID_CD_0_0.source_repo_entry->ht_item.key );
    }
    if( NULL != this_task->data._f_GRID_CD_1_0.source_repo_entry ) {
        data_repo_entry_used_once( this_task->data._f_GRID_CD_1_0.source_repo, this_task->data._f_GRID_CD_1_0.source_repo_entry->ht_item.key );
    }
    if( NULL != this_task->data._f_GRID_CD_2_0.source_repo_entry ) {
        data_repo_entry_used_once( this_task->data._f_GRID_CD_2_0.source_repo, this_task->data._f_GRID_CD_2_0.source_repo_entry->ht_item.key );
    }
    if( NULL != this_task->data._f_GRID_CD_0_1.source_repo_entry ) {
        data_repo_entry_used_once( this_task->data._f_GRID_CD_0_1.source_repo, this_task->data._f_GRID_CD_0_1.source_repo_entry->ht_item.key );
    }
    if( NULL != this_task->data._f_GRID_CD_1_1.source_repo_entry ) {
        data_repo_entry_used_once( this_task->data._f_GRID_CD_1_1.source_repo, this_task->data._f_GRID_CD_1_1.source_repo_entry->ht_item.key );
    }
    if( NULL != this_task->data._f_GRID_CD_2_1.source_repo_entry ) {
        data_repo_entry_used_once( this_task->data._f_GRID_CD_2_1.source_repo, this_task->data._f_GRID_CD_2_1.source_repo_entry->ht_item.key );
    }
  }
  consume_local_repo = (this_task->repo_entry != NULL);
  arg.output_repo = LBM_STEP_repo;
  arg.output_entry = this_task->repo_entry;
  arg.output_usage = 0;
  if( action_mask & (PARSEC_ACTION_RELEASE_LOCAL_DEPS | PARSEC_ACTION_GET_REPO_ENTRY) ) {
    arg.output_entry = data_repo_lookup_entry_and_create( es, arg.output_repo, __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals));
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(PARSEC_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  if(action_mask & ( PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE ) ){
    /* Generate the reshape promise for thet outputs that need it */
    iterate_successors_of_LBM_LBM_STEP(es, this_task, action_mask, parsec_set_up_reshape_promise, &arg);
   }
  iterate_successors_of_LBM_LBM_STEP(es, this_task, action_mask, parsec_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & PARSEC_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    parsec_remote_dep_activate(es, (parsec_task_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {
    data_repo_entry_addto_usage_limit(LBM_STEP_repo, arg.output_entry->ht_item.key, arg.output_usage);
    __parsec_schedule_vp(es, arg.ready_lists, 0);
  }
      if (consume_local_repo) {
         data_repo_entry_used_once( LBM_STEP_repo, this_task->repo_entry->ht_item.key );
      }
  if( action_mask & PARSEC_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data._f_GRID_CD_0_0.data_in ) {
        PARSEC_DATA_COPY_RELEASE(this_task->data._f_GRID_CD_0_0.data_in);
    }
    if( NULL != this_task->data._f_GRID_CD_1_0.data_in ) {
        PARSEC_DATA_COPY_RELEASE(this_task->data._f_GRID_CD_1_0.data_in);
    }
    if( NULL != this_task->data._f_GRID_CD_2_0.data_in ) {
        PARSEC_DATA_COPY_RELEASE(this_task->data._f_GRID_CD_2_0.data_in);
    }
    if( NULL != this_task->data._f_GRID_CD_0_1.data_in ) {
        PARSEC_DATA_COPY_RELEASE(this_task->data._f_GRID_CD_0_1.data_in);
    }
    if( NULL != this_task->data._f_GRID_CD_1_1.data_in ) {
        PARSEC_DATA_COPY_RELEASE(this_task->data._f_GRID_CD_1_1.data_in);
    }
    if( NULL != this_task->data._f_GRID_CD_2_1.data_in ) {
        PARSEC_DATA_COPY_RELEASE(this_task->data._f_GRID_CD_2_1.data_in);
    }
  }
PARSEC_PINS(es, RELEASE_DEPS_END, (parsec_task_t *)this_task);}
  return 0;
}

static int data_lookup_of_LBM_LBM_STEP(parsec_execution_stream_t *es, __parsec_LBM_LBM_STEP_task_t *this_task)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  parsec_assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__parsec_tp; (void)generic_locals; (void)es;
  parsec_data_copy_t *chunk = NULL;
  data_repo_t        *reshape_repo  = NULL, *consumed_repo  = NULL;
  data_repo_entry_t  *reshape_entry = NULL, *consumed_entry = NULL;
  parsec_key_t        reshape_entry_key = 0, consumed_entry_key = 0;
  uint8_t             consumed_flow_index;
  parsec_dep_data_description_t data;
  int ret;
  (void)reshape_repo; (void)reshape_entry; (void)reshape_entry_key;
  (void)consumed_repo; (void)consumed_entry; (void)consumed_entry_key;
  (void)consumed_flow_index;
  (void)chunk; (void)data; (void)ret;
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int s = this_task->locals.s.value; (void)s;
  if( NULL == this_task->repo_entry ){
    this_task->repo_entry = data_repo_lookup_entry_and_create(es, LBM_STEP_repo,                                       __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals));
    data_repo_entry_addto_usage_limit(LBM_STEP_repo, this_task->repo_entry->ht_item.key, 1);    this_task->repo_entry ->generator = (void*)this_task;  /* for AYU */
#if defined(PARSEC_SIM)
    assert(this_task->repo_entry ->sim_exec_date == 0);
    this_task->repo_entry ->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  /* The reshape repo is the current task repo. */  reshape_repo = LBM_STEP_repo;
  reshape_entry_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals) ;
  reshape_entry = this_task->repo_entry;
  /* Lookup the input data, and store them in the context if any */

if(! this_task->data._f_GRID_CD_0_0.fulfill ){ /* Flow GRID_CD_0_0 */
  consumed_repo = NULL;
  consumed_entry_key = 0;
  consumed_entry = NULL;
  chunk = NULL;

    this_task->data._f_GRID_CD_0_0.data_out = NULL;  /* By default, if nothing matches */
    if( (s) == (0) ) {
    /* Flow GRID_CD_0_0 [0] dependency [0] from predecessor FillGrid */
    if( NULL == (chunk = this_task->data._f_GRID_CD_0_0.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_FillGrid_parsec_assignment_t *target_locals = (__parsec_LBM_FillGrid_parsec_assignment_t*)&generic_locals;
      const int FillGridx = target_locals->x.value = x; (void)FillGridx;
      const int FillGridy = target_locals->y.value = y; (void)FillGridy;
      const int FillGridz = target_locals->z.value = z; (void)FillGridz;
      const int FillGridc = target_locals->c.value = 0; (void)FillGridc;
      const int FillGridd = target_locals->d.value = 0; (void)FillGridd;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 0;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = FillGrid_repo;
          consumed_entry_key = __jdf2c_make_key_FillGrid((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_CD_0_0", &LBM_FillGrid, "INITIAL_GRID", target_locals, chunk);
      this_task->data._f_GRID_CD_0_0.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_CD_0_0.source_repo;
      consumed_entry = this_task->data._f_GRID_CD_0_0.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_CD_0_0.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 0;
          assert( (this_task->data._f_GRID_CD_0_0.source_repo == reshape_repo)
               && (this_task->data._f_GRID_CD_0_0.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_CD_0_0.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0, 0);
#endif
    }
    } else {
    /* Flow GRID_CD_0_0 [0] dependency [0] from predecessor Exchange */
    if( NULL == (chunk = this_task->data._f_GRID_CD_0_0.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_Exchange_parsec_assignment_t *target_locals = (__parsec_LBM_Exchange_parsec_assignment_t*)&generic_locals;
      const int Exchangex = target_locals->x.value = x; (void)Exchangex;
      const int Exchangey = target_locals->y.value = y; (void)Exchangey;
      const int Exchangez = target_locals->z.value = z; (void)Exchangez;
      const int Exchanges = target_locals->s.value = (s - 1); (void)Exchanges;
      const int Exchangeconservative = target_locals->conservative.value = 0; (void)Exchangeconservative;
      const int Exchangedirection = target_locals->direction.value = 0; (void)Exchangedirection;
      const int Exchangedimension = target_locals->dimension.value = 2; (void)Exchangedimension;
      const int Exchangeside = target_locals->side.value = 1; (void)Exchangeside;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 0;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = Exchange_repo;
          consumed_entry_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_CD_0_0", &LBM_Exchange, "GRID_TO", target_locals, chunk);
      this_task->data._f_GRID_CD_0_0.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_CD_0_0.source_repo;
      consumed_entry = this_task->data._f_GRID_CD_0_0.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_CD_0_0.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[0] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 0;
          assert( (this_task->data._f_GRID_CD_0_0.source_repo == reshape_repo)
               && (this_task->data._f_GRID_CD_0_0.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_CD_0_0.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0, 0);
#endif
    }
    }
    this_task->data._f_GRID_CD_0_0.data_in     = chunk;
    this_task->data._f_GRID_CD_0_0.source_repo       = consumed_repo;
    this_task->data._f_GRID_CD_0_0.source_repo_entry = consumed_entry;
    if( this_task->data._f_GRID_CD_0_0.source_repo_entry == this_task->repo_entry ){
      /* in case we have consume from this task repo entry for the flow,
       * it is cleaned up, avoiding having old stuff during release_deps_of
       */
      this_task->repo_entry->data[0] = NULL;
    }
    this_task->data._f_GRID_CD_0_0.fulfill = 1;
}


if(! this_task->data._f_GRID_CD_1_0.fulfill ){ /* Flow GRID_CD_1_0 */
  consumed_repo = NULL;
  consumed_entry_key = 0;
  consumed_entry = NULL;
  chunk = NULL;

    this_task->data._f_GRID_CD_1_0.data_out = NULL;  /* By default, if nothing matches */
    if( (s) == (0) ) {
    /* Flow GRID_CD_1_0 [1] dependency [1] from predecessor FillGrid */
    if( NULL == (chunk = this_task->data._f_GRID_CD_1_0.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_FillGrid_parsec_assignment_t *target_locals = (__parsec_LBM_FillGrid_parsec_assignment_t*)&generic_locals;
      const int FillGridx = target_locals->x.value = x; (void)FillGridx;
      const int FillGridy = target_locals->y.value = y; (void)FillGridy;
      const int FillGridz = target_locals->z.value = z; (void)FillGridz;
      const int FillGridc = target_locals->c.value = 1; (void)FillGridc;
      const int FillGridd = target_locals->d.value = 0; (void)FillGridd;
      if( (reshape_entry != NULL) && (reshape_entry->data[1] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 1;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = FillGrid_repo;
          consumed_entry_key = __jdf2c_make_key_FillGrid((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 1, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_CD_1_0", &LBM_FillGrid, "INITIAL_GRID", target_locals, chunk);
      this_task->data._f_GRID_CD_1_0.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_CD_1_0.source_repo;
      consumed_entry = this_task->data._f_GRID_CD_1_0.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_CD_1_0.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[1] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 1;
          assert( (this_task->data._f_GRID_CD_1_0.source_repo == reshape_repo)
               && (this_task->data._f_GRID_CD_1_0.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 1, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_CD_1_0.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0, 0);
#endif
    }
    } else {
    /* Flow GRID_CD_1_0 [1] dependency [1] from predecessor Exchange */
    if( NULL == (chunk = this_task->data._f_GRID_CD_1_0.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_Exchange_parsec_assignment_t *target_locals = (__parsec_LBM_Exchange_parsec_assignment_t*)&generic_locals;
      const int Exchangex = target_locals->x.value = x; (void)Exchangex;
      const int Exchangey = target_locals->y.value = y; (void)Exchangey;
      const int Exchangez = target_locals->z.value = z; (void)Exchangez;
      const int Exchanges = target_locals->s.value = (s - 1); (void)Exchanges;
      const int Exchangeconservative = target_locals->conservative.value = 1; (void)Exchangeconservative;
      const int Exchangedirection = target_locals->direction.value = 0; (void)Exchangedirection;
      const int Exchangedimension = target_locals->dimension.value = 2; (void)Exchangedimension;
      const int Exchangeside = target_locals->side.value = 1; (void)Exchangeside;
      if( (reshape_entry != NULL) && (reshape_entry->data[1] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 1;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = Exchange_repo;
          consumed_entry_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 1, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_CD_1_0", &LBM_Exchange, "GRID_TO", target_locals, chunk);
      this_task->data._f_GRID_CD_1_0.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_CD_1_0.source_repo;
      consumed_entry = this_task->data._f_GRID_CD_1_0.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_CD_1_0.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[1] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 1;
          assert( (this_task->data._f_GRID_CD_1_0.source_repo == reshape_repo)
               && (this_task->data._f_GRID_CD_1_0.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 1, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_CD_1_0.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0, 0);
#endif
    }
    }
    this_task->data._f_GRID_CD_1_0.data_in     = chunk;
    this_task->data._f_GRID_CD_1_0.source_repo       = consumed_repo;
    this_task->data._f_GRID_CD_1_0.source_repo_entry = consumed_entry;
    if( this_task->data._f_GRID_CD_1_0.source_repo_entry == this_task->repo_entry ){
      /* in case we have consume from this task repo entry for the flow,
       * it is cleaned up, avoiding having old stuff during release_deps_of
       */
      this_task->repo_entry->data[1] = NULL;
    }
    this_task->data._f_GRID_CD_1_0.fulfill = 1;
}


if(! this_task->data._f_GRID_CD_2_0.fulfill ){ /* Flow GRID_CD_2_0 */
  consumed_repo = NULL;
  consumed_entry_key = 0;
  consumed_entry = NULL;
  chunk = NULL;

    this_task->data._f_GRID_CD_2_0.data_out = NULL;  /* By default, if nothing matches */
    if( (s) == (0) ) {
    /* Flow GRID_CD_2_0 [2] dependency [2] from predecessor FillGrid */
    if( NULL == (chunk = this_task->data._f_GRID_CD_2_0.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_FillGrid_parsec_assignment_t *target_locals = (__parsec_LBM_FillGrid_parsec_assignment_t*)&generic_locals;
      const int FillGridx = target_locals->x.value = x; (void)FillGridx;
      const int FillGridy = target_locals->y.value = y; (void)FillGridy;
      const int FillGridz = target_locals->z.value = z; (void)FillGridz;
      const int FillGridc = target_locals->c.value = 2; (void)FillGridc;
      const int FillGridd = target_locals->d.value = 0; (void)FillGridd;
      if( (reshape_entry != NULL) && (reshape_entry->data[2] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 2;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = FillGrid_repo;
          consumed_entry_key = __jdf2c_make_key_FillGrid((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 2, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_CD_2_0", &LBM_FillGrid, "INITIAL_GRID", target_locals, chunk);
      this_task->data._f_GRID_CD_2_0.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_CD_2_0.source_repo;
      consumed_entry = this_task->data._f_GRID_CD_2_0.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_CD_2_0.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[2] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 2;
          assert( (this_task->data._f_GRID_CD_2_0.source_repo == reshape_repo)
               && (this_task->data._f_GRID_CD_2_0.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 2, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_CD_2_0.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0, 0);
#endif
    }
    } else {
    /* Flow GRID_CD_2_0 [2] dependency [2] from predecessor Exchange */
    if( NULL == (chunk = this_task->data._f_GRID_CD_2_0.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_Exchange_parsec_assignment_t *target_locals = (__parsec_LBM_Exchange_parsec_assignment_t*)&generic_locals;
      const int Exchangex = target_locals->x.value = x; (void)Exchangex;
      const int Exchangey = target_locals->y.value = y; (void)Exchangey;
      const int Exchangez = target_locals->z.value = z; (void)Exchangez;
      const int Exchanges = target_locals->s.value = (s - 1); (void)Exchanges;
      const int Exchangeconservative = target_locals->conservative.value = 2; (void)Exchangeconservative;
      const int Exchangedirection = target_locals->direction.value = 0; (void)Exchangedirection;
      const int Exchangedimension = target_locals->dimension.value = 2; (void)Exchangedimension;
      const int Exchangeside = target_locals->side.value = 1; (void)Exchangeside;
      if( (reshape_entry != NULL) && (reshape_entry->data[2] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 2;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = Exchange_repo;
          consumed_entry_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 2, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_CD_2_0", &LBM_Exchange, "GRID_TO", target_locals, chunk);
      this_task->data._f_GRID_CD_2_0.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_CD_2_0.source_repo;
      consumed_entry = this_task->data._f_GRID_CD_2_0.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_CD_2_0.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[2] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 2;
          assert( (this_task->data._f_GRID_CD_2_0.source_repo == reshape_repo)
               && (this_task->data._f_GRID_CD_2_0.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 2, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_CD_2_0.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0, 0);
#endif
    }
    }
    this_task->data._f_GRID_CD_2_0.data_in     = chunk;
    this_task->data._f_GRID_CD_2_0.source_repo       = consumed_repo;
    this_task->data._f_GRID_CD_2_0.source_repo_entry = consumed_entry;
    if( this_task->data._f_GRID_CD_2_0.source_repo_entry == this_task->repo_entry ){
      /* in case we have consume from this task repo entry for the flow,
       * it is cleaned up, avoiding having old stuff during release_deps_of
       */
      this_task->repo_entry->data[2] = NULL;
    }
    this_task->data._f_GRID_CD_2_0.fulfill = 1;
}


if(! this_task->data._f_GRID_CD_0_1.fulfill ){ /* Flow GRID_CD_0_1 */
  consumed_repo = NULL;
  consumed_entry_key = 0;
  consumed_entry = NULL;
  chunk = NULL;

    this_task->data._f_GRID_CD_0_1.data_out = NULL;  /* By default, if nothing matches */
    if( (s) == (0) ) {
    /* Flow GRID_CD_0_1 [3] dependency [3] from predecessor FillGrid */
    if( NULL == (chunk = this_task->data._f_GRID_CD_0_1.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_FillGrid_parsec_assignment_t *target_locals = (__parsec_LBM_FillGrid_parsec_assignment_t*)&generic_locals;
      const int FillGridx = target_locals->x.value = x; (void)FillGridx;
      const int FillGridy = target_locals->y.value = y; (void)FillGridy;
      const int FillGridz = target_locals->z.value = z; (void)FillGridz;
      const int FillGridc = target_locals->c.value = 0; (void)FillGridc;
      const int FillGridd = target_locals->d.value = 1; (void)FillGridd;
      if( (reshape_entry != NULL) && (reshape_entry->data[3] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 3;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = FillGrid_repo;
          consumed_entry_key = __jdf2c_make_key_FillGrid((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 3, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_CD_0_1", &LBM_FillGrid, "INITIAL_GRID", target_locals, chunk);
      this_task->data._f_GRID_CD_0_1.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_CD_0_1.source_repo;
      consumed_entry = this_task->data._f_GRID_CD_0_1.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_CD_0_1.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[3] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 3;
          assert( (this_task->data._f_GRID_CD_0_1.source_repo == reshape_repo)
               && (this_task->data._f_GRID_CD_0_1.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 3, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_CD_0_1.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1, 0);
#endif
    }
    } else {
    /* Flow GRID_CD_0_1 [3] dependency [3] from predecessor Exchange */
    if( NULL == (chunk = this_task->data._f_GRID_CD_0_1.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_Exchange_parsec_assignment_t *target_locals = (__parsec_LBM_Exchange_parsec_assignment_t*)&generic_locals;
      const int Exchangex = target_locals->x.value = x; (void)Exchangex;
      const int Exchangey = target_locals->y.value = y; (void)Exchangey;
      const int Exchangez = target_locals->z.value = z; (void)Exchangez;
      const int Exchanges = target_locals->s.value = (s - 1); (void)Exchanges;
      const int Exchangeconservative = target_locals->conservative.value = 0; (void)Exchangeconservative;
      const int Exchangedirection = target_locals->direction.value = 1; (void)Exchangedirection;
      const int Exchangedimension = target_locals->dimension.value = 2; (void)Exchangedimension;
      const int Exchangeside = target_locals->side.value = 1; (void)Exchangeside;
      if( (reshape_entry != NULL) && (reshape_entry->data[3] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 3;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = Exchange_repo;
          consumed_entry_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 3, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_CD_0_1", &LBM_Exchange, "GRID_TO", target_locals, chunk);
      this_task->data._f_GRID_CD_0_1.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_CD_0_1.source_repo;
      consumed_entry = this_task->data._f_GRID_CD_0_1.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_CD_0_1.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[3] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 3;
          assert( (this_task->data._f_GRID_CD_0_1.source_repo == reshape_repo)
               && (this_task->data._f_GRID_CD_0_1.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 3, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_CD_0_1.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1, 0);
#endif
    }
    }
    this_task->data._f_GRID_CD_0_1.data_in     = chunk;
    this_task->data._f_GRID_CD_0_1.source_repo       = consumed_repo;
    this_task->data._f_GRID_CD_0_1.source_repo_entry = consumed_entry;
    if( this_task->data._f_GRID_CD_0_1.source_repo_entry == this_task->repo_entry ){
      /* in case we have consume from this task repo entry for the flow,
       * it is cleaned up, avoiding having old stuff during release_deps_of
       */
      this_task->repo_entry->data[3] = NULL;
    }
    this_task->data._f_GRID_CD_0_1.fulfill = 1;
}


if(! this_task->data._f_GRID_CD_1_1.fulfill ){ /* Flow GRID_CD_1_1 */
  consumed_repo = NULL;
  consumed_entry_key = 0;
  consumed_entry = NULL;
  chunk = NULL;

    this_task->data._f_GRID_CD_1_1.data_out = NULL;  /* By default, if nothing matches */
    if( (s) == (0) ) {
    /* Flow GRID_CD_1_1 [4] dependency [4] from predecessor FillGrid */
    if( NULL == (chunk = this_task->data._f_GRID_CD_1_1.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_FillGrid_parsec_assignment_t *target_locals = (__parsec_LBM_FillGrid_parsec_assignment_t*)&generic_locals;
      const int FillGridx = target_locals->x.value = x; (void)FillGridx;
      const int FillGridy = target_locals->y.value = y; (void)FillGridy;
      const int FillGridz = target_locals->z.value = z; (void)FillGridz;
      const int FillGridc = target_locals->c.value = 1; (void)FillGridc;
      const int FillGridd = target_locals->d.value = 1; (void)FillGridd;
      if( (reshape_entry != NULL) && (reshape_entry->data[4] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 4;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = FillGrid_repo;
          consumed_entry_key = __jdf2c_make_key_FillGrid((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 4, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_CD_1_1", &LBM_FillGrid, "INITIAL_GRID", target_locals, chunk);
      this_task->data._f_GRID_CD_1_1.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_CD_1_1.source_repo;
      consumed_entry = this_task->data._f_GRID_CD_1_1.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_CD_1_1.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[4] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 4;
          assert( (this_task->data._f_GRID_CD_1_1.source_repo == reshape_repo)
               && (this_task->data._f_GRID_CD_1_1.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 4, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_CD_1_1.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1, 0);
#endif
    }
    } else {
    /* Flow GRID_CD_1_1 [4] dependency [4] from predecessor Exchange */
    if( NULL == (chunk = this_task->data._f_GRID_CD_1_1.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_Exchange_parsec_assignment_t *target_locals = (__parsec_LBM_Exchange_parsec_assignment_t*)&generic_locals;
      const int Exchangex = target_locals->x.value = x; (void)Exchangex;
      const int Exchangey = target_locals->y.value = y; (void)Exchangey;
      const int Exchangez = target_locals->z.value = z; (void)Exchangez;
      const int Exchanges = target_locals->s.value = (s - 1); (void)Exchanges;
      const int Exchangeconservative = target_locals->conservative.value = 1; (void)Exchangeconservative;
      const int Exchangedirection = target_locals->direction.value = 1; (void)Exchangedirection;
      const int Exchangedimension = target_locals->dimension.value = 2; (void)Exchangedimension;
      const int Exchangeside = target_locals->side.value = 1; (void)Exchangeside;
      if( (reshape_entry != NULL) && (reshape_entry->data[4] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 4;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = Exchange_repo;
          consumed_entry_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 4, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_CD_1_1", &LBM_Exchange, "GRID_TO", target_locals, chunk);
      this_task->data._f_GRID_CD_1_1.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_CD_1_1.source_repo;
      consumed_entry = this_task->data._f_GRID_CD_1_1.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_CD_1_1.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[4] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 4;
          assert( (this_task->data._f_GRID_CD_1_1.source_repo == reshape_repo)
               && (this_task->data._f_GRID_CD_1_1.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 4, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_CD_1_1.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1, 0);
#endif
    }
    }
    this_task->data._f_GRID_CD_1_1.data_in     = chunk;
    this_task->data._f_GRID_CD_1_1.source_repo       = consumed_repo;
    this_task->data._f_GRID_CD_1_1.source_repo_entry = consumed_entry;
    if( this_task->data._f_GRID_CD_1_1.source_repo_entry == this_task->repo_entry ){
      /* in case we have consume from this task repo entry for the flow,
       * it is cleaned up, avoiding having old stuff during release_deps_of
       */
      this_task->repo_entry->data[4] = NULL;
    }
    this_task->data._f_GRID_CD_1_1.fulfill = 1;
}


if(! this_task->data._f_GRID_CD_2_1.fulfill ){ /* Flow GRID_CD_2_1 */
  consumed_repo = NULL;
  consumed_entry_key = 0;
  consumed_entry = NULL;
  chunk = NULL;

    this_task->data._f_GRID_CD_2_1.data_out = NULL;  /* By default, if nothing matches */
    if( (s) == (0) ) {
    /* Flow GRID_CD_2_1 [5] dependency [5] from predecessor FillGrid */
    if( NULL == (chunk = this_task->data._f_GRID_CD_2_1.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_FillGrid_parsec_assignment_t *target_locals = (__parsec_LBM_FillGrid_parsec_assignment_t*)&generic_locals;
      const int FillGridx = target_locals->x.value = x; (void)FillGridx;
      const int FillGridy = target_locals->y.value = y; (void)FillGridy;
      const int FillGridz = target_locals->z.value = z; (void)FillGridz;
      const int FillGridc = target_locals->c.value = 2; (void)FillGridc;
      const int FillGridd = target_locals->d.value = 1; (void)FillGridd;
      if( (reshape_entry != NULL) && (reshape_entry->data[5] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 5;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = FillGrid_repo;
          consumed_entry_key = __jdf2c_make_key_FillGrid((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 5, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_CD_2_1", &LBM_FillGrid, "INITIAL_GRID", target_locals, chunk);
      this_task->data._f_GRID_CD_2_1.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_CD_2_1.source_repo;
      consumed_entry = this_task->data._f_GRID_CD_2_1.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_CD_2_1.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[5] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 5;
          assert( (this_task->data._f_GRID_CD_2_1.source_repo == reshape_repo)
               && (this_task->data._f_GRID_CD_2_1.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 5, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_CD_2_1.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1, 0);
#endif
    }
    } else {
    /* Flow GRID_CD_2_1 [5] dependency [5] from predecessor Exchange */
    if( NULL == (chunk = this_task->data._f_GRID_CD_2_1.data_in) ) {
    /* No data set up by predecessor on this task input flow */
__parsec_LBM_Exchange_parsec_assignment_t *target_locals = (__parsec_LBM_Exchange_parsec_assignment_t*)&generic_locals;
      const int Exchangex = target_locals->x.value = x; (void)Exchangex;
      const int Exchangey = target_locals->y.value = y; (void)Exchangey;
      const int Exchangez = target_locals->z.value = z; (void)Exchangez;
      const int Exchanges = target_locals->s.value = (s - 1); (void)Exchanges;
      const int Exchangeconservative = target_locals->conservative.value = 2; (void)Exchangeconservative;
      const int Exchangedirection = target_locals->direction.value = 1; (void)Exchangedirection;
      const int Exchangedimension = target_locals->dimension.value = 2; (void)Exchangedimension;
      const int Exchangeside = target_locals->side.value = 1; (void)Exchangeside;
      if( (reshape_entry != NULL) && (reshape_entry->data[5] != NULL) ){
          /* Reshape promise set up on this task repo by predecessor */
          consumed_repo = reshape_repo;
          consumed_entry = reshape_entry;
          consumed_entry_key = reshape_entry_key;
          consumed_flow_index = 5;
      }else{
          /* Consume from predecessor's repo */
          consumed_repo = Exchange_repo;
          consumed_entry_key = __jdf2c_make_key_Exchange((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)target_locals) ;
          consumed_entry = data_repo_lookup_entry( consumed_repo, consumed_entry_key );
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 5, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      ACQUIRE_FLOW(this_task, "GRID_CD_2_1", &LBM_Exchange, "GRID_TO", target_locals, chunk);
      this_task->data._f_GRID_CD_2_1.data_out = chunk;
    } 
    else {
      /* Data set up by predecessor on this task input flow */
      consumed_repo = this_task->data._f_GRID_CD_2_1.source_repo;
      consumed_entry = this_task->data._f_GRID_CD_2_1.source_repo_entry;
      consumed_entry_key = this_task->data._f_GRID_CD_2_1.source_repo_entry->ht_item.key;
      if( (reshape_entry != NULL) && (reshape_entry->data[5] != NULL) ){
          /* Reshape promise set up on input by predecessor is on this task repo */
          consumed_flow_index = 5;
          assert( (this_task->data._f_GRID_CD_2_1.source_repo == reshape_repo)
               && (this_task->data._f_GRID_CD_2_1.source_repo_entry == reshape_entry));
      }else{
          /* Reshape promise set up on input by predecessor is the predecesssor task repo */
          consumed_flow_index = 0;
      }
      data.data   = NULL;
      data.data_future   = (parsec_datacopy_future_t*)consumed_entry->data[consumed_flow_index];
          data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
      if( (ret = parsec_get_copy_reshape_from_dep(es, this_task->taskpool, (parsec_task_t *)this_task, 5, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
          return ret;
      }
      this_task->data._f_GRID_CD_2_1.data_out = parsec_data_get_copy(chunk->original, target_device);
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
      parsec_prof_grapher_data_input(chunk->original, (parsec_task_t*)this_task, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1, 0);
#endif
    }
    }
    this_task->data._f_GRID_CD_2_1.data_in     = chunk;
    this_task->data._f_GRID_CD_2_1.source_repo       = consumed_repo;
    this_task->data._f_GRID_CD_2_1.source_repo_entry = consumed_entry;
    if( this_task->data._f_GRID_CD_2_1.source_repo_entry == this_task->repo_entry ){
      /* in case we have consume from this task repo entry for the flow,
       * it is cleaned up, avoiding having old stuff during release_deps_of
       */
      this_task->repo_entry->data[5] = NULL;
    }
    this_task->data._f_GRID_CD_2_1.fulfill = 1;
}

  /* X is a control flow */
  this_task->data._f_X.data_in         = NULL;
  this_task->data._f_X.data_out        = NULL;
  this_task->data._f_X.source_repo       = NULL;
  this_task->data._f_X.source_repo_entry = NULL;
  /** Generate profiling information */
#if defined(PARSEC_PROF_TRACE)
  this_task->prof_info.desc = (parsec_data_collection_t*)__parsec_tp->super._g_descGridDC;
  this_task->prof_info.priority = this_task->priority;
  this_task->prof_info.data_id   = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->data_key((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, x, y, z, 0, 0);
  this_task->prof_info.task_class_id = this_task->task_class->task_class_id;
  this_task->prof_info.task_return_code = -1;
#endif  /* defined(PARSEC_PROF_TRACE) */
  return PARSEC_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_LBM_LBM_STEP(parsec_execution_stream_t *es, const __parsec_LBM_LBM_STEP_task_t *this_task,
              uint32_t* flow_mask, parsec_dep_data_description_t* data)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  (void)__parsec_tp; (void)es; (void)this_task; (void)data;
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int s = this_task->locals.s.value; (void)s;
  data->local.arena = data->remote.arena = NULL;
  data->local.src_datatype  = data->local.dst_datatype = PARSEC_DATATYPE_NULL;
  data->local.src_count     = data->local.dst_count = 0;
  data->local.src_displ     = data->local.dst_displ = 0;
  data->data_future  = NULL;
  if( (*flow_mask) & 0x80000000U ) { /* these are the input dependencies remote datatypes  */
if( (*flow_mask) & 0x1U ) {  /* Flow GRID_CD_0_0 */
    if( ((*flow_mask) & 0x1U) ) {  /* Have unconditional! */
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x1U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x1U) */
if( (*flow_mask) & 0x2U ) {  /* Flow GRID_CD_1_0 */
    if( ((*flow_mask) & 0x2U) ) {  /* Have unconditional! */
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x2U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x2U) */
if( (*flow_mask) & 0x4U ) {  /* Flow GRID_CD_2_0 */
    if( ((*flow_mask) & 0x4U) ) {  /* Have unconditional! */
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x4U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x4U) */
if( (*flow_mask) & 0x8U ) {  /* Flow GRID_CD_0_1 */
    if( ((*flow_mask) & 0x8U) ) {  /* Have unconditional! */
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x8U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x8U) */
if( (*flow_mask) & 0x10U ) {  /* Flow GRID_CD_1_1 */
    if( ((*flow_mask) & 0x10U) ) {  /* Have unconditional! */
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x10U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x10U) */
if( (*flow_mask) & 0x20U ) {  /* Flow GRID_CD_2_1 */
    if( ((*flow_mask) & 0x20U) ) {  /* Have unconditional! */
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x20U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x20U) */
    goto no_mask_match;
  }

  /* these are the output dependencies remote datatypes */
if( (*flow_mask) & 0x1U ) {  /* Flow GRID_CD_0_0 */
    if( ((*flow_mask) & 0x1U) ) {
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x1U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x1U) */
if( (*flow_mask) & 0x2U ) {  /* Flow GRID_CD_1_0 */
    if( ((*flow_mask) & 0x2U) ) {
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x2U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x2U) */
if( (*flow_mask) & 0x4U ) {  /* Flow GRID_CD_2_0 */
    if( ((*flow_mask) & 0x4U) ) {
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x4U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x4U) */
if( (*flow_mask) & 0x8U ) {  /* Flow GRID_CD_0_1 */
    if( ((*flow_mask) & 0x8U) ) {
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x8U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x8U) */
if( (*flow_mask) & 0x10U ) {  /* Flow GRID_CD_1_1 */
    if( ((*flow_mask) & 0x10U) ) {
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x10U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x10U) */
if( (*flow_mask) & 0x20U ) {  /* Flow GRID_CD_2_1 */
    if( ((*flow_mask) & 0x20U) ) {
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x20U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x20U) */
 no_mask_match:
  data->data                             = NULL;
  data->local.arena = data->remote.arena = NULL;
  data->local.src_datatype  = data->local.dst_datatype = PARSEC_DATATYPE_NULL;
  data->remote.src_datatype = data->remote.dst_datatype = PARSEC_DATATYPE_NULL;
  data->local.src_count     = data->local.dst_count = 0;
  data->remote.src_count    = data->remote.dst_count = 0;
  data->local.src_displ     = data->local.dst_displ = 0;
  data->remote.src_displ    = data->remote.dst_displ = 0;
  data->data_future  = NULL;
  (*flow_mask) = 0;  /* nothing left */
  (void)x;  (void)y;  (void)z;  (void)s;
  return PARSEC_HOOK_RETURN_DONE;
}
#if defined(PARSEC_HAVE_CUDA)
struct parsec_body_cuda_LBM_LBM_STEP_s {
  uint8_t      index;
  cudaStream_t stream;
  void*           dyld_fn;
};

static int cuda_kernel_submit_LBM_LBM_STEP(parsec_device_gpu_module_t  *gpu_device,
                                    parsec_gpu_task_t           *gpu_task,
                                    parsec_gpu_exec_stream_t    *gpu_stream )
{
  __parsec_LBM_LBM_STEP_task_t *this_task = (__parsec_LBM_LBM_STEP_task_t *)gpu_task->ec;
  parsec_device_cuda_module_t *cuda_device = (parsec_device_cuda_module_t*)gpu_device;
  parsec_cuda_exec_stream_t *cuda_stream = (parsec_cuda_exec_stream_t*)gpu_stream;
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  struct parsec_body_cuda_LBM_LBM_STEP_s parsec_body = { cuda_device->cuda_index, cuda_stream->cuda_stream, NULL };
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int s = this_task->locals.s.value; (void)s;
  (void)x;  (void)y;  (void)z;  (void)s;

  (void)gpu_device; (void)gpu_stream; (void)__parsec_tp; (void)parsec_body; (void)cuda_device; (void)cuda_stream;
  /** Declare the variables that will hold the data, and all the accounting for each */
    parsec_data_copy_t *_f_GRID_CD_0_0 = this_task->data._f_GRID_CD_0_0.data_out;
    void *GRID_CD_0_0 = PARSEC_DATA_COPY_GET_PTR(_f_GRID_CD_0_0); (void)GRID_CD_0_0;
    parsec_data_copy_t *_f_GRID_CD_1_0 = this_task->data._f_GRID_CD_1_0.data_out;
    void *GRID_CD_1_0 = PARSEC_DATA_COPY_GET_PTR(_f_GRID_CD_1_0); (void)GRID_CD_1_0;
    parsec_data_copy_t *_f_GRID_CD_2_0 = this_task->data._f_GRID_CD_2_0.data_out;
    void *GRID_CD_2_0 = PARSEC_DATA_COPY_GET_PTR(_f_GRID_CD_2_0); (void)GRID_CD_2_0;
    parsec_data_copy_t *_f_GRID_CD_0_1 = this_task->data._f_GRID_CD_0_1.data_out;
    void *GRID_CD_0_1 = PARSEC_DATA_COPY_GET_PTR(_f_GRID_CD_0_1); (void)GRID_CD_0_1;
    parsec_data_copy_t *_f_GRID_CD_1_1 = this_task->data._f_GRID_CD_1_1.data_out;
    void *GRID_CD_1_1 = PARSEC_DATA_COPY_GET_PTR(_f_GRID_CD_1_1); (void)GRID_CD_1_1;
    parsec_data_copy_t *_f_GRID_CD_2_1 = this_task->data._f_GRID_CD_2_1.data_out;
    void *GRID_CD_2_1 = PARSEC_DATA_COPY_GET_PTR(_f_GRID_CD_2_1); (void)GRID_CD_2_1;

  /** Update starting simulation date */
#if defined(PARSEC_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eGRID_CD_0_0 = this_task->data._f_GRID_CD_0_0.source_repo_entry;
    if( (NULL != eGRID_CD_0_0) && (eGRID_CD_0_0->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eGRID_CD_0_0->sim_exec_date;
    data_repo_entry_t *eGRID_CD_1_0 = this_task->data._f_GRID_CD_1_0.source_repo_entry;
    if( (NULL != eGRID_CD_1_0) && (eGRID_CD_1_0->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eGRID_CD_1_0->sim_exec_date;
    data_repo_entry_t *eGRID_CD_2_0 = this_task->data._f_GRID_CD_2_0.source_repo_entry;
    if( (NULL != eGRID_CD_2_0) && (eGRID_CD_2_0->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eGRID_CD_2_0->sim_exec_date;
    data_repo_entry_t *eGRID_CD_0_1 = this_task->data._f_GRID_CD_0_1.source_repo_entry;
    if( (NULL != eGRID_CD_0_1) && (eGRID_CD_0_1->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eGRID_CD_0_1->sim_exec_date;
    data_repo_entry_t *eGRID_CD_1_1 = this_task->data._f_GRID_CD_1_1.source_repo_entry;
    if( (NULL != eGRID_CD_1_1) && (eGRID_CD_1_1->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eGRID_CD_1_1->sim_exec_date;
    data_repo_entry_t *eGRID_CD_2_1 = this_task->data._f_GRID_CD_2_1.source_repo_entry;
    if( (NULL != eGRID_CD_2_1) && (eGRID_CD_2_1->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eGRID_CD_2_1->sim_exec_date;
    if( this_task->task_class->sim_cost_fct != NULL ) {
      this_task->sim_exec_date += this_task->task_class->sim_cost_fct(this_task);
    }
    if( es->largest_simulation_date < this_task->sim_exec_date )
      es->largest_simulation_date = this_task->sim_exec_date;
  }
#endif
  /** Cache Awareness Accounting */
#if defined(PARSEC_CACHE_AWARENESS)
  cache_buf_referenced(es->closest_cache, GRID_CD_0_0);
  cache_buf_referenced(es->closest_cache, GRID_CD_1_0);
  cache_buf_referenced(es->closest_cache, GRID_CD_2_0);
  cache_buf_referenced(es->closest_cache, GRID_CD_0_1);
  cache_buf_referenced(es->closest_cache, GRID_CD_1_1);
  cache_buf_referenced(es->closest_cache, GRID_CD_2_1);
#endif /* PARSEC_CACHE_AWARENESS */
#if defined(PARSEC_DEBUG_NOISIER)
  {
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream, "GPU[%s]:\tEnqueue on device %s priority %d", gpu_device->super.name, 
           parsec_task_snprintf(tmp, MAX_TASK_STRLEN, (parsec_task_t *)this_task),
           this_task->priority );
  }
#endif /* defined(PARSEC_DEBUG_NOISIER) */


#if !defined(PARSEC_PROF_DRY_BODY)

/*-----                                LBM_STEP BODY                                  -----*/

#if defined(PARSEC_PROF_TRACE)
  if(gpu_stream->prof_event_track_enable) {
    PARSEC_TASK_PROF_TRACE(gpu_stream->profiling,
                           PARSEC_PROF_FUNC_KEY_START(this_task->taskpool,
                                     this_task->task_class->task_class_id),
                           (parsec_task_t*)this_task);
    gpu_task->prof_key_end = PARSEC_PROF_FUNC_KEY_END(this_task->taskpool,
                                   this_task->task_class->task_class_id);
    gpu_task->prof_event_id = this_task->task_class->key_functions->
           key_hash(this_task->task_class->make_key(this_task->taskpool, ((parsec_task_t*)this_task)->locals), NULL);
    gpu_task->prof_tp_id = this_task->taskpool->taskpool_id;
  }
#endif /* PARSEC_PROF_TRACE */
#line 337 "LBM.jdf"
    double *subgrid[gridParameters.conservativesNumber][gridParameters.directionsNumber];
    subgrid[0][0] = GRID_CD_0_0;
    subgrid[1][0] = GRID_CD_1_0;
    subgrid[2][0] = GRID_CD_2_0;
    subgrid[0][1] = GRID_CD_0_1;
    subgrid[1][1] = GRID_CD_1_1;
    subgrid[2][1] = GRID_CD_2_1;

    printf("[Process %d] kernel LBM_STEP (%d %d %d %d) grid|0][0]=%p, grid[1][0]=%p, grid[2][0]=%p, grid[0][1]=%p, grid[1][1]=%p, grid[2][1]=%p\n",
                rank, x, y, z, s, subgrid[0][0], subgrid[1][0], subgrid[2][0], subgrid[0][1], subgrid[1][1], subgrid[2][1]);

#line 9660 "LBM.c"
/*-----                              END OF LBM_STEP BODY                              -----*/



#endif /*!defined(PARSEC_PROF_DRY_BODY)*/

  return PARSEC_HOOK_RETURN_DONE;
}

static int hook_of_LBM_LBM_STEP_CUDA(parsec_execution_stream_t *es, __parsec_LBM_LBM_STEP_task_t *this_task)
{
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  parsec_gpu_task_t *gpu_task;
  double ratio;
  int dev_index;
    const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int s = this_task->locals.s.value; (void)s;
  (void)x;  (void)y;  (void)z;  (void)s;

  (void)es; (void)__parsec_tp;

  ratio = 1.;
  dev_index = parsec_get_best_device((parsec_task_t*)this_task, ratio);
  assert(dev_index >= 0);
  if( dev_index < 2 ) {
    return PARSEC_HOOK_RETURN_NEXT;  /* Fall back */
  }

  gpu_task = (parsec_gpu_task_t*)calloc(1, sizeof(parsec_gpu_task_t));
  PARSEC_OBJ_CONSTRUCT(gpu_task, parsec_list_item_t);
  gpu_task->ec = (parsec_task_t*)this_task;
  gpu_task->submit = &cuda_kernel_submit_LBM_LBM_STEP;
  gpu_task->task_type = 0;
  gpu_task->load = ratio * parsec_device_sweight[dev_index];
  gpu_task->last_data_check_epoch = -1;  /* force at least one validation for the task */
  gpu_task->stage_in  = parsec_default_cuda_stage_in;
  gpu_task->stage_out = parsec_default_cuda_stage_out;
  gpu_task->pushout = 0;
  gpu_task->flow[0]         = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0;
  gpu_task->flow_dc[0] = NULL;
  gpu_task->flow_nb_elts[0] = gpu_task->ec->data[0].data_in->original->nb_elts;
  gpu_task->flow[1]         = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0;
  gpu_task->flow_dc[1] = NULL;
  gpu_task->flow_nb_elts[1] = gpu_task->ec->data[1].data_in->original->nb_elts;
  gpu_task->flow[2]         = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0;
  gpu_task->flow_dc[2] = NULL;
  gpu_task->flow_nb_elts[2] = gpu_task->ec->data[2].data_in->original->nb_elts;
  gpu_task->flow[3]         = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1;
  gpu_task->flow_dc[3] = NULL;
  gpu_task->flow_nb_elts[3] = gpu_task->ec->data[3].data_in->original->nb_elts;
  gpu_task->flow[4]         = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1;
  gpu_task->flow_dc[4] = NULL;
  gpu_task->flow_nb_elts[4] = gpu_task->ec->data[4].data_in->original->nb_elts;
  gpu_task->flow[5]         = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1;
  gpu_task->flow_dc[5] = NULL;
  gpu_task->flow_nb_elts[5] = gpu_task->ec->data[5].data_in->original->nb_elts;
  gpu_task->flow[6]         = &flow_of_LBM_LBM_STEP_for_X;
  gpu_task->flow_dc[6] = NULL;
  gpu_task->flow_nb_elts[6] = 0;
  parsec_device_load[dev_index] += gpu_task->load;

  return parsec_cuda_kernel_scheduler( es, gpu_task, dev_index );
}

#endif  /*  defined(PARSEC_HAVE_CUDA) */
static int complete_hook_of_LBM_LBM_STEP(parsec_execution_stream_t *es, __parsec_LBM_LBM_STEP_task_t *this_task)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
#if defined(DISTRIBUTED)
    const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int s = this_task->locals.s.value; (void)s;
#endif  /* defined(DISTRIBUTED) */
  (void)es; (void)__parsec_tp;
parsec_data_t* data_t_desc = NULL; (void)data_t_desc;
  if ( NULL != this_task->data._f_GRID_CD_0_0.data_out ) {
#if defined(PARSEC_DEBUG_NOISIER)
     char tmp[128];
#endif
     this_task->data._f_GRID_CD_0_0.data_out->version++;  /* GRID_CD_0_0 */
     PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                          "Complete hook of %s: change Data copy %p to version %d at %s:%d",
                          parsec_task_snprintf(tmp, 128, (parsec_task_t*)(this_task)),
                          this_task->data._f_GRID_CD_0_0.data_out, this_task->data._f_GRID_CD_0_0.data_out->version, __FILE__, __LINE__);
  }
  if ( NULL != this_task->data._f_GRID_CD_1_0.data_out ) {
#if defined(PARSEC_DEBUG_NOISIER)
     char tmp[128];
#endif
     this_task->data._f_GRID_CD_1_0.data_out->version++;  /* GRID_CD_1_0 */
     PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                          "Complete hook of %s: change Data copy %p to version %d at %s:%d",
                          parsec_task_snprintf(tmp, 128, (parsec_task_t*)(this_task)),
                          this_task->data._f_GRID_CD_1_0.data_out, this_task->data._f_GRID_CD_1_0.data_out->version, __FILE__, __LINE__);
  }
  if ( NULL != this_task->data._f_GRID_CD_2_0.data_out ) {
#if defined(PARSEC_DEBUG_NOISIER)
     char tmp[128];
#endif
     this_task->data._f_GRID_CD_2_0.data_out->version++;  /* GRID_CD_2_0 */
     PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                          "Complete hook of %s: change Data copy %p to version %d at %s:%d",
                          parsec_task_snprintf(tmp, 128, (parsec_task_t*)(this_task)),
                          this_task->data._f_GRID_CD_2_0.data_out, this_task->data._f_GRID_CD_2_0.data_out->version, __FILE__, __LINE__);
  }
  if ( NULL != this_task->data._f_GRID_CD_0_1.data_out ) {
#if defined(PARSEC_DEBUG_NOISIER)
     char tmp[128];
#endif
     this_task->data._f_GRID_CD_0_1.data_out->version++;  /* GRID_CD_0_1 */
     PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                          "Complete hook of %s: change Data copy %p to version %d at %s:%d",
                          parsec_task_snprintf(tmp, 128, (parsec_task_t*)(this_task)),
                          this_task->data._f_GRID_CD_0_1.data_out, this_task->data._f_GRID_CD_0_1.data_out->version, __FILE__, __LINE__);
  }
  if ( NULL != this_task->data._f_GRID_CD_1_1.data_out ) {
#if defined(PARSEC_DEBUG_NOISIER)
     char tmp[128];
#endif
     this_task->data._f_GRID_CD_1_1.data_out->version++;  /* GRID_CD_1_1 */
     PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                          "Complete hook of %s: change Data copy %p to version %d at %s:%d",
                          parsec_task_snprintf(tmp, 128, (parsec_task_t*)(this_task)),
                          this_task->data._f_GRID_CD_1_1.data_out, this_task->data._f_GRID_CD_1_1.data_out->version, __FILE__, __LINE__);
  }
  if ( NULL != this_task->data._f_GRID_CD_2_1.data_out ) {
#if defined(PARSEC_DEBUG_NOISIER)
     char tmp[128];
#endif
     this_task->data._f_GRID_CD_2_1.data_out->version++;  /* GRID_CD_2_1 */
     PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                          "Complete hook of %s: change Data copy %p to version %d at %s:%d",
                          parsec_task_snprintf(tmp, 128, (parsec_task_t*)(this_task)),
                          this_task->data._f_GRID_CD_2_1.data_out, this_task->data._f_GRID_CD_2_1.data_out->version, __FILE__, __LINE__);
  }
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  (void)x;  (void)y;  (void)z;  (void)s;

#endif /* DISTRIBUTED */
#if defined(PARSEC_PROF_GRAPHER)
  parsec_prof_grapher_task((parsec_task_t*)this_task, es->th_id, es->virtual_process->vp_id,
     __jdf2c_key_fns_LBM_STEP.key_hash(this_task->task_class->make_key( (parsec_taskpool_t*)this_task->taskpool, ((parsec_task_t*)this_task)->locals), NULL));
#endif  /* defined(PARSEC_PROF_GRAPHER) */
  release_deps_of_LBM_LBM_STEP(es, this_task,
      PARSEC_ACTION_RELEASE_REMOTE_DEPS |
      PARSEC_ACTION_RELEASE_LOCAL_DEPS |
      PARSEC_ACTION_RELEASE_LOCAL_REFS |
      PARSEC_ACTION_RESHAPE_ON_RELEASE |
      0x7f,  /* mask of all dep_index */ 
      NULL);
  return PARSEC_HOOK_RETURN_DONE;
}

static parsec_hook_return_t release_task_of_LBM_LBM_STEP(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    const __parsec_LBM_internal_taskpool_t *__parsec_tp =
        (const __parsec_LBM_internal_taskpool_t *)this_task->taskpool;
    parsec_hash_table_t *ht = (parsec_hash_table_t*)__parsec_tp->super.super.dependencies_array[1];
    parsec_key_t key = this_task->task_class->make_key((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals);
    parsec_hashable_dependency_t *hash_dep = (parsec_hashable_dependency_t *)parsec_hash_table_remove(ht, key);
    parsec_thread_mempool_free(hash_dep->mempool_owner, hash_dep);
    return parsec_release_task_to_mempool_update_nbtasks(es, this_task);
}

static char *LBM_LBM_LBM_STEP_internal_init_deps_key_functions_key_print(char *buffer, size_t buffer_size, parsec_key_t __parsec_key_, void *user_data)
{
  uint64_t __parsec_key = (uint64_t)(uintptr_t)__parsec_key_;
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)user_data;
  int __jdf2c_x_min = 0;
  int x = (__parsec_key) % __parsec_tp->LBM_STEP_x_range + __jdf2c_x_min;
  __parsec_key = __parsec_key / __parsec_tp->LBM_STEP_x_range;
  int __jdf2c_y_min = 0;
  int y = (__parsec_key) % __parsec_tp->LBM_STEP_y_range + __jdf2c_y_min;
  __parsec_key = __parsec_key / __parsec_tp->LBM_STEP_y_range;
  int __jdf2c_z_min = 0;
  int z = (__parsec_key) % __parsec_tp->LBM_STEP_z_range + __jdf2c_z_min;
  __parsec_key = __parsec_key / __parsec_tp->LBM_STEP_z_range;
  int __jdf2c_s_min = 0;
  int s = (__parsec_key) % __parsec_tp->LBM_STEP_s_range + __jdf2c_s_min;
  __parsec_key = __parsec_key / __parsec_tp->LBM_STEP_s_range;
  snprintf(buffer, buffer_size, "LBM_STEP(%d, %d, %d, %d)", x, y, z, s);
  return buffer;
}

static parsec_key_fn_t LBM_LBM_LBM_STEP_internal_init_deps_key_functions = {
   .key_equal = parsec_hash_table_generic_64bits_key_equal,
   .key_print = LBM_LBM_LBM_STEP_internal_init_deps_key_functions_key_print,
   .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

/* Needs: min-max count-tasks iterate */
static int LBM_LBM_STEP_internal_init(parsec_execution_stream_t * es, __parsec_LBM_LBM_STEP_task_t * this_task)
{
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t*)this_task->taskpool;
  int32_t nb_tasks = 0, saved_nb_tasks = 0;
int32_t __x_min = 0x7fffffff, __x_max = 0;int32_t __jdf2c_x_min = 0x7fffffff, __jdf2c_x_max = 0;int32_t __y_min = 0x7fffffff, __y_max = 0;int32_t __jdf2c_y_min = 0x7fffffff, __jdf2c_y_max = 0;int32_t __z_min = 0x7fffffff, __z_max = 0;int32_t __jdf2c_z_min = 0x7fffffff, __jdf2c_z_max = 0;int32_t __s_min = 0x7fffffff, __s_max = 0;int32_t __jdf2c_s_min = 0x7fffffff, __jdf2c_s_max = 0;  __parsec_LBM_LBM_STEP_parsec_assignment_t assignments = {  .x.value = 0, .y.value = 0, .z.value = 0, .s.value = 0 };
  int32_t  x,  y,  z,  s;
  int32_t __jdf2c_x_start, __jdf2c_x_end, __jdf2c_x_inc;
  int32_t __jdf2c_y_start, __jdf2c_y_end, __jdf2c_y_inc;
  int32_t __jdf2c_z_start, __jdf2c_z_end, __jdf2c_z_inc;
  int32_t __jdf2c_s_start, __jdf2c_s_end, __jdf2c_s_inc;
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)
  PARSEC_PROFILING_TRACE(es->es_profile,
                         this_task->taskpool->profiling_array[2 * this_task->task_class->task_class_id],
                         0,
                         this_task->taskpool->taskpool_id, NULL);
#endif /* defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */
    __jdf2c_x_start = 0;
    __jdf2c_x_end = (subgrid_number_x - 1);
    __jdf2c_x_inc = 1;
    __x_min = parsec_imin(__jdf2c_x_start, __jdf2c_x_end);
    __x_max = parsec_imax(__jdf2c_x_start, __jdf2c_x_end);
    __jdf2c_x_min = parsec_imin(__jdf2c_x_min, __x_min);
    __jdf2c_x_max = parsec_imax(__jdf2c_x_max, __x_max);
    for(x =  __jdf2c_x_start;
        x <= __jdf2c_x_end;
        x += __jdf2c_x_inc) {
    assignments.x.value = x;
      __jdf2c_y_start = 0;
      __jdf2c_y_end = (subgrid_number_y - 1);
      __jdf2c_y_inc = 1;
      __y_min = parsec_imin(__jdf2c_y_start, __jdf2c_y_end);
      __y_max = parsec_imax(__jdf2c_y_start, __jdf2c_y_end);
      __jdf2c_y_min = parsec_imin(__jdf2c_y_min, __y_min);
      __jdf2c_y_max = parsec_imax(__jdf2c_y_max, __y_max);
      for(y =  __jdf2c_y_start;
          y <= __jdf2c_y_end;
          y += __jdf2c_y_inc) {
      assignments.y.value = y;
        __jdf2c_z_start = 0;
        __jdf2c_z_end = (subgrid_number_z - 1);
        __jdf2c_z_inc = 1;
        __z_min = parsec_imin(__jdf2c_z_start, __jdf2c_z_end);
        __z_max = parsec_imax(__jdf2c_z_start, __jdf2c_z_end);
        __jdf2c_z_min = parsec_imin(__jdf2c_z_min, __z_min);
        __jdf2c_z_max = parsec_imax(__jdf2c_z_max, __z_max);
        for(z =  __jdf2c_z_start;
            z <= __jdf2c_z_end;
            z += __jdf2c_z_inc) {
        assignments.z.value = z;
          __jdf2c_s_start = 0;
          __jdf2c_s_end = (number_of_steps - 1);
          __jdf2c_s_inc = 1;
          __s_min = parsec_imin(__jdf2c_s_start, __jdf2c_s_end);
          __s_max = parsec_imax(__jdf2c_s_start, __jdf2c_s_end);
          __jdf2c_s_min = parsec_imin(__jdf2c_s_min, __s_min);
          __jdf2c_s_max = parsec_imax(__jdf2c_s_max, __s_max);
          for(s =  __jdf2c_s_start;
              s <= __jdf2c_s_end;
              s += __jdf2c_s_inc) {
          assignments.s.value = s;
          if( !LBM_STEP_pred(x, y, z, s) ) continue;
          nb_tasks++;
        } /* Loop on normal range s */
      } /* For loop of z */ 
    } /* For loop of y */ 
  } /* For loop of x */ 
   if( 0 != nb_tasks ) {
     (void)parsec_atomic_fetch_add_int32(&__parsec_tp->initial_number_tasks, nb_tasks);
   }
  /* Set the range variables for the collision-free hash-computation */
  __parsec_tp->LBM_STEP_x_range = (__jdf2c_x_max - __jdf2c_x_min) + 1;
  __parsec_tp->LBM_STEP_y_range = (__jdf2c_y_max - __jdf2c_y_min) + 1;
  __parsec_tp->LBM_STEP_z_range = (__jdf2c_z_max - __jdf2c_z_min) + 1;
  __parsec_tp->LBM_STEP_s_range = (__jdf2c_s_max - __jdf2c_s_min) + 1;
  this_task->status = PARSEC_TASK_STATUS_COMPLETE;

  PARSEC_AYU_REGISTER_TASK(&LBM_LBM_STEP);
  __parsec_tp->super.super.dependencies_array[1] = PARSEC_OBJ_NEW(parsec_hash_table_t);
  parsec_hash_table_init(__parsec_tp->super.super.dependencies_array[1], offsetof(parsec_hashable_dependency_t, ht_item), 10, LBM_LBM_LBM_STEP_internal_init_deps_key_functions, this_task->taskpool);
  __parsec_tp->repositories[1] = data_repo_create_nothreadsafe(nb_tasks, __jdf2c_key_fns_LBM_STEP, (parsec_taskpool_t*)__parsec_tp, 7);
(void)saved_nb_tasks;
(void)__x_min; (void)__x_max;(void)__y_min; (void)__y_max;(void)__z_min; (void)__z_max;(void)__s_min; (void)__s_max;  (void)__jdf2c_x_start; (void)__jdf2c_x_end; (void)__jdf2c_x_inc;  (void)__jdf2c_y_start; (void)__jdf2c_y_end; (void)__jdf2c_y_inc;  (void)__jdf2c_z_start; (void)__jdf2c_z_end; (void)__jdf2c_z_inc;  (void)__jdf2c_s_start; (void)__jdf2c_s_end; (void)__jdf2c_s_inc;  (void)assignments; (void)__parsec_tp; (void)es;
  if(1 == parsec_atomic_fetch_dec_int32(&__parsec_tp->sync_point)) {
    /* Last initialization task complete. Update the number of tasks. */
    __parsec_tp->super.super.tdm.module->taskpool_addto_nb_tasks(&__parsec_tp->super.super, __parsec_tp->initial_number_tasks);
    parsec_mfence();  /* write memory barrier to guarantee that the scheduler gets the correct number of tasks */
    parsec_taskpool_enable((parsec_taskpool_t*)__parsec_tp, &__parsec_tp->startup_queue,
                           (parsec_task_t*)this_task, es, __parsec_tp->super.super.nb_pending_actions);
    __parsec_tp->super.super.tdm.module->taskpool_ready(&__parsec_tp->super.super);
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)
    PARSEC_PROFILING_TRACE(es->es_profile,
                           this_task->taskpool->profiling_array[2 * this_task->task_class->task_class_id + 1],
                           0,
                           this_task->taskpool->taskpool_id, NULL);
#endif /* defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */
    return PARSEC_HOOK_RETURN_DONE;
  }
  return PARSEC_HOOK_RETURN_DONE;
}

static const __parsec_chore_t __LBM_LBM_STEP_chores[] ={
#if defined(PARSEC_HAVE_CUDA)
    { .type     = PARSEC_DEV_CUDA,
      .dyld     = NULL,
      .evaluate = NULL,
      .hook     = (parsec_hook_t*)hook_of_LBM_LBM_STEP_CUDA },
#endif  /* defined(PARSEC_HAVE_CUDA) */
    { .type     = PARSEC_DEV_NONE,
      .evaluate = NULL,
      .hook     = (parsec_hook_t*)NULL },  /* End marker */
};

static const parsec_task_class_t LBM_LBM_STEP = {
  .name = "LBM_STEP",
  .task_class_id = 1,
  .nb_flows = 7,
  .nb_parameters = 4,
  .nb_locals = 4,
  .task_class_type = PARSEC_TASK_CLASS_TYPE_PTG,
  .params = { &symb_LBM_LBM_STEP_x, &symb_LBM_LBM_STEP_y, &symb_LBM_LBM_STEP_z, &symb_LBM_LBM_STEP_s, NULL },
  .locals = { &symb_LBM_LBM_STEP_x, &symb_LBM_LBM_STEP_y, &symb_LBM_LBM_STEP_z, &symb_LBM_LBM_STEP_s, NULL },
  .data_affinity = (parsec_data_ref_fn_t*)affinity_of_LBM_LBM_STEP,
  .initial_data = (parsec_data_ref_fn_t*)affinity_of_LBM_LBM_STEP,
  .final_data = (parsec_data_ref_fn_t*)affinity_of_LBM_LBM_STEP,
  .priority = NULL,
  .properties = properties_of_LBM_LBM_STEP,
#if MAX_PARAM_COUNT < 6  /* number of read flows of LBM_STEP */
  #error Too many read flows for task LBM_STEP
#endif  /* MAX_PARAM_COUNT */
#if MAX_PARAM_COUNT < 6  /* number of write flows of LBM_STEP */
  #error Too many write flows for task LBM_STEP
#endif  /* MAX_PARAM_COUNT */
  .in = { &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0, &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1, &flow_of_LBM_LBM_STEP_for_X, NULL },
  .out = { &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0, &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1, &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1, &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1, &flow_of_LBM_LBM_STEP_for_X, NULL },
  .flags = 0x0 | PARSEC_HAS_IN_IN_DEPENDENCIES | PARSEC_USE_DEPS_MASK,
  .dependencies_goal = 0x7f,
  .make_key = __jdf2c_make_key_LBM_STEP,
  .task_snprintf = parsec_task_snprintf,
  .key_functions = &__jdf2c_key_fns_LBM_STEP,
  .fini = (parsec_hook_t*)NULL,
  .incarnations = __LBM_LBM_STEP_chores,
  .find_deps = parsec_hash_find_deps,
  .update_deps = parsec_update_deps_with_mask,
  .iterate_successors = (parsec_traverse_function_t*)iterate_successors_of_LBM_LBM_STEP,
  .iterate_predecessors = (parsec_traverse_function_t*)iterate_predecessors_of_LBM_LBM_STEP,
  .release_deps = (parsec_release_deps_t*)release_deps_of_LBM_LBM_STEP,
  .prepare_output = (parsec_hook_t*)NULL,
  .prepare_input = (parsec_hook_t*)data_lookup_of_LBM_LBM_STEP,
  .get_datatype = (parsec_datatype_lookup_t*)datatype_lookup_of_LBM_LBM_STEP,
  .complete_execution = (parsec_hook_t*)complete_hook_of_LBM_LBM_STEP,
  .release_task = &release_task_of_LBM_LBM_STEP,
#if defined(PARSEC_SIM)
  .sim_cost_fct = (parsec_sim_cost_fct_t*)NULL,
#endif
};


/******                                    FillGrid                                    ******/

static inline int32_t minexpr_of_symb_LBM_FillGrid_x_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_FillGrid_x = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_FillGrid_x_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_FillGrid_x_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return (subgrid_number_x - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_FillGrid_x = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_FillGrid_x_fct }
                   }
};
static const parsec_symbol_t symb_LBM_FillGrid_x = { .name = "x", .context_index = 0, .min = &minexpr_of_symb_LBM_FillGrid_x, .max = &maxexpr_of_symb_LBM_FillGrid_x, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_FillGrid_y_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_FillGrid_y = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_FillGrid_y_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_FillGrid_y_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return (subgrid_number_y - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_FillGrid_y = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_FillGrid_y_fct }
                   }
};
static const parsec_symbol_t symb_LBM_FillGrid_y = { .name = "y", .context_index = 1, .min = &minexpr_of_symb_LBM_FillGrid_y, .max = &maxexpr_of_symb_LBM_FillGrid_y, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_FillGrid_z_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_FillGrid_z = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_FillGrid_z_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_FillGrid_z_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return (subgrid_number_z - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_FillGrid_z = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_FillGrid_z_fct }
                   }
};
static const parsec_symbol_t symb_LBM_FillGrid_z = { .name = "z", .context_index = 2, .min = &minexpr_of_symb_LBM_FillGrid_z, .max = &maxexpr_of_symb_LBM_FillGrid_z, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_FillGrid_c_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_FillGrid_c = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_FillGrid_c_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_FillGrid_c_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return (conservatives_number - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_FillGrid_c = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_FillGrid_c_fct }
                   }
};
static const parsec_symbol_t symb_LBM_FillGrid_c = { .name = "c", .context_index = 3, .min = &minexpr_of_symb_LBM_FillGrid_c, .max = &maxexpr_of_symb_LBM_FillGrid_c, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int32_t minexpr_of_symb_LBM_FillGrid_d_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  (void)__parsec_tp; (void)locals;
  return 0;
}
static const parsec_expr_t minexpr_of_symb_LBM_FillGrid_d = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32,
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)minexpr_of_symb_LBM_FillGrid_d_fct }
                   }
};
static inline int maxexpr_of_symb_LBM_FillGrid_d_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return (directions_number - 1);
}
static const parsec_expr_t maxexpr_of_symb_LBM_FillGrid_d = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)maxexpr_of_symb_LBM_FillGrid_d_fct }
                   }
};
static const parsec_symbol_t symb_LBM_FillGrid_d = { .name = "d", .context_index = 4, .min = &minexpr_of_symb_LBM_FillGrid_d, .max = &maxexpr_of_symb_LBM_FillGrid_d, .cst_inc = 1, .expr_inc = NULL,  .flags = PARSEC_SYMBOL_IS_STANDALONE};

static inline int affinity_of_LBM_FillGrid(__parsec_LBM_FillGrid_task_t *this_task,
                     parsec_data_ref_t *ref)
{
    const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t*)this_task->taskpool;
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int c = this_task->locals.c.value; (void)c;
  const int d = this_task->locals.d.value; (void)d;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  ref->dc = (parsec_data_collection_t *)__parsec_tp->super._g_descGridDC;
  /* Compute data key */
  ref->key = ref->dc->data_key(ref->dc, x, y, z, c, d);
  return 1;
}
static const parsec_property_t properties_of_LBM_FillGrid[1] = {
  {.name = NULL, .expr = NULL}
};
static parsec_data_t *flow_of_LBM_FillGrid_for_INITIAL_GRID_dep1_atline_285_direct_access(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *assignments)
{
  const int x = assignments->x.value; (void)x;
  const int y = assignments->y.value; (void)y;
  const int z = assignments->z.value; (void)z;
  const int c = assignments->c.value; (void)c;
  const int d = assignments->d.value; (void)d;

  /* Silence Warnings: should look into parameters to know what variables are useful */
  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  if( __parsec_tp->super.super.context->my_rank == (int32_t)rank_of_descGridDC(x, y, z, c, d) )
    return data_of_descGridDC(x, y, z, c, d);
  return NULL;
}

static const parsec_dep_t flow_of_LBM_FillGrid_for_INITIAL_GRID_dep1_atline_285 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .task_class_id = PARSEC_LOCAL_DATA_TASK_CLASS_ID, /* LBM_descGridDC */
  .direct_data = (parsec_data_lookup_func_t)&flow_of_LBM_FillGrid_for_INITIAL_GRID_dep1_atline_285_direct_access,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_FillGrid_for_INITIAL_GRID,
};
static inline int expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep2_atline_286_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return ((c) == (0) && (d) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep2_atline_286 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep2_atline_286_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_FillGrid_for_INITIAL_GRID_dep2_atline_286 = {
  .cond = &expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep2_atline_286,  /* ((c) == (0) && (d) == (0)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_0,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_FillGrid_for_INITIAL_GRID,
};
static inline int expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep3_atline_287_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return ((c) == (1) && (d) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep3_atline_287 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep3_atline_287_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_FillGrid_for_INITIAL_GRID_dep3_atline_287 = {
  .cond = &expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep3_atline_287,  /* ((c) == (1) && (d) == (0)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_0,
  .dep_index = 1,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_FillGrid_for_INITIAL_GRID,
};
static inline int expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep4_atline_288_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return ((c) == (2) && (d) == (0));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep4_atline_288 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep4_atline_288_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_FillGrid_for_INITIAL_GRID_dep4_atline_288 = {
  .cond = &expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep4_atline_288,  /* ((c) == (2) && (d) == (0)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_0,
  .dep_index = 2,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_FillGrid_for_INITIAL_GRID,
};
static inline int expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep5_atline_289_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return ((c) == (0) && (d) == (1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep5_atline_289 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep5_atline_289_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_FillGrid_for_INITIAL_GRID_dep5_atline_289 = {
  .cond = &expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep5_atline_289,  /* ((c) == (0) && (d) == (1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_0_1,
  .dep_index = 3,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_FillGrid_for_INITIAL_GRID,
};
static inline int expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep6_atline_290_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return ((c) == (1) && (d) == (1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep6_atline_290 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep6_atline_290_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_FillGrid_for_INITIAL_GRID_dep6_atline_290 = {
  .cond = &expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep6_atline_290,  /* ((c) == (1) && (d) == (1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_1_1,
  .dep_index = 4,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_FillGrid_for_INITIAL_GRID,
};
static inline int expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep7_atline_291_fct(const __parsec_LBM_internal_taskpool_t *__parsec_tp, const __parsec_LBM_FillGrid_parsec_assignment_t *locals)
{
  const int x = locals->x.value; (void)x;
  const int y = locals->y.value; (void)y;
  const int z = locals->z.value; (void)z;
  const int c = locals->c.value; (void)c;
  const int d = locals->d.value; (void)d;

  (void)x;
  (void)y;
  (void)z;
  (void)c;
  (void)d;
  (void)__parsec_tp; (void)locals;
  return ((c) == (2) && (d) == (1));
}
static const parsec_expr_t expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep7_atline_291 = {
  .op = PARSEC_EXPR_OP_INLINE,
  .u_expr.v_func = { .type = PARSEC_RETURN_TYPE_INT32, /* PARSEC_RETURN_TYPE_INT32 */
                     .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep7_atline_291_fct }
                   }
};
static const parsec_dep_t flow_of_LBM_FillGrid_for_INITIAL_GRID_dep7_atline_291 = {
  .cond = &expr_of_cond_for_flow_of_LBM_FillGrid_for_INITIAL_GRID_dep7_atline_291,  /* ((c) == (2) && (d) == (1)) */
  .ctl_gather_nb = NULL,
  .task_class_id = 1, /* LBM_LBM_STEP */
  .direct_data = (parsec_data_lookup_func_t)NULL,
  .flow = &flow_of_LBM_LBM_STEP_for_GRID_CD_2_1,
  .dep_index = 5,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_LBM_FillGrid_for_INITIAL_GRID,
};
#if MAX_DEP_IN_COUNT < 1  /* number of input dependencies */
    #error Too many input dependencies (supports up to MAX_DEP_IN_COUNT [=24] but found 1). Fix the code or recompile PaRSEC with a larger MAX_DEP_IN_COUNT.
#endif
#if MAX_DEP_OUT_COUNT < 6  /* number of output dependencies */
    #error Too many output dependencies (supports up to MAX_DEP_OUT_COUNT [=24] but found 6). Fix the code or recompile PaRSEC with a larger MAX_DEP_OUT_COUNT.
#endif

static const parsec_flow_t flow_of_LBM_FillGrid_for_INITIAL_GRID = {
  .name               = "INITIAL_GRID",
  .sym_type           = PARSEC_SYM_INOUT,
  .flow_flags         = PARSEC_FLOW_ACCESS_RW|PARSEC_FLOW_HAS_IN_DEPS,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_LBM_FillGrid_for_INITIAL_GRID_dep1_atline_285 },
  .dep_out    = { &flow_of_LBM_FillGrid_for_INITIAL_GRID_dep2_atline_286,
 &flow_of_LBM_FillGrid_for_INITIAL_GRID_dep3_atline_287,
 &flow_of_LBM_FillGrid_for_INITIAL_GRID_dep4_atline_288,
 &flow_of_LBM_FillGrid_for_INITIAL_GRID_dep5_atline_289,
 &flow_of_LBM_FillGrid_for_INITIAL_GRID_dep6_atline_290,
 &flow_of_LBM_FillGrid_for_INITIAL_GRID_dep7_atline_291 }
};

static void
iterate_successors_of_LBM_FillGrid(parsec_execution_stream_t *es, const __parsec_LBM_FillGrid_task_t *this_task,
               uint32_t action_mask, parsec_ontask_function_t *ontask, void *ontask_arg)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t*)this_task->taskpool;
  parsec_task_t nc;  /* generic placeholder for locals */
  parsec_dep_data_description_t data;
  __parsec_LBM_FillGrid_parsec_assignment_t __jdf2c__tmp_locals = *(__parsec_LBM_FillGrid_parsec_assignment_t*)&this_task->locals;   /* copy of this_task locals in R/W mode to manage local definitions */
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int x = __jdf2c__tmp_locals.x.value; (void)x;
  const int y = __jdf2c__tmp_locals.y.value; (void)y;
  const int z = __jdf2c__tmp_locals.z.value; (void)z;
  const int c = __jdf2c__tmp_locals.c.value; (void)c;
  const int d = __jdf2c__tmp_locals.d.value; (void)d;
  (void)rank_src; (void)rank_dst; (void)__parsec_tp; (void)vpid_dst;
   data_repo_t *successor_repo; parsec_key_t successor_repo_key;  (void)x;  (void)y;  (void)z;  (void)c;  (void)d;
  nc.taskpool  = this_task->taskpool;
  nc.priority  = this_task->priority;
  nc.chore_mask  = PARSEC_DEV_ALL;
#if defined(DISTRIBUTED)
  rank_src = rank_of_descGridDC(x, y, z, c, d);
#endif
  if( action_mask & 0x3f ) {  /* Flow of data INITIAL_GRID [0] */
    data.data   = this_task->data._f_INITIAL_GRID.data_out;
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
    data.data_future  = NULL;
    data.local.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.remote.arena  = PARSEC_LBM_DEFAULT_ADT->arena;
    data.remote.src_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.dst_datatype = (data.data != NULL ? data.data->dtt : PARSEC_DATATYPE_NULL );
    data.remote.src_count  = 1;
    data.remote.dst_count  = 1;
    data.remote.src_displ    = 0;
    data.remote.dst_displ    = 0;
  }
  if( action_mask & 0x1 ) {
        if( ((c) == (0) && (d) == (0)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = 0;
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "INITIAL_GRID", this_task, "GRID_CD_0_0", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_FillGrid_for_INITIAL_GRID_dep2_atline_286, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x2 ) {
        if( ((c) == (1) && (d) == (0)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = 0;
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "INITIAL_GRID", this_task, "GRID_CD_1_0", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_FillGrid_for_INITIAL_GRID_dep3_atline_287, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x4 ) {
        if( ((c) == (2) && (d) == (0)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = 0;
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "INITIAL_GRID", this_task, "GRID_CD_2_0", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_FillGrid_for_INITIAL_GRID_dep4_atline_288, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x8 ) {
        if( ((c) == (0) && (d) == (1)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = 0;
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "INITIAL_GRID", this_task, "GRID_CD_0_1", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_FillGrid_for_INITIAL_GRID_dep5_atline_289, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x10 ) {
        if( ((c) == (1) && (d) == (1)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = 0;
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "INITIAL_GRID", this_task, "GRID_CD_1_1", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_FillGrid_for_INITIAL_GRID_dep6_atline_290, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    if (action_mask & (PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE | PARSEC_ACTION_SEND_REMOTE_DEPS)) {
  }
  if( action_mask & 0x20 ) {
        if( ((c) == (2) && (d) == (1)) ) {
      __parsec_LBM_LBM_STEP_task_t* ncc = (__parsec_LBM_LBM_STEP_task_t*)&nc;
      nc.task_class = __parsec_tp->super.super.task_classes_array[LBM_LBM_STEP.task_class_id];
        const int LBM_STEP_x = x;
        if( (LBM_STEP_x >= (0)) && (LBM_STEP_x <= ((subgrid_number_x - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.x.value);
          ncc->locals.x.value = LBM_STEP_x;
          const int LBM_STEP_y = y;
          if( (LBM_STEP_y >= (0)) && (LBM_STEP_y <= ((subgrid_number_y - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.y.value);
            ncc->locals.y.value = LBM_STEP_y;
            const int LBM_STEP_z = z;
            if( (LBM_STEP_z >= (0)) && (LBM_STEP_z <= ((subgrid_number_z - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.z.value);
              ncc->locals.z.value = LBM_STEP_z;
              const int LBM_STEP_s = 0;
              if( (LBM_STEP_s >= (0)) && (LBM_STEP_s <= ((number_of_steps - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.s.value);
                ncc->locals.s.value = LBM_STEP_s;
#if defined(DISTRIBUTED)
                rank_dst = rank_of_descGridDC(ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                if( (NULL != es) && (rank_dst == es->virtual_process->parsec_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, ncc->locals.x.value, ncc->locals.y.value, ncc->locals.z.value, 0, 0);
                nc.priority = __parsec_tp->super.super.priority;
                successor_repo = LBM_STEP_repo;
                successor_repo_key = __jdf2c_make_key_LBM_STEP((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&ncc->locals);
              RELEASE_DEP_OUTPUT(es, "INITIAL_GRID", this_task, "GRID_CD_2_1", &nc, rank_src, rank_dst, &data);
              if( PARSEC_ITERATE_STOP == ontask(es, &nc, (const parsec_task_t *)this_task, &flow_of_LBM_FillGrid_for_INITIAL_GRID_dep7_atline_291, &data, rank_src, rank_dst, vpid_dst, successor_repo, successor_repo_key, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
  }
  (void)data;(void)nc;(void)es;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_LBM_FillGrid(parsec_execution_stream_t *es, __parsec_LBM_FillGrid_task_t *this_task, uint32_t action_mask, parsec_remote_deps_t *deps)
{
PARSEC_PINS(es, RELEASE_DEPS_BEGIN, (parsec_task_t *)this_task);{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (const __parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  parsec_release_dep_fct_arg_t arg;
  int __vp_id;
  int consume_local_repo = 0;
  arg.action_mask = action_mask;
  arg.output_entry = NULL;
  arg.output_repo = NULL;
#if defined(DISTRIBUTED)
  arg.remote_deps = deps;
#endif  /* defined(DISTRIBUTED) */
  assert(NULL != es);
  arg.ready_lists = alloca(sizeof(parsec_task_t *) * es->virtual_process->parsec_context->nb_vp);
  for( __vp_id = 0; __vp_id < es->virtual_process->parsec_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );
  (void)__parsec_tp; (void)deps;
  if( action_mask & PARSEC_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data._f_INITIAL_GRID.source_repo_entry ) {
        data_repo_entry_used_once( this_task->data._f_INITIAL_GRID.source_repo, this_task->data._f_INITIAL_GRID.source_repo_entry->ht_item.key );
    }
  }
  consume_local_repo = (this_task->repo_entry != NULL);
  arg.output_repo = FillGrid_repo;
  arg.output_entry = this_task->repo_entry;
  arg.output_usage = 0;
  if( action_mask & (PARSEC_ACTION_RELEASE_LOCAL_DEPS | PARSEC_ACTION_GET_REPO_ENTRY) ) {
    arg.output_entry = data_repo_lookup_entry_and_create( es, arg.output_repo, __jdf2c_make_key_FillGrid((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals));
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(PARSEC_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  if(action_mask & ( PARSEC_ACTION_RESHAPE_ON_RELEASE | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE ) ){
    /* Generate the reshape promise for thet outputs that need it */
    iterate_successors_of_LBM_FillGrid(es, this_task, action_mask, parsec_set_up_reshape_promise, &arg);
   }
  iterate_successors_of_LBM_FillGrid(es, this_task, action_mask, parsec_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & PARSEC_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    parsec_remote_dep_activate(es, (parsec_task_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & PARSEC_ACTION_RELEASE_LOCAL_DEPS) {
    data_repo_entry_addto_usage_limit(FillGrid_repo, arg.output_entry->ht_item.key, arg.output_usage);
    __parsec_schedule_vp(es, arg.ready_lists, 0);
  }
      if (consume_local_repo) {
         data_repo_entry_used_once( FillGrid_repo, this_task->repo_entry->ht_item.key );
      }
  if( action_mask & PARSEC_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data._f_INITIAL_GRID.data_in ) {
        PARSEC_DATA_COPY_RELEASE(this_task->data._f_INITIAL_GRID.data_in);
    }
  }
PARSEC_PINS(es, RELEASE_DEPS_END, (parsec_task_t *)this_task);}
  return 0;
}

static int data_lookup_of_LBM_FillGrid(parsec_execution_stream_t *es, __parsec_LBM_FillGrid_task_t *this_task)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  parsec_assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__parsec_tp; (void)generic_locals; (void)es;
  parsec_data_copy_t *chunk = NULL;
  data_repo_t        *reshape_repo  = NULL, *consumed_repo  = NULL;
  data_repo_entry_t  *reshape_entry = NULL, *consumed_entry = NULL;
  parsec_key_t        reshape_entry_key = 0, consumed_entry_key = 0;
  uint8_t             consumed_flow_index;
  parsec_dep_data_description_t data;
  int ret;
  (void)reshape_repo; (void)reshape_entry; (void)reshape_entry_key;
  (void)consumed_repo; (void)consumed_entry; (void)consumed_entry_key;
  (void)consumed_flow_index;
  (void)chunk; (void)data; (void)ret;
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int c = this_task->locals.c.value; (void)c;
  const int d = this_task->locals.d.value; (void)d;
  if( NULL == this_task->repo_entry ){
    this_task->repo_entry = data_repo_lookup_entry_and_create(es, FillGrid_repo,                                       __jdf2c_make_key_FillGrid((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals));
    data_repo_entry_addto_usage_limit(FillGrid_repo, this_task->repo_entry->ht_item.key, 1);    this_task->repo_entry ->generator = (void*)this_task;  /* for AYU */
#if defined(PARSEC_SIM)
    assert(this_task->repo_entry ->sim_exec_date == 0);
    this_task->repo_entry ->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  /* The reshape repo is the current task repo. */  reshape_repo = FillGrid_repo;
  reshape_entry_key = __jdf2c_make_key_FillGrid((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals) ;
  reshape_entry = this_task->repo_entry;
  /* Lookup the input data, and store them in the context if any */

if(! this_task->data._f_INITIAL_GRID.fulfill ){ /* Flow INITIAL_GRID */
  consumed_repo = NULL;
  consumed_entry_key = 0;
  consumed_entry = NULL;
  chunk = NULL;

    this_task->data._f_INITIAL_GRID.data_out = NULL;  /* By default, if nothing matches */
  /* Flow INITIAL_GRID [0] dependency [0] from memory descGridDC */
  if( NULL == (chunk = this_task->data._f_INITIAL_GRID.data_in) ) {
  /* No data set up by predecessor on this task input flow */
#if defined(PARSEC_PROF_GRAPHER) && defined(PARSEC_PROF_TRACE)
  parsec_prof_grapher_data_input(data_of_descGridDC(x, y, z, c, d), (parsec_task_t*)this_task, &flow_of_LBM_FillGrid_for_INITIAL_GRID, 1);
#endif
    chunk = parsec_data_get_copy(data_of_descGridDC(x, y, z, c, d), target_device);
    data.data = chunk;        data.local.arena  = NULL;
    data.local.src_datatype = PARSEC_DATATYPE_NULL;
    data.local.dst_datatype = PARSEC_DATATYPE_NULL;
    data.local.src_count  = 1;
    data.local.dst_count  = 1;
    data.local.src_displ    = 0;
    data.local.dst_displ    = 0;
    data.data_future   = NULL;
    if( (ret = parsec_get_copy_reshape_from_desc(es, this_task->taskpool, (parsec_task_t *)this_task, 0, reshape_repo, reshape_entry_key, &data, &chunk)) < 0){
        return ret;
    }
    this_task->data._f_INITIAL_GRID.data_out = chunk;
    PARSEC_OBJ_RETAIN(chunk);
  } 
    this_task->data._f_INITIAL_GRID.data_in     = chunk;
    this_task->data._f_INITIAL_GRID.source_repo       = consumed_repo;
    this_task->data._f_INITIAL_GRID.source_repo_entry = consumed_entry;
    if( this_task->data._f_INITIAL_GRID.source_repo_entry == this_task->repo_entry ){
      /* in case we have consume from this task repo entry for the flow,
       * it is cleaned up, avoiding having old stuff during release_deps_of
       */
      this_task->repo_entry->data[0] = NULL;
    }
    this_task->data._f_INITIAL_GRID.fulfill = 1;
}

  /** Generate profiling information */
#if defined(PARSEC_PROF_TRACE)
  this_task->prof_info.desc = (parsec_data_collection_t*)__parsec_tp->super._g_descGridDC;
  this_task->prof_info.priority = this_task->priority;
  this_task->prof_info.data_id   = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->data_key((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, x, y, z, c, d);
  this_task->prof_info.task_class_id = this_task->task_class->task_class_id;
  this_task->prof_info.task_return_code = -1;
#endif  /* defined(PARSEC_PROF_TRACE) */
  return PARSEC_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_LBM_FillGrid(parsec_execution_stream_t *es, const __parsec_LBM_FillGrid_task_t *this_task,
              uint32_t* flow_mask, parsec_dep_data_description_t* data)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  (void)__parsec_tp; (void)es; (void)this_task; (void)data;
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int c = this_task->locals.c.value; (void)c;
  const int d = this_task->locals.d.value; (void)d;
  data->local.arena = data->remote.arena = NULL;
  data->local.src_datatype  = data->local.dst_datatype = PARSEC_DATATYPE_NULL;
  data->local.src_count     = data->local.dst_count = 0;
  data->local.src_displ     = data->local.dst_displ = 0;
  data->data_future  = NULL;
  if( (*flow_mask) & 0x80000000U ) { /* these are the input dependencies remote datatypes  */
if( (*flow_mask) & 0x1U ) {  /* Flow INITIAL_GRID */
    if( ((*flow_mask) & 0x1U) ) {  /* Have unconditional! */
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x1U;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x1U) */
    goto no_mask_match;
  }

  /* these are the output dependencies remote datatypes */
if( (*flow_mask) & 0x3fU ) {  /* Flow INITIAL_GRID */
    if( ((*flow_mask) & 0x3fU)
 && (((c) == (0) && (d) == (0)) || ((c) == (1) && (d) == (0)) || ((c) == (2) && (d) == (0)) || ((c) == (0) && (d) == (1)) || ((c) == (1) && (d) == (1)) || ((c) == (2) && (d) == (1))) ) {
    data->remote.arena  =  PARSEC_LBM_DEFAULT_ADT->arena ;
    data->remote.src_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.dst_datatype = PARSEC_LBM_DEFAULT_ADT->opaque_dtt;
    data->remote.src_count  = 1;
    data->remote.dst_count  = 1;
    data->remote.src_displ    = 0;
    data->remote.dst_displ    = 0;
      (*flow_mask) &= ~0x3fU;
      return PARSEC_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x3fU) */
 no_mask_match:
  data->data                             = NULL;
  data->local.arena = data->remote.arena = NULL;
  data->local.src_datatype  = data->local.dst_datatype = PARSEC_DATATYPE_NULL;
  data->remote.src_datatype = data->remote.dst_datatype = PARSEC_DATATYPE_NULL;
  data->local.src_count     = data->local.dst_count = 0;
  data->remote.src_count    = data->remote.dst_count = 0;
  data->local.src_displ     = data->local.dst_displ = 0;
  data->remote.src_displ    = data->remote.dst_displ = 0;
  data->data_future  = NULL;
  (*flow_mask) = 0;  /* nothing left */
  (void)x;  (void)y;  (void)z;  (void)c;  (void)d;
  return PARSEC_HOOK_RETURN_DONE;
}
static int hook_of_LBM_FillGrid(parsec_execution_stream_t *es, __parsec_LBM_FillGrid_task_t *this_task)
{
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
  (void)es; (void)__parsec_tp;
  const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int c = this_task->locals.c.value; (void)c;
  const int d = this_task->locals.d.value; (void)d;
  (void)x;  (void)y;  (void)z;  (void)c;  (void)d;

  /** Declare the variables that will hold the data, and all the accounting for each */
    parsec_data_copy_t *_f_INITIAL_GRID = this_task->data._f_INITIAL_GRID.data_in;
    void *INITIAL_GRID = PARSEC_DATA_COPY_GET_PTR(_f_INITIAL_GRID); (void)INITIAL_GRID;

  /** Update starting simulation date */
#if defined(PARSEC_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eINITIAL_GRID = this_task->data._f_INITIAL_GRID.source_repo_entry;
    if( (NULL != eINITIAL_GRID) && (eINITIAL_GRID->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eINITIAL_GRID->sim_exec_date;
    if( this_task->task_class->sim_cost_fct != NULL ) {
      this_task->sim_exec_date += this_task->task_class->sim_cost_fct(this_task);
    }
    if( es->largest_simulation_date < this_task->sim_exec_date )
      es->largest_simulation_date = this_task->sim_exec_date;
  }
#endif
  /** Transfer the ownership to the CPU */
#if defined(PARSEC_HAVE_CUDA)
    if ( NULL != _f_INITIAL_GRID ) {
      parsec_data_transfer_ownership_to_copy( _f_INITIAL_GRID->original, 0 /* device */,
                                           PARSEC_FLOW_ACCESS_RW);
    }
#endif  /* defined(PARSEC_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(PARSEC_CACHE_AWARENESS)
  cache_buf_referenced(es->closest_cache, INITIAL_GRID);
#endif /* PARSEC_CACHE_AWARENESS */


#if !defined(PARSEC_PROF_DRY_BODY)

/*-----                                FillGrid BODY                                  -----*/

#line 294 "LBM.jdf"
    double *grid = INITIAL_GRID;

    for(int i=0;i<tile_size_x*tile_size_y*tile_size_z;++i)
    {
        grid[i] = 1;
    }

    printf("FillGrid: %d %d %d %d %d (grid=%p)\n", x, y, z, c, d, grid);

#line 10998 "LBM.c"
/*-----                              END OF FillGrid BODY                              -----*/



#endif /*!defined(PARSEC_PROF_DRY_BODY)*/

  return PARSEC_HOOK_RETURN_DONE;
}
static int complete_hook_of_LBM_FillGrid(parsec_execution_stream_t *es, __parsec_LBM_FillGrid_task_t *this_task)
{
  const __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)this_task->taskpool;
#if defined(DISTRIBUTED)
    const int x = this_task->locals.x.value; (void)x;
  const int y = this_task->locals.y.value; (void)y;
  const int z = this_task->locals.z.value; (void)z;
  const int c = this_task->locals.c.value; (void)c;
  const int d = this_task->locals.d.value; (void)d;
#endif  /* defined(DISTRIBUTED) */
  (void)es; (void)__parsec_tp;
parsec_data_t* data_t_desc = NULL; (void)data_t_desc;
  if ( NULL != this_task->data._f_INITIAL_GRID.data_out ) {
#if defined(PARSEC_DEBUG_NOISIER)
     char tmp[128];
#endif
     this_task->data._f_INITIAL_GRID.data_out->version++;  /* INITIAL_GRID */
     PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                          "Complete hook of %s: change Data copy %p to version %d at %s:%d",
                          parsec_task_snprintf(tmp, 128, (parsec_task_t*)(this_task)),
                          this_task->data._f_INITIAL_GRID.data_out, this_task->data._f_INITIAL_GRID.data_out->version, __FILE__, __LINE__);
  }
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  (void)x;  (void)y;  (void)z;  (void)c;  (void)d;

#endif /* DISTRIBUTED */
#if defined(PARSEC_PROF_GRAPHER)
  parsec_prof_grapher_task((parsec_task_t*)this_task, es->th_id, es->virtual_process->vp_id,
     __jdf2c_key_fns_FillGrid.key_hash(this_task->task_class->make_key( (parsec_taskpool_t*)this_task->taskpool, ((parsec_task_t*)this_task)->locals), NULL));
#endif  /* defined(PARSEC_PROF_GRAPHER) */
  release_deps_of_LBM_FillGrid(es, this_task,
      PARSEC_ACTION_RELEASE_REMOTE_DEPS |
      PARSEC_ACTION_RELEASE_LOCAL_DEPS |
      PARSEC_ACTION_RELEASE_LOCAL_REFS |
      PARSEC_ACTION_RESHAPE_ON_RELEASE |
      0x3f,  /* mask of all dep_index */ 
      NULL);
  return PARSEC_HOOK_RETURN_DONE;
}

static parsec_hook_return_t release_task_of_LBM_FillGrid(parsec_execution_stream_t *es, parsec_task_t *this_task)
{
    const __parsec_LBM_internal_taskpool_t *__parsec_tp =
        (const __parsec_LBM_internal_taskpool_t *)this_task->taskpool;
    parsec_hash_table_t *ht = (parsec_hash_table_t*)__parsec_tp->super.super.dependencies_array[0];
    parsec_key_t key = this_task->task_class->make_key((const parsec_taskpool_t*)__parsec_tp, (const parsec_assignment_t*)&this_task->locals);
    parsec_hashable_dependency_t *hash_dep = (parsec_hashable_dependency_t *)parsec_hash_table_remove(ht, key);
    parsec_thread_mempool_free(hash_dep->mempool_owner, hash_dep);
    return parsec_release_task_to_mempool_update_nbtasks(es, this_task);
}

static char *LBM_LBM_FillGrid_internal_init_deps_key_functions_key_print(char *buffer, size_t buffer_size, parsec_key_t __parsec_key_, void *user_data)
{
  uint64_t __parsec_key = (uint64_t)(uintptr_t)__parsec_key_;
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t *)user_data;
  int __jdf2c_x_min = 0;
  int x = (__parsec_key) % __parsec_tp->FillGrid_x_range + __jdf2c_x_min;
  __parsec_key = __parsec_key / __parsec_tp->FillGrid_x_range;
  int __jdf2c_y_min = 0;
  int y = (__parsec_key) % __parsec_tp->FillGrid_y_range + __jdf2c_y_min;
  __parsec_key = __parsec_key / __parsec_tp->FillGrid_y_range;
  int __jdf2c_z_min = 0;
  int z = (__parsec_key) % __parsec_tp->FillGrid_z_range + __jdf2c_z_min;
  __parsec_key = __parsec_key / __parsec_tp->FillGrid_z_range;
  int __jdf2c_c_min = 0;
  int c = (__parsec_key) % __parsec_tp->FillGrid_c_range + __jdf2c_c_min;
  __parsec_key = __parsec_key / __parsec_tp->FillGrid_c_range;
  int __jdf2c_d_min = 0;
  int d = (__parsec_key) % __parsec_tp->FillGrid_d_range + __jdf2c_d_min;
  __parsec_key = __parsec_key / __parsec_tp->FillGrid_d_range;
  snprintf(buffer, buffer_size, "FillGrid(%d, %d, %d, %d, %d)", x, y, z, c, d);
  return buffer;
}

static parsec_key_fn_t LBM_LBM_FillGrid_internal_init_deps_key_functions = {
   .key_equal = parsec_hash_table_generic_64bits_key_equal,
   .key_print = LBM_LBM_FillGrid_internal_init_deps_key_functions_key_print,
   .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

/* Needs: min-max count-tasks iterate */
static int LBM_FillGrid_internal_init(parsec_execution_stream_t * es, __parsec_LBM_FillGrid_task_t * this_task)
{
  __parsec_LBM_internal_taskpool_t *__parsec_tp = (__parsec_LBM_internal_taskpool_t*)this_task->taskpool;
  int32_t nb_tasks = 0, saved_nb_tasks = 0;
int32_t __x_min = 0x7fffffff, __x_max = 0;int32_t __jdf2c_x_min = 0x7fffffff, __jdf2c_x_max = 0;int32_t __y_min = 0x7fffffff, __y_max = 0;int32_t __jdf2c_y_min = 0x7fffffff, __jdf2c_y_max = 0;int32_t __z_min = 0x7fffffff, __z_max = 0;int32_t __jdf2c_z_min = 0x7fffffff, __jdf2c_z_max = 0;int32_t __c_min = 0x7fffffff, __c_max = 0;int32_t __jdf2c_c_min = 0x7fffffff, __jdf2c_c_max = 0;int32_t __d_min = 0x7fffffff, __d_max = 0;int32_t __jdf2c_d_min = 0x7fffffff, __jdf2c_d_max = 0;  __parsec_LBM_FillGrid_parsec_assignment_t assignments = {  .x.value = 0, .y.value = 0, .z.value = 0, .c.value = 0, .d.value = 0 };
  int32_t  x,  y,  z,  c,  d;
  int32_t __jdf2c_x_start, __jdf2c_x_end, __jdf2c_x_inc;
  int32_t __jdf2c_y_start, __jdf2c_y_end, __jdf2c_y_inc;
  int32_t __jdf2c_z_start, __jdf2c_z_end, __jdf2c_z_inc;
  int32_t __jdf2c_c_start, __jdf2c_c_end, __jdf2c_c_inc;
  int32_t __jdf2c_d_start, __jdf2c_d_end, __jdf2c_d_inc;
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)
  PARSEC_PROFILING_TRACE(es->es_profile,
                         this_task->taskpool->profiling_array[2 * this_task->task_class->task_class_id],
                         0,
                         this_task->taskpool->taskpool_id, NULL);
#endif /* defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */
    __jdf2c_x_start = 0;
    __jdf2c_x_end = (subgrid_number_x - 1);
    __jdf2c_x_inc = 1;
    __x_min = parsec_imin(__jdf2c_x_start, __jdf2c_x_end);
    __x_max = parsec_imax(__jdf2c_x_start, __jdf2c_x_end);
    __jdf2c_x_min = parsec_imin(__jdf2c_x_min, __x_min);
    __jdf2c_x_max = parsec_imax(__jdf2c_x_max, __x_max);
    for(x =  __jdf2c_x_start;
        x <= __jdf2c_x_end;
        x += __jdf2c_x_inc) {
    assignments.x.value = x;
      __jdf2c_y_start = 0;
      __jdf2c_y_end = (subgrid_number_y - 1);
      __jdf2c_y_inc = 1;
      __y_min = parsec_imin(__jdf2c_y_start, __jdf2c_y_end);
      __y_max = parsec_imax(__jdf2c_y_start, __jdf2c_y_end);
      __jdf2c_y_min = parsec_imin(__jdf2c_y_min, __y_min);
      __jdf2c_y_max = parsec_imax(__jdf2c_y_max, __y_max);
      for(y =  __jdf2c_y_start;
          y <= __jdf2c_y_end;
          y += __jdf2c_y_inc) {
      assignments.y.value = y;
        __jdf2c_z_start = 0;
        __jdf2c_z_end = (subgrid_number_z - 1);
        __jdf2c_z_inc = 1;
        __z_min = parsec_imin(__jdf2c_z_start, __jdf2c_z_end);
        __z_max = parsec_imax(__jdf2c_z_start, __jdf2c_z_end);
        __jdf2c_z_min = parsec_imin(__jdf2c_z_min, __z_min);
        __jdf2c_z_max = parsec_imax(__jdf2c_z_max, __z_max);
        for(z =  __jdf2c_z_start;
            z <= __jdf2c_z_end;
            z += __jdf2c_z_inc) {
        assignments.z.value = z;
          __jdf2c_c_start = 0;
          __jdf2c_c_end = (conservatives_number - 1);
          __jdf2c_c_inc = 1;
          __c_min = parsec_imin(__jdf2c_c_start, __jdf2c_c_end);
          __c_max = parsec_imax(__jdf2c_c_start, __jdf2c_c_end);
          __jdf2c_c_min = parsec_imin(__jdf2c_c_min, __c_min);
          __jdf2c_c_max = parsec_imax(__jdf2c_c_max, __c_max);
          for(c =  __jdf2c_c_start;
              c <= __jdf2c_c_end;
              c += __jdf2c_c_inc) {
          assignments.c.value = c;
            __jdf2c_d_start = 0;
            __jdf2c_d_end = (directions_number - 1);
            __jdf2c_d_inc = 1;
            __d_min = parsec_imin(__jdf2c_d_start, __jdf2c_d_end);
            __d_max = parsec_imax(__jdf2c_d_start, __jdf2c_d_end);
            __jdf2c_d_min = parsec_imin(__jdf2c_d_min, __d_min);
            __jdf2c_d_max = parsec_imax(__jdf2c_d_max, __d_max);
            for(d =  __jdf2c_d_start;
                d <= __jdf2c_d_end;
                d += __jdf2c_d_inc) {
            assignments.d.value = d;
            if( !FillGrid_pred(x, y, z, c, d) ) continue;
            nb_tasks++;
          } /* Loop on normal range d */
        } /* For loop of c */ 
      } /* For loop of z */ 
    } /* For loop of y */ 
  } /* For loop of x */ 
   if( 0 != nb_tasks ) {
     (void)parsec_atomic_fetch_add_int32(&__parsec_tp->initial_number_tasks, nb_tasks);
   }
  /* Set the range variables for the collision-free hash-computation */
  __parsec_tp->FillGrid_x_range = (__jdf2c_x_max - __jdf2c_x_min) + 1;
  __parsec_tp->FillGrid_y_range = (__jdf2c_y_max - __jdf2c_y_min) + 1;
  __parsec_tp->FillGrid_z_range = (__jdf2c_z_max - __jdf2c_z_min) + 1;
  __parsec_tp->FillGrid_c_range = (__jdf2c_c_max - __jdf2c_c_min) + 1;
  __parsec_tp->FillGrid_d_range = (__jdf2c_d_max - __jdf2c_d_min) + 1;
    do {
      this_task->super.list_next = (parsec_list_item_t*)__parsec_tp->startup_queue;
    } while(!parsec_atomic_cas_ptr(&__parsec_tp->startup_queue, (parsec_list_item_t*)this_task->super.list_next, this_task));
    this_task->status = PARSEC_TASK_STATUS_HOOK;

  PARSEC_AYU_REGISTER_TASK(&LBM_FillGrid);
  __parsec_tp->super.super.dependencies_array[0] = PARSEC_OBJ_NEW(parsec_hash_table_t);
  parsec_hash_table_init(__parsec_tp->super.super.dependencies_array[0], offsetof(parsec_hashable_dependency_t, ht_item), 10, LBM_LBM_FillGrid_internal_init_deps_key_functions, this_task->taskpool);
  __parsec_tp->repositories[0] = data_repo_create_nothreadsafe(nb_tasks, __jdf2c_key_fns_FillGrid, (parsec_taskpool_t*)__parsec_tp, 1);
(void)saved_nb_tasks;
(void)__x_min; (void)__x_max;(void)__y_min; (void)__y_max;(void)__z_min; (void)__z_max;(void)__c_min; (void)__c_max;(void)__d_min; (void)__d_max;  (void)__jdf2c_x_start; (void)__jdf2c_x_end; (void)__jdf2c_x_inc;  (void)__jdf2c_y_start; (void)__jdf2c_y_end; (void)__jdf2c_y_inc;  (void)__jdf2c_z_start; (void)__jdf2c_z_end; (void)__jdf2c_z_inc;  (void)__jdf2c_c_start; (void)__jdf2c_c_end; (void)__jdf2c_c_inc;  (void)__jdf2c_d_start; (void)__jdf2c_d_end; (void)__jdf2c_d_inc;  (void)assignments; (void)__parsec_tp; (void)es;
  if(1 == parsec_atomic_fetch_dec_int32(&__parsec_tp->sync_point)) {
    /* Last initialization task complete. Update the number of tasks. */
    __parsec_tp->super.super.tdm.module->taskpool_addto_nb_tasks(&__parsec_tp->super.super, __parsec_tp->initial_number_tasks);
    parsec_mfence();  /* write memory barrier to guarantee that the scheduler gets the correct number of tasks */
    parsec_taskpool_enable((parsec_taskpool_t*)__parsec_tp, &__parsec_tp->startup_queue,
                           (parsec_task_t*)this_task, es, __parsec_tp->super.super.nb_pending_actions);
    __parsec_tp->super.super.tdm.module->taskpool_ready(&__parsec_tp->super.super);
#if defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)
    PARSEC_PROFILING_TRACE(es->es_profile,
                           this_task->taskpool->profiling_array[2 * this_task->task_class->task_class_id + 1],
                           0,
                           this_task->taskpool->taskpool_id, NULL);
#endif /* defined(PARSEC_PROF_TRACE) && defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */
    if( 1 >= __parsec_tp->super.super.nb_pending_actions ) {
        /* if no tasks will be generated let's prevent the runtime from calling the hook and instead go directly to complete the task */
        this_task->status = PARSEC_TASK_STATUS_COMPLETE;
    }
    return PARSEC_HOOK_RETURN_DONE;
  }
  return PARSEC_HOOK_RETURN_ASYNC;
}

static int __jdf2c_startup_FillGrid(parsec_execution_stream_t * es, __parsec_LBM_FillGrid_task_t *this_task)
{
  __parsec_LBM_FillGrid_task_t* new_task;
  __parsec_LBM_internal_taskpool_t* __parsec_tp = (__parsec_LBM_internal_taskpool_t*)this_task->taskpool;
  parsec_context_t *context = __parsec_tp->super.super.context;
  int vpid = 0, nb_tasks = 0;
  size_t total_nb_tasks = 0;
  parsec_list_item_t* pready_ring[context->nb_vp];
  int restore_context = 0;
  int x = this_task->locals.x.value;  /* retrieve value saved during the last iteration */
  int y = this_task->locals.y.value;  /* retrieve value saved during the last iteration */
  int z = this_task->locals.z.value;  /* retrieve value saved during the last iteration */
  int c = this_task->locals.c.value;  /* retrieve value saved during the last iteration */
  int d = this_task->locals.d.value;  /* retrieve value saved during the last iteration */
  for(int _i = 0; _i < context->nb_vp; pready_ring[_i++] = NULL );
  if( 0 != this_task->locals.reserved[0].value ) {
    this_task->locals.reserved[0].value = 1; /* reset the submission process */
    restore_context = 1;
    goto restore_context_0;
  }
  this_task->locals.reserved[0].value = 1; /* a sane default value */
  for(this_task->locals.x.value = x = 0;
      this_task->locals.x.value <= (subgrid_number_x - 1);
      this_task->locals.x.value += 1, x = this_task->locals.x.value) {
    for(this_task->locals.y.value = y = 0;
        this_task->locals.y.value <= (subgrid_number_y - 1);
        this_task->locals.y.value += 1, y = this_task->locals.y.value) {
      for(this_task->locals.z.value = z = 0;
          this_task->locals.z.value <= (subgrid_number_z - 1);
          this_task->locals.z.value += 1, z = this_task->locals.z.value) {
        for(this_task->locals.c.value = c = 0;
            this_task->locals.c.value <= (conservatives_number - 1);
            this_task->locals.c.value += 1, c = this_task->locals.c.value) {
          for(this_task->locals.d.value = d = 0;
              this_task->locals.d.value <= (directions_number - 1);
              this_task->locals.d.value += 1, d = this_task->locals.d.value) {
            if( !FillGrid_pred(x, y, z, c, d) ) continue;
  __parsec_LBM_FillGrid_parsec_assignment_t __jdf2c__tmp_locals = *(__parsec_LBM_FillGrid_parsec_assignment_t*)&this_task->locals;
  (void)__jdf2c__tmp_locals;
  /* Flow for INITIAL_GRID is always a memory reference */
            if( NULL != ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of ) {
              vpid = ((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC)->vpid_of((parsec_data_collection_t*)__parsec_tp->super._g_descGridDC, x, y, z, c, d);
              assert(context->nb_vp >= vpid);
            } else {
              vpid = (vpid + 1) % context->nb_vp;  /* spread the initial joy */
            }
            new_task = (__parsec_LBM_FillGrid_task_t*)parsec_thread_mempool_allocate( context->virtual_processes[vpid]->execution_streams[0]->context_mempool );
            new_task->status = PARSEC_TASK_STATUS_NONE;
            /* Copy only the valid elements from this_task to new_task one */
            new_task->taskpool   = this_task->taskpool;
            new_task->task_class = __parsec_tp->super.super.task_classes_array[LBM_FillGrid.task_class_id];
            new_task->chore_mask   = PARSEC_DEV_ALL;
            new_task->locals.x.value = this_task->locals.x.value;
            new_task->locals.y.value = this_task->locals.y.value;
            new_task->locals.z.value = this_task->locals.z.value;
            new_task->locals.c.value = this_task->locals.c.value;
            new_task->locals.d.value = this_task->locals.d.value;
            PARSEC_LIST_ITEM_SINGLETON(new_task);
            new_task->priority = __parsec_tp->super.super.priority;
            new_task->repo_entry = NULL;
            new_task->data._f_INITIAL_GRID.source_repo_entry = NULL;
            new_task->data._f_INITIAL_GRID.source_repo       = NULL;
            new_task->data._f_INITIAL_GRID.data_in         = NULL;
            new_task->data._f_INITIAL_GRID.data_out        = NULL;
            new_task->data._f_INITIAL_GRID.fulfill         = 0;
#if defined(PARSEC_DEBUG_NOISIER)
            {
              char tmp[128];
              PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Add startup task %s to vpid %d",
                     parsec_task_snprintf(tmp, 128, (parsec_task_t*)new_task), vpid);
            }
#endif
            parsec_dependencies_mark_task_as_startup((parsec_task_t*)new_task, es);
            pready_ring[vpid] = parsec_list_item_ring_push_sorted(pready_ring[vpid],
                                                                  (parsec_list_item_t*)new_task,
                                                                  parsec_execution_context_priority_comparator);
            nb_tasks++;
           restore_context_0:  /* we jump here just so that we have code after the label */
            restore_context = 0;
            (void)restore_context;
            if( nb_tasks > this_task->locals.reserved[0].value ) {
              if( (size_t)this_task->locals.reserved[0].value < parsec_task_startup_iter ) this_task->locals.reserved[0].value <<= 1;
              __parsec_schedule_vp(es, (parsec_task_t**)pready_ring, 0);
              total_nb_tasks += nb_tasks;
              nb_tasks = 0;
              if( total_nb_tasks > parsec_task_startup_chunk ) {  /* stop here and request to be rescheduled */
                return PARSEC_HOOK_RETURN_AGAIN;
              }
            }
          } /* Loop on normal range d */
        } /* Loop on normal range c */
      } /* Loop on normal range z */
    } /* Loop on normal range y */
  } /* Loop on normal range x */
  (void)vpid;
  if( 0 != nb_tasks ) {
    __parsec_schedule_vp(es, (parsec_task_t**)pready_ring, 0);
    nb_tasks = 0;
  }
  return PARSEC_HOOK_RETURN_DONE;
}

static const __parsec_chore_t __LBM_FillGrid_chores[] ={
    { .type     = PARSEC_DEV_CPU,
      .evaluate = NULL,
      .hook     = (parsec_hook_t*)hook_of_LBM_FillGrid },
    { .type     = PARSEC_DEV_NONE,
      .evaluate = NULL,
      .hook     = (parsec_hook_t*)NULL },  /* End marker */
};

static const parsec_task_class_t LBM_FillGrid = {
  .name = "FillGrid",
  .task_class_id = 0,
  .nb_flows = 1,
  .nb_parameters = 5,
  .nb_locals = 5,
  .task_class_type = PARSEC_TASK_CLASS_TYPE_PTG,
  .params = { &symb_LBM_FillGrid_x, &symb_LBM_FillGrid_y, &symb_LBM_FillGrid_z, &symb_LBM_FillGrid_c, &symb_LBM_FillGrid_d, NULL },
  .locals = { &symb_LBM_FillGrid_x, &symb_LBM_FillGrid_y, &symb_LBM_FillGrid_z, &symb_LBM_FillGrid_c, &symb_LBM_FillGrid_d, NULL },
  .data_affinity = (parsec_data_ref_fn_t*)affinity_of_LBM_FillGrid,
  .initial_data = (parsec_data_ref_fn_t*)affinity_of_LBM_FillGrid,
  .final_data = (parsec_data_ref_fn_t*)affinity_of_LBM_FillGrid,
  .priority = NULL,
  .properties = properties_of_LBM_FillGrid,
#if MAX_PARAM_COUNT < 1  /* number of read flows of FillGrid */
  #error Too many read flows for task FillGrid
#endif  /* MAX_PARAM_COUNT */
#if MAX_PARAM_COUNT < 1  /* number of write flows of FillGrid */
  #error Too many write flows for task FillGrid
#endif  /* MAX_PARAM_COUNT */
  .in = { &flow_of_LBM_FillGrid_for_INITIAL_GRID, NULL },
  .out = { &flow_of_LBM_FillGrid_for_INITIAL_GRID, NULL },
  .flags = 0x0 | PARSEC_HAS_IN_IN_DEPENDENCIES | PARSEC_USE_DEPS_MASK,
  .dependencies_goal = 0x1,
  .make_key = __jdf2c_make_key_FillGrid,
  .task_snprintf = parsec_task_snprintf,
  .key_functions = &__jdf2c_key_fns_FillGrid,
  .fini = (parsec_hook_t*)NULL,
  .incarnations = __LBM_FillGrid_chores,
  .find_deps = parsec_hash_find_deps,
  .update_deps = parsec_update_deps_with_mask,
  .iterate_successors = (parsec_traverse_function_t*)iterate_successors_of_LBM_FillGrid,
  .iterate_predecessors = (parsec_traverse_function_t*)NULL,
  .release_deps = (parsec_release_deps_t*)release_deps_of_LBM_FillGrid,
  .prepare_output = (parsec_hook_t*)NULL,
  .prepare_input = (parsec_hook_t*)data_lookup_of_LBM_FillGrid,
  .get_datatype = (parsec_datatype_lookup_t*)datatype_lookup_of_LBM_FillGrid,
  .complete_execution = (parsec_hook_t*)complete_hook_of_LBM_FillGrid,
  .release_task = &release_task_of_LBM_FillGrid,
#if defined(PARSEC_SIM)
  .sim_cost_fct = (parsec_sim_cost_fct_t*)NULL,
#endif
};


static const parsec_task_class_t *LBM_task_classes[] = {
  &LBM_FillGrid,
  &LBM_LBM_STEP,
  &LBM_Exchange,
  &LBM_WriteBack
};

static void LBM_startup(parsec_context_t *context, __parsec_LBM_internal_taskpool_t *__parsec_tp, parsec_list_item_t ** ready_tasks)
{
  uint32_t i, supported_dev = 0;
 
  for( i = 0; i < parsec_nb_devices; i++ ) {
    if( !(__parsec_tp->super.super.devices_index_mask & (1<<i)) ) continue;
    parsec_device_module_t* device = parsec_mca_device_get(i);
    parsec_data_collection_t* parsec_dc;
 
    if(NULL == device) continue;
    if(NULL != device->taskpool_register)
      if( PARSEC_SUCCESS != device->taskpool_register(device, (parsec_taskpool_t*)__parsec_tp) ) {
        parsec_debug_verbose(5, parsec_debug_output, "Device %s refused to register taskpool %p", device->name, __parsec_tp);
        __parsec_tp->super.super.devices_index_mask &= ~(1 << device->device_index);
        continue;
      }
    if(NULL != device->memory_register) {  /* Register all the data */
      parsec_dc = (parsec_data_collection_t*)__parsec_tp->super._g_descGridDC;
      if( (NULL != parsec_dc->register_memory) &&
          (PARSEC_SUCCESS != parsec_dc->register_memory(parsec_dc, device)) ) {
        parsec_debug_verbose(3, parsec_debug_output, "Device %s refused to register memory for data %s (%p) from taskpool %p",
                     device->name, parsec_dc->key_base, parsec_dc, __parsec_tp);
        __parsec_tp->super.super.devices_index_mask &= ~(1 << device->device_index);
      }
    }
    supported_dev |= device->type;
  }
  /* Remove all the chores without a backend device */
  for( i = 0; i < PARSEC_LBM_NB_TASK_CLASSES; i++ ) {
    parsec_task_class_t* tc = (parsec_task_class_t*)__parsec_tp->super.super.task_classes_array[i];
    __parsec_chore_t* chores = (__parsec_chore_t*)tc->incarnations;
    uint32_t idx = 0, j;
    for( j = 0; NULL != chores[j].hook; j++ ) {
      if( !(supported_dev & chores[j].type) ) continue;
      if( j != idx ) {
        chores[idx] = chores[j];
        parsec_debug_verbose(20, parsec_debug_output, "Device type %i disabled for function %s"
, chores[j].type, tc->name);
      }
      idx++;
    }
    chores[idx].type     = PARSEC_DEV_NONE;
    chores[idx].evaluate = NULL;
    chores[idx].hook     = NULL;
    /* Create the initialization tasks for each taskclass */
    parsec_task_t* task = (parsec_task_t*)parsec_thread_mempool_allocate(context->virtual_processes[0]->execution_streams[0]->context_mempool);
    task->taskpool = (parsec_taskpool_t *)__parsec_tp;
    task->chore_mask = PARSEC_DEV_CPU;
    task->status = PARSEC_TASK_STATUS_NONE;
    memset(&task->locals, 0, sizeof(parsec_assignment_t) * MAX_LOCAL_COUNT);
    PARSEC_LIST_ITEM_SINGLETON(task);
    task->priority = -1;
    task->task_class = task->taskpool->task_classes_array[PARSEC_LBM_NB_TASK_CLASSES + i];
    int where = i % context->nb_vp;
    if( NULL == ready_tasks[where] ) ready_tasks[where] = &task->super;
    else ready_tasks[where] = parsec_list_item_ring_push(ready_tasks[where], &task->super);
  }
}
static void __parsec_LBM_internal_destructor( __parsec_LBM_internal_taskpool_t *__parsec_tp )
{
  uint32_t i;
  parsec_taskpool_unregister( &__parsec_tp->super.super );
  for( i = 0; i < (uint32_t)(2 * __parsec_tp->super.super.nb_task_classes); i++ ) {  /* Extra startup function added at the end */
    parsec_task_class_t* tc = (parsec_task_class_t*)__parsec_tp->super.super.task_classes_array[i];
    free((void*)tc->incarnations);
    free(tc);
  }
  free(__parsec_tp->super.super.task_classes_array); __parsec_tp->super.super.task_classes_array = NULL;
  __parsec_tp->super.super.nb_task_classes = 0;

  for(i = 0; i < (uint32_t)__parsec_tp->super.arenas_datatypes_size; i++) {
    if( NULL != __parsec_tp->super.arenas_datatypes[i].arena ) {
      PARSEC_OBJ_RELEASE(__parsec_tp->super.arenas_datatypes[i].arena);
    }
  }
  /* Destroy the data repositories for this object */
   data_repo_destroy_nothreadsafe(__parsec_tp->repositories[3]);  /* WriteBack */
   data_repo_destroy_nothreadsafe(__parsec_tp->repositories[2]);  /* Exchange */
   data_repo_destroy_nothreadsafe(__parsec_tp->repositories[1]);  /* LBM_STEP */
   data_repo_destroy_nothreadsafe(__parsec_tp->repositories[0]);  /* FillGrid */
  /* Release the dependencies arrays for this object */
  parsec_hash_table_fini( (parsec_hash_table_t*)__parsec_tp->super.super.dependencies_array[3] );
  PARSEC_OBJ_RELEASE(__parsec_tp->super.super.dependencies_array[3]);
  __parsec_tp->super.super.dependencies_array[3] = NULL;
  parsec_hash_table_fini( (parsec_hash_table_t*)__parsec_tp->super.super.dependencies_array[2] );
  PARSEC_OBJ_RELEASE(__parsec_tp->super.super.dependencies_array[2]);
  __parsec_tp->super.super.dependencies_array[2] = NULL;
  parsec_hash_table_fini( (parsec_hash_table_t*)__parsec_tp->super.super.dependencies_array[1] );
  PARSEC_OBJ_RELEASE(__parsec_tp->super.super.dependencies_array[1]);
  __parsec_tp->super.super.dependencies_array[1] = NULL;
  parsec_hash_table_fini( (parsec_hash_table_t*)__parsec_tp->super.super.dependencies_array[0] );
  PARSEC_OBJ_RELEASE(__parsec_tp->super.super.dependencies_array[0]);
  __parsec_tp->super.super.dependencies_array[0] = NULL;
  free( __parsec_tp->super.super.dependencies_array );
  __parsec_tp->super.super.dependencies_array = NULL;
  /* Unregister all the data */
  uint32_t _i;
  for( _i = 0; _i < parsec_nb_devices; _i++ ) {
    parsec_device_module_t* device;
    parsec_data_collection_t* parsec_dc;
    if(!(__parsec_tp->super.super.devices_index_mask & (1 << _i))) continue;
    if((NULL == (device = parsec_mca_device_get(_i))) || (NULL == device->memory_unregister)) continue;
    parsec_dc = (parsec_data_collection_t*)__parsec_tp->super._g_descGridDC;
  if( NULL != parsec_dc->unregister_memory ) { (void)parsec_dc->unregister_memory(parsec_dc, device); };
}
  /* Unregister the taskpool from the devices */
  for( i = 0; i < parsec_nb_devices; i++ ) {
    if(!(__parsec_tp->super.super.devices_index_mask & (1 << i))) continue;
    __parsec_tp->super.super.devices_index_mask ^= (1 << i);
    parsec_device_module_t* device = parsec_mca_device_get(i);
    if((NULL == device) || (NULL == device->taskpool_unregister)) continue;
    if( PARSEC_SUCCESS != device->taskpool_unregister(device, &__parsec_tp->super.super) ) continue;
  }
  free(__parsec_tp->super.super.taskpool_name); __parsec_tp->super.super.taskpool_name = NULL;
}

void __parsec_LBM_internal_constructor(__parsec_LBM_internal_taskpool_t* __parsec_tp)
{
  parsec_task_class_t* tc;
  uint32_t i, j;

  __parsec_tp->super.super.nb_task_classes = PARSEC_LBM_NB_TASK_CLASSES;
  __parsec_tp->super.super.devices_index_mask = PARSEC_DEVICES_ALL;
  __parsec_tp->super.super.taskpool_name = strdup("LBM");
  parsec_termdet_open_module(&__parsec_tp->super.super, "local");
  __parsec_tp->super.super.tdm.module->monitor_taskpool(&__parsec_tp->super.super,
                                                        parsec_taskpool_termination_detected);
  __parsec_tp->super.super.update_nb_runtime_task = parsec_add_fetch_runtime_task;
  __parsec_tp->super.super.dependencies_array = (void **)
              calloc(__parsec_tp->super.super.nb_task_classes, sizeof(void*));
  /* Twice the size to hold the startup tasks function_t */
  __parsec_tp->super.super.task_classes_array = (const parsec_task_class_t**)
              calloc((2 * PARSEC_LBM_NB_TASK_CLASSES + 1), sizeof(parsec_task_class_t*));
  __parsec_tp->super.super.tdm.module->taskpool_addto_runtime_actions(&__parsec_tp->super.super, PARSEC_LBM_NB_TASK_CLASSES);  /* for the startup tasks */
  __parsec_tp->super.super.taskpool_type = PARSEC_TASKPOOL_TYPE_PTG;
  __parsec_tp->sync_point = __parsec_tp->super.super.nb_task_classes;
  __parsec_tp->initial_number_tasks = 0;
  __parsec_tp->startup_queue = NULL;
  for( i = 0; i < __parsec_tp->super.super.nb_task_classes; i++ ) {
    __parsec_tp->super.super.task_classes_array[i] = tc = malloc(sizeof(parsec_task_class_t));
    memcpy(tc, LBM_task_classes[i], sizeof(parsec_task_class_t));
    for( j = 0; NULL != tc->incarnations[j].hook; j++);  /* compute the number of incarnations */
    tc->incarnations = (__parsec_chore_t*)malloc((j+1) * sizeof(__parsec_chore_t));
        memcpy((__parsec_chore_t*)tc->incarnations, LBM_task_classes[i]->incarnations, (j+1) * sizeof(__parsec_chore_t));

    /* Add a placeholder for initialization and startup task */
    __parsec_tp->super.super.task_classes_array[__parsec_tp->super.super.nb_task_classes+i] = tc = (parsec_task_class_t*)malloc(sizeof(parsec_task_class_t));
    memcpy(tc, (void*)&__parsec_generic_startup, sizeof(parsec_task_class_t));
    tc->task_class_id = __parsec_tp->super.super.nb_task_classes + i;
    tc->incarnations = (__parsec_chore_t*)malloc(2 * sizeof(__parsec_chore_t));
    memcpy((__parsec_chore_t*)tc->incarnations, (void*)__parsec_generic_startup.incarnations, 2 * sizeof(__parsec_chore_t));
    tc->release_task = parsec_release_task_to_mempool_and_count_as_runtime_tasks;
  }
  /* Startup task for WriteBack */
  tc = (parsec_task_class_t *)__parsec_tp->super.super.task_classes_array[__parsec_tp->super.super.nb_task_classes+0];
  tc->name = "Startup for WriteBack";
  tc->prepare_input = (parsec_hook_t*)LBM_WriteBack_internal_init;
  /* Startup task for Exchange */
  tc = (parsec_task_class_t *)__parsec_tp->super.super.task_classes_array[__parsec_tp->super.super.nb_task_classes+1];
  tc->name = "Startup for Exchange";
  tc->prepare_input = (parsec_hook_t*)LBM_Exchange_internal_init;
  /* Startup task for LBM_STEP */
  tc = (parsec_task_class_t *)__parsec_tp->super.super.task_classes_array[__parsec_tp->super.super.nb_task_classes+2];
  tc->name = "Startup for LBM_STEP";
  tc->prepare_input = (parsec_hook_t*)LBM_LBM_STEP_internal_init;
  /* Startup task for FillGrid */
  tc = (parsec_task_class_t *)__parsec_tp->super.super.task_classes_array[__parsec_tp->super.super.nb_task_classes+3];
  tc->name = "Startup for FillGrid";
  tc->prepare_input = (parsec_hook_t*)LBM_FillGrid_internal_init;
  ((__parsec_chore_t*)&tc->incarnations[0])->hook = (parsec_hook_t *)__jdf2c_startup_FillGrid;
  /* Compute the number of arenas_datatypes: */
  /*   PARSEC_LBM_DEFAULT_ARENA  ->  0 */
  __parsec_tp->super.arenas_datatypes_size = 1;
  memset(&__parsec_tp->super.arenas_datatypes[0], 0, __parsec_tp->super.arenas_datatypes_size*sizeof(parsec_arena_datatype_t));
  /* If profiling is enabled, the keys for profiling */
#  if defined(PARSEC_PROF_TRACE)
  __parsec_tp->super.super.profiling_array = LBM_profiling_array;
  if( -1 == LBM_profiling_array[0] ) {
    #if defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)
parsec_profiling_add_dictionary_keyword("LBM::WriteBack::internal init", "fill:34D8D8",
                                       0,
                                       NULL,
                                       (int*)&__parsec_tp->super.super.profiling_array[0 + 2 * LBM_WriteBack.task_class_id  + 2 * PARSEC_LBM_NB_TASK_CLASSES/* WriteBack (internal init) start key */],
                                       (int*)&__parsec_tp->super.super.profiling_array[1 + 2 * LBM_WriteBack.task_class_id  + 2 * PARSEC_LBM_NB_TASK_CLASSES/* WriteBack (internal init) end key */]);
#endif /* defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */
parsec_profiling_add_dictionary_keyword("LBM::WriteBack", "fill:CC2828",
                                       sizeof(parsec_task_prof_info_t)+5*sizeof(parsec_assignment_t),
                                       "dc_key{uint64_t};priority{int32_t};dc_dataid{uint32_t};tcid{int32_t};trc{int32_t};x{int32_t};y{int32_t};z{int32_t};c{int32_t};d{int32_t}",
                                       (int*)&__parsec_tp->super.super.profiling_array[0 + 2 * LBM_WriteBack.task_class_id /* WriteBack start key */],
                                       (int*)&__parsec_tp->super.super.profiling_array[1 + 2 * LBM_WriteBack.task_class_id /* WriteBack end key */]);

    #if defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)
parsec_profiling_add_dictionary_keyword("LBM::Exchange::internal init", "fill:8634D8",
                                       0,
                                       NULL,
                                       (int*)&__parsec_tp->super.super.profiling_array[0 + 2 * LBM_Exchange.task_class_id  + 2 * PARSEC_LBM_NB_TASK_CLASSES/* Exchange (internal init) start key */],
                                       (int*)&__parsec_tp->super.super.profiling_array[1 + 2 * LBM_Exchange.task_class_id  + 2 * PARSEC_LBM_NB_TASK_CLASSES/* Exchange (internal init) end key */]);
#endif /* defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */
parsec_profiling_add_dictionary_keyword("LBM::Exchange", "fill:7ACC28",
                                       sizeof(parsec_task_prof_info_t)+8*sizeof(parsec_assignment_t),
                                       "dc_key{uint64_t};priority{int32_t};dc_dataid{uint32_t};tcid{int32_t};trc{int32_t};x{int32_t};y{int32_t};z{int32_t};s{int32_t};conservative{int32_t};direction{int32_t};dimension{int32_t};side{int32_t}",
                                       (int*)&__parsec_tp->super.super.profiling_array[0 + 2 * LBM_Exchange.task_class_id /* Exchange start key */],
                                       (int*)&__parsec_tp->super.super.profiling_array[1 + 2 * LBM_Exchange.task_class_id /* Exchange end key */]);

    #if defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)
parsec_profiling_add_dictionary_keyword("LBM::LBM_STEP::internal init", "fill:D83434",
                                       0,
                                       NULL,
                                       (int*)&__parsec_tp->super.super.profiling_array[0 + 2 * LBM_LBM_STEP.task_class_id  + 2 * PARSEC_LBM_NB_TASK_CLASSES/* LBM_STEP (internal init) start key */],
                                       (int*)&__parsec_tp->super.super.profiling_array[1 + 2 * LBM_LBM_STEP.task_class_id  + 2 * PARSEC_LBM_NB_TASK_CLASSES/* LBM_STEP (internal init) end key */]);
#endif /* defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */
parsec_profiling_add_dictionary_keyword("LBM::LBM_STEP", "fill:28CCCC",
                                       sizeof(parsec_task_prof_info_t)+4*sizeof(parsec_assignment_t),
                                       "dc_key{uint64_t};priority{int32_t};dc_dataid{uint32_t};tcid{int32_t};trc{int32_t};x{int32_t};y{int32_t};z{int32_t};s{int32_t}",
                                       (int*)&__parsec_tp->super.super.profiling_array[0 + 2 * LBM_LBM_STEP.task_class_id /* LBM_STEP start key */],
                                       (int*)&__parsec_tp->super.super.profiling_array[1 + 2 * LBM_LBM_STEP.task_class_id /* LBM_STEP end key */]);

    #if defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT)
parsec_profiling_add_dictionary_keyword("LBM::FillGrid::internal init", "fill:86D834",
                                       0,
                                       NULL,
                                       (int*)&__parsec_tp->super.super.profiling_array[0 + 2 * LBM_FillGrid.task_class_id  + 2 * PARSEC_LBM_NB_TASK_CLASSES/* FillGrid (internal init) start key */],
                                       (int*)&__parsec_tp->super.super.profiling_array[1 + 2 * LBM_FillGrid.task_class_id  + 2 * PARSEC_LBM_NB_TASK_CLASSES/* FillGrid (internal init) end key */]);
#endif /* defined(PARSEC_PROF_TRACE_PTG_INTERNAL_INIT) */
parsec_profiling_add_dictionary_keyword("LBM::FillGrid", "fill:7A28CC",
                                       sizeof(parsec_task_prof_info_t)+5*sizeof(parsec_assignment_t),
                                       "dc_key{uint64_t};priority{int32_t};dc_dataid{uint32_t};tcid{int32_t};trc{int32_t};x{int32_t};y{int32_t};z{int32_t};c{int32_t};d{int32_t}",
                                       (int*)&__parsec_tp->super.super.profiling_array[0 + 2 * LBM_FillGrid.task_class_id /* FillGrid start key */],
                                       (int*)&__parsec_tp->super.super.profiling_array[1 + 2 * LBM_FillGrid.task_class_id /* FillGrid end key */]);

  }
#  endif /* defined(PARSEC_PROF_TRACE) */
  __parsec_tp->super.super.repo_array = __parsec_tp->repositories;
  __parsec_tp->super.super.startup_hook = (parsec_startup_fn_t)LBM_startup;
  (void)parsec_taskpool_reserve_id((parsec_taskpool_t*)__parsec_tp);
}

  /** Generate class declaration and instance using the above constructor and destructor */
  PARSEC_OBJ_CLASS_DECLARATION(__parsec_LBM_internal_taskpool_t);
  PARSEC_OBJ_CLASS_INSTANCE(__parsec_LBM_internal_taskpool_t, parsec_LBM_taskpool_t,
                            __parsec_LBM_internal_constructor, __parsec_LBM_internal_destructor);

#undef descGridDC
#undef gridParameters
#undef rank
#undef nodes
#undef subgrid_number_x
#undef subgrid_number_y
#undef subgrid_number_z
#undef tile_size_x
#undef tile_size_y
#undef tile_size_z
#undef conservatives_number
#undef directions_number
#undef overlap_x
#undef overlap_y
#undef overlap_z
#undef number_of_steps
#undef CuHI

parsec_LBM_taskpool_t *parsec_LBM_new(parsec_multidimensional_grid_t* descGridDC /* data descGridDC */, Grid gridParameters, int rank, int nodes, int subgrid_number_x, int subgrid_number_y, int subgrid_number_z, int tile_size_x, int tile_size_y, int tile_size_z, int conservatives_number, int directions_number, int overlap_x, int overlap_y, int overlap_z, int number_of_steps, parsec_info_id_t CuHI)
{
  __parsec_LBM_internal_taskpool_t *__parsec_tp = PARSEC_OBJ_NEW(__parsec_LBM_internal_taskpool_t);
  /* Dump the hidden parameters with default values */

  /* Now the Parameter-dependent structures: */
  __parsec_tp->super._g_descGridDC = descGridDC;
  __parsec_tp->super._g_gridParameters = gridParameters;
  __parsec_tp->super._g_rank = rank;
  __parsec_tp->super._g_nodes = nodes;
  __parsec_tp->super._g_subgrid_number_x = subgrid_number_x;
  __parsec_tp->super._g_subgrid_number_y = subgrid_number_y;
  __parsec_tp->super._g_subgrid_number_z = subgrid_number_z;
  __parsec_tp->super._g_tile_size_x = tile_size_x;
  __parsec_tp->super._g_tile_size_y = tile_size_y;
  __parsec_tp->super._g_tile_size_z = tile_size_z;
  __parsec_tp->super._g_conservatives_number = conservatives_number;
  __parsec_tp->super._g_directions_number = directions_number;
  __parsec_tp->super._g_overlap_x = overlap_x;
  __parsec_tp->super._g_overlap_y = overlap_y;
  __parsec_tp->super._g_overlap_z = overlap_z;
  __parsec_tp->super._g_number_of_steps = number_of_steps;
  __parsec_tp->super._g_CuHI = CuHI;
  PARSEC_AYU_REGISTER_TASK(&LBM_WriteBack);
  PARSEC_AYU_REGISTER_TASK(&LBM_Exchange);
  PARSEC_AYU_REGISTER_TASK(&LBM_LBM_STEP);
  PARSEC_AYU_REGISTER_TASK(&LBM_FillGrid);
  __parsec_tp->super.super.startup_hook = (parsec_startup_fn_t)LBM_startup;
  (void)parsec_taskpool_reserve_id((parsec_taskpool_t*)__parsec_tp);
/* Prevent warnings related to not used hidden global variables */
;
  return (parsec_LBM_taskpool_t*)__parsec_tp;
}

#line 431 "LBM.jdf"

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rc;
    int rank, world;
    parsec_LBM_taskpool_t *tp;
    //int mycounter;

    parsec_arena_datatype_t adt;
    parsec_datatype_t otype;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif

    // LBM parameters
    Grid grid = newGrid(rank, world);

    double delta_x = grid.physicalSize[0] / (double)grid.size[0];
    double dt = CFL * delta_x / (ALPHA > BETA ? ALPHA : BETA);

    double tmax = 1;
    int number_of_steps = (int)((tmax-EPSILON) / dt) + 1;
    number_of_steps = 3;

    parsec = parsec_init(-1, &argc, &argv);

    int nodes = world;

    #if defined(PARSEC_HAVE_CUDA)
    parsec_info_id_t CuHI = parsec_info_register(&parsec_per_stream_infos, "CUBLAS::HANDLE",
                                                 destroy_cublas_handle, NULL,
                                                 create_cublas_handle, NULL,
                                                 NULL);
    assert(CuHI != -1);
#else
    int CuHI = -1;
#endif

    parsec_translate_matrix_type(PARSEC_MATRIX_DOUBLE, &otype);
    parsec_add2arena_rect(&adt, otype,
                                 grid.subgridTrueSize[0], grid.subgridTrueSize[1]*grid.subgridTrueSize[2]*grid.conservativesNumber*grid.directionsNumber, grid.subgridTrueSize[0]);

    tp = (parsec_LBM_taskpool_t*)parsec_LBM_new(
                                grid.desc, grid,
                                rank, world,
                                grid.subgridNumber[0], grid.subgridNumber[1], grid.subgridNumber[2],
                                grid.subgridTrueSize[0], grid.subgridTrueSize[1], grid.subgridTrueSize[2], grid.conservativesNumber, grid.directionsNumber,
                                grid.overlapSize[0], grid.overlapSize[1], grid.overlapSize[2],
                                number_of_steps,
                                CuHI);

    assert( NULL != tp );
    tp->arenas_datatypes[PARSEC_LBM_DEFAULT_ADT_IDX] = adt;
    PARSEC_OBJ_RETAIN(adt.arena);

    rc = parsec_context_add_taskpool( parsec, (parsec_taskpool_t*)tp );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

printf("final : %f\n", ((double*)(grid.desc[0].grid))[0]);


    parsec_taskpool_free(&tp->super);

    for(int i=0;i<grid.conservativesNumber*grid.directionsNumber;++i)
    {
        parsec_data_free(grid.desc[i].grid);
        parsec_grid_destroy(&grid.desc[i]);
    }

    parsec_fini(&parsec);
#if defined(PARSEC_HAVE_MPI)
    MPI_Finalize();
#endif

    return 0;
}

#line 11759 "LBM.c"
PARSEC_OBJ_CLASS_INSTANCE(parsec_LBM_taskpool_t, parsec_taskpool_t, NULL, NULL);

