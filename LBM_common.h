#ifndef LBM_COMMON_H_INCLUDED
#define LBM_COMMON_H_INCLUDED

#define PRECISION float
// #define DOUBLE_PRECISION

// block/thread num for read_horizontal_slices, write_horizontal_slices, read_vertical_slices, lbm_step

// Read vertical slices
#ifndef READ_VERTICAL_SLICES_BLOCK_NUM
#define READ_VERTICAL_SLICES_BLOCK_NUM 256
#endif
#ifndef READ_VERTICAL_SLICES_THREAD_NUM
#define READ_VERTICAL_SLICES_THREAD_NUM 256
#endif
// // LBM step
// #ifndef LBM_STEP_BLOCK_NUM
// #define LBM_STEP_BLOCK_NUM 1024
// #endif
// #ifndef LBM_STEP_THREAD_NUM
// #define LBM_STEP_THREAD_NUM 256
// #endif

// LBM defines

#define EPSILON 0.000000001

#define CFL 0.45
// #define ALPHA 0.9 // TODO change to match cylinder test case
// #define BETA 0.9
#define MAXIMUM_VELOCITY ((PRECISION)1.0)

#define OMEGA_RELAX ((PRECISION)1.9)

#define CYLINDER_RADIUS ((PRECISION)0.3)
#define CYLINDER_CENTER_X ((PRECISION)0.0)
#define CYLINDER_CENTER_Y ((PRECISION)0.0)

// HPC defines

#define DIMENSIONS_NUMBER 2


#define MAX_SUBGRIDS 256

#define PROBLEM_SIZE_X (1024)
#define PROBLEM_SIZE_Y (512)
// #define PROBLEM_SIZE_X (384)
// #define PROBLEM_SIZE_Y (384/2)
#define PROBLEM_SIZE_Z (1)

#define SAVED_DATA_SIZE_X (1024)
#define SAVED_DATA_SIZE_Y (512)
// #define SAVED_DATA_SIZE_X (PROBLEM_SIZE_X)
// #define SAVED_DATA_SIZE_Y (PROBLEM_SIZE_Y)

#define SAVE_STENCIL_X ((PROBLEM_SIZE_X)/(SAVED_DATA_SIZE_X))
#define SAVE_STENCIL_Y ((PROBLEM_SIZE_Y)/(SAVED_DATA_SIZE_Y))

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
	int subgridOwnedSize[DIMENSIONS_NUMBER]; // cells owned uniquely by this subgrid
	int subgridLogicalSize[DIMENSIONS_NUMBER]; // cells that need computation on this subgrid (includes cells that share the same computation)
	int subgridTrueSize[DIMENSIONS_NUMBER]; // all the cells of the subgrid (including ghost cells = halo = overlap)
    int sharedLayers[DIMENSIONS_NUMBER][2]; // number of values that are shared with the neighbor [dim][dir] (shared meaning that the values are computed twice and not exchanged)
	int overlapSize[DIMENSIONS_NUMBER]; // depth of the overlap on each dimension

    int sizeOfSavedData[DIMENSIONS_NUMBER];
    int saveStencilSize[DIMENSIONS_NUMBER];

    // Conceptually, the grid is a 5D array: (x, y, z, c, d)

	int cellsPerSubgrid;
	int subgridsNumber;
	int currentSubgrid;

	//double mass[MAX_SUBGRIDS];

    // physical coords of the problem, used by the physical model
	PRECISION physicalMinCoords[DIMENSIONS_NUMBER];
	PRECISION physicalSize[DIMENSIONS_NUMBER];

    void *desc;
} Grid;

typedef struct SubgridArray
{
    PRECISION *subgrid[3];
} SubgridArray;


void d2q9_initial_value_d_caller(Grid grid, PRECISION *subgrid, int subgridX, int subgridY, int d);
void d2q9_save_reduce_caller(Grid grid, PRECISION *base_subgrid, PRECISION *reduced_subgrid, int subgridX, int subgridY, int d);

void d2q9_read_horizontal_slices_caller(Grid grid, PRECISION **subgrid_d, PRECISION *interface_left, PRECISION *interface_right, int subgridX, int subgridY);
void d2q9_write_horizontal_slices_caller(Grid grid, PRECISION **subgrid_d, PRECISION *interface_left, PRECISION *interface_right, int subgridX, int subgridY);
void d2q9_read_vertical_slices_caller(Grid grid, PRECISION **subgrid_d, PRECISION *interface_down, PRECISION *interface_up, int subgridX, int subgridY);
void d2q9_LBM_step_caller(Grid grid,
                PRECISION **subgrid_FROM_D,
                PRECISION **subgrid_TO_D,
                int horizontal_uncomputed_number, int vertical_uncomputed_number,
                bool has_from_interface_horizontal,
                bool has_from_interface_vertical,
                bool has_to_interface_horizontal,
                bool has_to_interface_vertical,
                PRECISION *interface_down, PRECISION *interface_up,
                int subgridX, int subgridY,
                int kernel_version
                );



void addKernelType(char name[]);
void recordStart(char name[]);
void recordEnd(char name[]);
void printSummary();
float getAverageTime(char name[]);
float getTotalTime(char name[]);


#endif // LBM_COMMON_H_INCLUDED