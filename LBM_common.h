#ifndef LBM_COMMON_H_INCLUDED
#define LBM_COMMON_H_INCLUDED

// LBM defines

#define EPSILON 0.000000001

#define CFL 0.45
#define ALPHA 0.9 // TODO change to match cylinder test case
#define BETA 0.9

#define CYLINDER_RADIUS 0.3
#define CYLINDER_CENTER_X 0.0
#define CYLINDER_CENTER_Y 0.0

// HPC defines

#define DIMENSIONS_NUMBER 2


#define MAX_SUBGRIDS 256

#define PROBLEM_SIZE_X (64)
#define PROBLEM_SIZE_Y (64)
#define PROBLEM_SIZE_Z (8)

#define SAVED_DATA_SIZE_X (PROBLEM_SIZE_X)
#define SAVED_DATA_SIZE_Y (PROBLEM_SIZE_Y)

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
	double physicalMinCoords[DIMENSIONS_NUMBER];
	double physicalSize[DIMENSIONS_NUMBER];

    void *desc;
} Grid;



void d2q9_initial_value_d_caller(Grid grid, double *subgrid, int subgridX, int subgridY, int d);
void d2q9_save_reduce_caller(Grid grid, double *base_subgrid, double *reduced_subgrid, int subgridX, int subgridY, int d);
void d2q9_LBM_step_caller(Grid grid, double (*subgrid_FROM_D)[], double (*subgrid_TO_D)[], int subgridX, int subgridY);


#endif // LBM_COMMON_H_INCLUDED