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

void d2q9_LBM_step_caller(Grid grid,
                double **subgrid_FROM_D,
                double **subgrid_TO_D,
                int horizontal_uncomputed_number, int vertical_uncomputed_number,
                bool has_from_interface_horizontal,
                bool has_from_interface_vertical,
                bool has_to_interface_horizontal,
                bool has_to_interface_vertical,
                double *interface_left, double *interface_right,
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

    printf("d2q9_LBM_step_caller: has_from_interface_horizontal=%d, has_from_interface_vertical=%d, has_to_interface_horizontal=%d, has_to_interface_vertical=%d\n",
        has_from_interface_horizontal, has_from_interface_vertical, has_to_interface_horizontal, has_to_interface_vertical);

    d2q9_LBM_step<<<256, 256>>>(grid,
                subgrid_FROM_D_wrapped,
                subgrid_TO_D_wrapped,
                horizontal_uncomputed_number, vertical_uncomputed_number,
                has_from_interface_horizontal,
                has_from_interface_vertical,
                has_to_interface_horizontal,
                has_to_interface_vertical,
                interface_left, interface_right,
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
    u = 0.03;
    v = 0.00001; // to get instability

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



// void d2q9_step(d2q9 *lbm) {
//   d2q9_shift(lbm);
//   d2q9_relax(lbm);
//   d2q9_boundary(lbm);
// }

// void d2q9_relax(d2q9 *lbm) {

// #pragma omp for schedule(static) nowait
//   for (size_t i = 0; i < lbm->nx; i++) {
//     for (size_t j = 0; j < lbm->ny; j++) {
//       double f[9];
//       double feq[9];
//       for (size_t k = 0; k < 9; k++) {
//         f[k] = lbm->fnext[k][i * lbm->ny + j];
//       }
//       double w[3];
//       kin_to_fluid(f, w, lbm);
//       for (size_t k = 0; k < 3; k++) {
//         lbm->w[k][i * lbm->ny + j] = w[k];
//         // printf("u=%f\n",w[1]/w[0]);
//       }
//       fluid_to_kin(w, feq, lbm);
//       for (size_t k = 0; k < 9; k++) {
//         lbm->f[k][i * lbm->ny + j] =
// 	  _RELAX * feq[k] + (1 - _RELAX) * lbm->fnext[k][i * lbm->ny + j];
//       }
//     }
//   }
// }

// void d2q9_boundary(d2q9 *lbm) {

// #pragma omp for schedule(static) nowait
//   for (size_t i = 0; i < lbm->nx; i++) {
//     for (size_t j = 0; j < lbm->ny; j++) {
//       double x = i * lbm->dx;
//       double y = j * lbm->dx;
//       if (mask(x, y)) {
//         double wb[3];
//         imposed_data(x, y, lbm->tnow, wb);
//         double fb[9];
//         fluid_to_kin(wb, fb, lbm);
//         for (size_t k = 0; k < 9; k++) {
//           lbm->f[k][i * lbm->ny + j] =
// 	    _RELAX * fb[k] + (1 - _RELAX) * lbm->f[k][i * lbm->ny + j];
//         }
//       }
//     }
//   }
// }

// void d2q9_solve(d2q9 *lbm, double tmax) {
//   lbm->tmax = tmax;
//   double local_tnow = lbm->tnow;
//   const double dt = lbm->dx / lbm->smax;
//   const size_t num_iter = (size_t)(ceil((tmax - local_tnow) / dt));
//   const size_t inter_print = num_iter / 10 == 0 ? 1 : num_iter / 10;
// #ifdef _OPENMP
//   double tstart_chunk;
//   tstart_chunk = omp_get_wtime();
// #endif
// #pragma omp parallel default(none) shared(lbm, tmax, tstart_chunk, dt, inter_print)	\
//   firstprivate(local_tnow)
//   {
//     size_t iter_count = 0;

//     while (local_tnow < tmax) {
//       d2q9_step(lbm);
// #pragma omp single nowait
//       lbm->tnow += dt;
//       if (!iter_count) {
// #ifdef _OPENMP
// #pragma omp master
//         {
//           double tend_chunk = omp_get_wtime();
//           printf("t=%f dt=%f tmax=%f (%zu iter in %.3fs)\n", lbm->tnow, dt,
//                  lbm->tmax, inter_print, tend_chunk - tstart_chunk);
//           tstart_chunk = tend_chunk;
//         }
// #else
//         printf("t=%f dt=%f tmax=%f\n", lbm->tnow, dt, lbm->tmax);
// #endif
//       }
// #pragma omp barrier
//       iter_count = iter_count == inter_print ? 0 : iter_count + 1;
//       local_tnow += dt;
//     }
//   }
// }

// void fluid_to_kin(const double *w, double *f, d2q9 *lbm) {
//   static const double c2 = 1. / 3.;
//   double dotvel = 0, vel2 = 0, l2 = 0, l4 = 0, c4 = 0;

//   l2 = lbm->smax * lbm->smax;
//   double l2_ov_c2 = l2 / c2;

//   l4 = l2 * l2;
//   c4 = c2 * c2;

//   vel2 = (w[1] * w[1] + w[2] * w[2]) / (w[0] * w[0]);
//   dotvel = sqrt(l2) * (lbm->vel[0][0] * w[1] + lbm->vel[0][1] * w[2]) / w[0];

//   f[0] = (4. / 9.) * w[0] *
//     (1.0 + (l2_ov_c2)*dotvel + l4 / (2. * c4) * dotvel * dotvel -
//      l2 / (2. * c2) * vel2);

//   for (size_t i = 1; i < 5; i++) {
//     dotvel = sqrt(l2) * (lbm->vel[i][0] * w[1] + lbm->vel[i][1] * w[2]) / w[0];
//     f[i] = (1. / 9.) * w[0] *
//       (1.0 + (l2_ov_c2)*dotvel + l4 / (2. * c4) * dotvel * dotvel -
//        l2 / (2. * c2) * vel2);
//   }
//   for (size_t i = 5; i < 9; i++) {
//     dotvel = sqrt(l2) * (lbm->vel[i][0] * w[1] + lbm->vel[i][1] * w[2]) / w[0];
//     f[i] = (1. / 36.) * w[0] *
//       (1.0 + (l2_ov_c2)*dotvel + l4 / (2. * c4) * dotvel * dotvel -
//        l2 / (2. * c2) * vel2);
//   }
// }

// void kin_to_fluid(const double *restrict f, double *restrict w, d2q9 *lbm) {

//   w[0] = 0;
//   w[1] = 0;
//   w[2] = 0;
//   double c = lbm->smax;

//   for (size_t i = 0; i < 9; i++) {
//     w[0] = w[0] + f[i];
//     w[1] = w[1] + c * lbm->vel[i][0] * f[i];
//     w[2] = w[2] + c * lbm->vel[i][1] * f[i];
//   }
// }



//   lbm->vel[0][0] = 0;
//   lbm->vel[0][1] = 0;

//   lbm->vel[1][0] = 1;
//   lbm->vel[1][1] = 0;

//   lbm->vel[2][0] = 0;
//   lbm->vel[2][1] = 1;

//   lbm->vel[3][0] = -1;
//   lbm->vel[3][1] = 0;

//   lbm->vel[4][0] = 0;
//   lbm->vel[4][1] = -1;

//   lbm->vel[5][0] = 1;
//   lbm->vel[5][1] = 1;

//   lbm->vel[6][0] = -1;
//   lbm->vel[6][1] = 1;

//   lbm->vel[7][0] = -1;
//   lbm->vel[7][1] = -1;

//   lbm->vel[8][0] = 1;
//   lbm->vel[8][1] = -1;



// similar to D2Q9_step
__global__
void d2q9_LBM_step(Grid grid,
                        SubgridArray subgrid_FROM_D,
                        SubgridArray subgrid_TO_D,
                        int horizontal_uncomputed_number, int vertical_uncomputed_number,
                        bool has_from_interface_horizontal,
                        bool has_from_interface_vertical,
                        bool has_to_interface_horizontal,
                        bool has_to_interface_vertical,
                        double *interface_left, double *interface_right,
                        double *interface_down, double *interface_up,
                        int subgridX, int subgridY)
{
    int stride = blockDim.x * gridDim.x;

    int cellNum = grid.subgridTrueSize[0] * grid.subgridTrueSize[1];

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < cellNum; id += stride)
    {
        int subgrid_true_x = id % grid.subgridTrueSize[0];
        int subgrid_true_y = id / grid.subgridTrueSize[0];

        if(subgrid_true_x < horizontal_uncomputed_number || subgrid_true_x >= grid.subgridTrueSize[0] - horizontal_uncomputed_number ||
           subgrid_true_y < vertical_uncomputed_number || subgrid_true_y >= grid.subgridTrueSize[1] - vertical_uncomputed_number)
        {
            continue;
        }

        double f[3][3];

        // shift
        for(int d=0; d<grid.directionsNumber; d++)
        {
            double *target_FROM_subgrid = subgrid_FROM_D.subgrid[d];
            int target_true_x = subgrid_true_x - get_dir(d,0);
            int target_true_y = subgrid_true_y - get_dir(d,1);
            for(int c=0; c<grid.conservativesNumber; c++)
            {
                if(has_from_interface_vertical && target_true_y < grid.overlapSize[1])
                { // Read from the down interface
                    assert(target_true_y == 0);
                    assert(grid.overlapSize[1] == 1);
                    f[d][c] = interface_down[c*grid.subgridTrueSize[0] + target_true_x];
                    //printf("%d %d %d %d reads value %f from down interface\n", subgrid_true_x, subgrid_true_y, c, d, f[d][c]);
                }
                else if(has_from_interface_vertical && target_true_y >= grid.subgridTrueSize[1] - grid.overlapSize[1])
                { // Read from the up interface
                    assert(target_true_y == grid.subgridTrueSize[1] - 1);
                    assert(grid.overlapSize[1] == 1);
                    f[d][c] = interface_up[c*grid.subgridTrueSize[0] + target_true_x];
                    //printf("%d %d %d %d reads value %f from up interface\n", subgrid_true_x, subgrid_true_y, c, d, f[d][c]);
                }
                else if(has_from_interface_horizontal && target_true_x < grid.overlapSize[0])
                { // Read from the left interface
                    f[d][c] = interface_left[
                        d*grid.overlapSize[0]*grid.subgridTrueSize[1]*grid.conservativesNumber +
                        c*grid.overlapSize[0]*grid.subgridTrueSize[1] +
                        target_true_y*grid.overlapSize[0] +
                        target_true_x];
                    //printf("%d %d %d %d reads value %f from left interface\n", subgrid_true_x, subgrid_true_y, c, d, f[d][c]);
                }
                else if(has_from_interface_horizontal && target_true_x >= grid.subgridTrueSize[0] - grid.overlapSize[1])
                { // Read from the right interface
                    f[d][c] = interface_right[
                        d*grid.overlapSize[0]*grid.subgridTrueSize[1]*grid.conservativesNumber +
                        c*grid.overlapSize[0]*grid.subgridTrueSize[1] +
                        target_true_y*grid.overlapSize[0] +
                        target_true_x];
                    //printf("%d %d %d %d reads value %f from right interface\n", subgrid_true_x, subgrid_true_y, c, d, f[d][c]);
                }
                else
                { // Main case: in the logical space
                    f[d][c] = target_FROM_subgrid[c*grid.subgridTrueSize[0]*grid.subgridTrueSize[1] + target_true_y * grid.subgridTrueSize[0] + target_true_x];
                }
            }
        }

        // relax
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

        int position_in_interface_left_x = subgrid_true_x - grid.overlapSize[0];
        int position_in_interface_left_y = subgrid_true_y - grid.overlapSize[1];
        int position_in_interface_right_x = subgrid_true_x - grid.subgridOwnedSize[0];
        int position_in_interface_right_y = subgrid_true_y - grid.overlapSize[1];

        int position_in_interface_down_x = subgrid_true_x - grid.overlapSize[0];
        int position_in_interface_down_y = subgrid_true_y - grid.overlapSize[1];
        int position_in_interface_up_x = subgrid_true_x - grid.overlapSize[0];
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

                if(has_to_interface_vertical && position_in_interface_down_y >= 0 && position_in_interface_down_y < grid.overlapSize[1])
                {
                    assert(grid.overlapSize[1] == 1);
                    interface_down[c*grid.overlapSize[0] + position_in_interface_down_x] = f[d][c];
                }
                if(has_to_interface_vertical && position_in_interface_up_y >= 0 && position_in_interface_up_y < grid.overlapSize[1])
                {
                    assert(grid.overlapSize[1] == 1);
                    interface_up[c*grid.overlapSize[0] + position_in_interface_up_x] = f[d][c];
                }
                if(has_to_interface_horizontal && position_in_interface_left_x >= 0 && position_in_interface_left_x < grid.overlapSize[0])
                {
                    interface_left[d*grid.overlapSize[0]*grid.subgridTrueSize[1]*grid.conservativesNumber +
                            c*grid.overlapSize[0]*grid.subgridTrueSize[1] +
                            position_in_interface_left_y*grid.overlapSize[0] +
                            position_in_interface_left_x]
                        = f[d][c];
                }
                if(has_to_interface_horizontal && position_in_interface_right_x >= 0 && position_in_interface_right_x < grid.overlapSize[0])
                {
                    interface_right[d*grid.overlapSize[0]*grid.subgridTrueSize[1]*grid.conservativesNumber +
                            c*grid.overlapSize[0]*grid.subgridTrueSize[1] +
                            position_in_interface_left_y*grid.overlapSize[0] +
                            position_in_interface_left_x]
                        = f[d][c];
                }
                
                // In all cases, write to the subgrid
                target_TO_subgrid[c*grid.subgridTrueSize[0]*grid.subgridTrueSize[1] + subgrid_true_y * grid.subgridTrueSize[0] + subgrid_true_x] = f[d][c];
            }
        }
    }
}
