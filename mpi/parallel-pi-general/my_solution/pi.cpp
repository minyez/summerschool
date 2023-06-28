#include <iostream>
#include <vector>
#include <mpi.h>
#include <numeric>
#include <cmath>

double integrand(const double &step, const int &igrid)
{
    double x = (igrid - 0.5) * step;
    return 1.0 / (1.0 + x*x);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    constexpr int ngrids_total = 100000;
    constexpr double step = 1.0 / ngrids_total;
    int nprocs, myid, ngrids_local, ngrids_left;

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // determine the number of grids each process should have
    ngrids_local = ngrids_total / nprocs;
    ngrids_left = ngrids_total - ngrids_local * nprocs;
    if (myid < ngrids_left)
        ngrids_local++;

    std::vector<int> ngrids_on_proc(nprocs);

    MPI_Allgather(&ngrids_local, 1, MPI_INT, ngrids_on_proc.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int igrid_st = std::accumulate(ngrids_on_proc.cbegin(), ngrids_on_proc.cbegin() + myid, 0);
    int igrid_ed = std::accumulate(ngrids_on_proc.cbegin(), ngrids_on_proc.cbegin() + myid + 1, 0);
    if (igrid_ed - igrid_st != ngrids_local)
        printf("Inconsistent size with starting and ending grid indices!!!\n");
    
    double integral_local = 0.0;
    for (int i = 0; i != ngrids_local; i++)
        integral_local += integrand(step, i + igrid_st);

    double pi_inte;
    MPI_Reduce(&integral_local, &pi_inte, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    pi_inte *= 4.0 / double(ngrids_total);

    if (myid == 0)
    {
        printf("Integration with %10d grids\n", ngrids_total);
        printf("Integrated Pi: %.20f\n", pi_inte);
        printf("  Exact value: %.20f\n", M_PI);
    }

    MPI_Finalize();
}
