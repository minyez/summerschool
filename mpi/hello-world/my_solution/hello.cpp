#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rc;
    int size, rank;

    rc = MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // TODO: say hello! in parallel
    // std::cout << "Hello! from rank " << rank << std::endl;
    std::cout << "Hello! from rank " << rank << "\n";

    MPI_Finalize();
}
