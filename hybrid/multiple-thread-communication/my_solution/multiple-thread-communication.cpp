#include <cstdio>
#include <omp.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
    int size, myid, tid, msg, provided;
    const int request = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, request, &provided);

    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    msg = -1;

    #pragma omp parallel firstprivate(tid,msg)
    {
        tid = omp_get_thread_num();
        int tidtag = 200 + tid;
        if (myid == 0)
        {
            msg = tid;
        }
        printf("msg %d on %03d:%03d before communication\n", msg, myid, tid);

        if (myid == 0)
            for (int i = 1; i < size; i++)
                MPI_Send(&tid, 1, MPI_INT, i, tidtag, MPI_COMM_WORLD);
        else
            MPI_Recv(&msg, 1, MPI_INT, 0, tidtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("msg %d on %03d:%03d after communication\n", msg, myid, tid);
    }

    MPI_Finalize();

    return 0;
}
