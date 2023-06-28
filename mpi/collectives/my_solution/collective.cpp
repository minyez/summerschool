#include <cstdio>
#include <vector>
#include <mpi.h>

#define NTASKS 4

void init_buffers(std::vector<int> &sendbuffer, std::vector<int> &recvbuffer);
void print_buffers(std::vector<int> &buffer);


int main(int argc, char *argv[])
{
    int ntasks, rank;
    std::vector<int> sendbuf(2 * NTASKS), recvbuf(2 * NTASKS);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (ntasks != NTASKS) {
        if (rank == 0) {
            fprintf(stderr, "Run this program with %i tasks.\n", NTASKS);
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* Initialize message buffers */
    init_buffers(sendbuf, recvbuf);

    /* Print data that will be sent */
    print_buffers(sendbuf);

    /* TODO: use a single collective communication call
     *       (and maybe prepare some parameters for the call)
     */
    // case 1
    {
        if (rank == 0) printf("Case 1\n");
        if (rank == 0) recvbuf = sendbuf;
        MPI_Bcast(recvbuf.data(), 2 * NTASKS, MPI_INT, 0, MPI_COMM_WORLD);
        print_buffers(recvbuf);
        fflush(stdout);
    }

    // case 2
    {
        if (rank == 0) printf("Case 2\n");
        init_buffers(sendbuf, recvbuf); // reset the recvbuf
        MPI_Scatter(sendbuf.data(), 2, MPI_INT, recvbuf.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
        print_buffers(recvbuf);
        fflush(stdout);
    }

    // case 3
    {
        if (rank == 0) printf("Case 3\n");
        init_buffers(sendbuf, recvbuf); // reset the recvbuf
        constexpr int sendcounts[NTASKS] = {1, 1, 2, 4};
        int displs[NTASKS] = {0};
        for (int i = 1; i != NTASKS; i++)
            displs[i] = displs[i-1] + sendcounts[i-1];
        MPI_Gatherv(sendbuf.data(), sendcounts[rank], MPI_INT, recvbuf.data(), sendcounts, displs, MPI_INT, 1, MPI_COMM_WORLD);
        print_buffers(recvbuf);
        fflush(stdout);
    }

    // case 4
    {
        if (rank == 0) printf("Case 3\n");
        init_buffers(sendbuf, recvbuf); // reset the recvbuf
        for (int i = 0; i != NTASKS; i++)
            MPI_Scatter(sendbuf.data(), 2, MPI_INT, recvbuf.data() + 2 * i, 2, MPI_INT, i, MPI_COMM_WORLD);
        print_buffers(recvbuf);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}


void init_buffers(std::vector<int> &sendbuffer, std::vector<int> &recvbuffer)
{
    int rank;
    int buffersize = sendbuffer.size();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < buffersize; i++) {
        recvbuffer[i] = -1;
        sendbuffer[i] = i + buffersize * rank;
    }
}


void print_buffers(std::vector<int> &buffer)
{
    int rank, ntasks;
    int buffersize = buffer.size();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    std::vector<int> printbuffer(buffersize * ntasks);

    MPI_Gather(buffer.data(), buffersize, MPI_INT,
               printbuffer.data(), buffersize, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int j = 0; j < ntasks; j++) {
            printf("Task %2i:", j);
            for (int i = 0; i < buffersize; i++) {
                printf(" %2i", printbuffer[i + buffersize * j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
