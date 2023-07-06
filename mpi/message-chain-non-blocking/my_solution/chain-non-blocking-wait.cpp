#include <cstdio>
#include <vector>
#include <mpi.h>

void print_ordered(double t);

int main(int argc, char *argv[])
{
    int i, myid, ntasks;
    constexpr int size = 10000000;
    std::vector<int> message(size);
    std::vector<int> receiveBuffer(size);
    MPI_Status status;
    MPI_Request req1, req2;
    int flag;

    double t0, t1;

    int source, destination;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // Initialize buffers
    for (i = 0; i < size; i++) {
        message[i] = myid;
        receiveBuffer[i] = -1;
    }

    // TODO: Set source and destination ranks
	source = myid > 0? myid - 1: MPI_PROC_NULL;
	destination = myid < ntasks - 1? myid + 1: MPI_PROC_NULL;
    // TODO: Treat boundaries with MPI_PROC_NULL

    // Start measuring the time spent in communication
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();

    // TODO: Send messages
    MPI_Isend(message.data(), size, MPI_INT, destination, 1, MPI_COMM_WORLD, &req1);

    // TODO: Receive messages
    MPI_Irecv(receiveBuffer.data(), size, MPI_INT, source, 1, MPI_COMM_WORLD, &req2);

    // MPI_Test(&req, &flag, &status);
    // printf("Operation finished? %d\n", flag);

    MPI_Wait(&req2, &status);

    // in this case, modifying message is fine without waiting Isend request,
    // as Isend is only performed once and always before Irecv
    // otherwise message among different isend calls can be different from what one expects
    // message[0] = -myid;

    printf("Sender: %d. Sent elements: %d. Tag: %d. Receiver: %d\n",
           myid, size, myid + 1, destination);
    printf("Receiver: %d. first element %d.\n",
           myid, receiveBuffer[0]);

    // Finalize measuring the time and print it out
    t1 = MPI_Wtime();
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    print_ordered(t1 - t0);

    MPI_Finalize();
    return 0;
}

void print_ordered(double t)
{
    int i, rank, ntasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    if (rank == 0) {
        printf("Time elapsed in rank %2d: %6.3f\n", rank, t);
        for (i = 1; i < ntasks; i++) {
            MPI_Recv(&t, 1, MPI_DOUBLE, i, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Time elapsed in rank %2d: %6.3f\n", i, t);
        }
    } else {
        MPI_Send(&t, 1, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD);
    }
}
