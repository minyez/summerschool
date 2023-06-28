#include <cstdio>
#include <vector>
#include <mpi.h>
#include <ctime>

int main(int argc, char *argv[])
{
    int myid, ntasks, nrecv;
    constexpr int arraysize = 100000;
    constexpr int msgsize = 100;
    // constexpr int msgsize = 100000;
    std::vector<int> message(arraysize);
    std::vector<int> receiveBuffer(arraysize);
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // Initialize message and receive buffer
    for (int i = 0; i < arraysize; i++) {
        message[i] = myid;
        receiveBuffer[i] = -1;
    }

    // TODO: Implement sending and receiving as defined in the assignment
    // Send msgsize elements from the array "message", and receive into
    // "receiveBuffer"
    if (myid == 0) {
		clock_t c_st = clock();
        MPI_Send(message.data(), msgsize, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(receiveBuffer.data(), msgsize, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
		clock_t c_ed = clock();
		MPI_Get_count(&status, MPI_INT, &nrecv);
        double timediff = double(c_ed - c_st) / CLOCKS_PER_SEC;
        printf("Rank %i received %i elements, first %i\n", myid, nrecv, receiveBuffer[0]);
		printf("Time Send/Recv on rank %i: %f\n", myid, timediff);
    } else if (myid == 1) {
		clock_t c_st = clock();
        MPI_Recv(receiveBuffer.data(), msgsize, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Send(message.data(), msgsize, MPI_INT, 0, 0, MPI_COMM_WORLD);
		clock_t c_ed = clock();
        double timediff = double(c_ed - c_st) / CLOCKS_PER_SEC;
		MPI_Get_count(&status, MPI_INT, &nrecv);
        printf("Rank %i received %i elements, first %i\n", myid, nrecv, receiveBuffer[0]);
		printf("Time Send/Recv on rank %i: %f\n", myid, timediff);
    }

    MPI_Finalize();
    return 0;
}
