// Main solver routines for heat equation solver

#include <vector>
#include <mpi.h>

#include "heat.hpp"

//! Sendrecv version
static void exchange_nonblocking(Field& field, ParallelData& parallel)
{
    double* sbuf_up;
    double* sbuf_dn;
    double* rbuf_up;
    double* rbuf_dn;

    // Send to up, receive from down
    sbuf_up = field.temperature.data(1, 0);
    MPI_Isend(sbuf_up, field.ny + 2, MPI_DOUBLE, parallel.nup, 0, MPI_COMM_WORLD, parallel.requests);
    rbuf_dn = field.temperature.data(field.nx + 1, 0);
    MPI_Irecv(rbuf_dn, field.ny + 2, MPI_DOUBLE, parallel.ndown, 0, MPI_COMM_WORLD, parallel.requests + 1);

    // Send to down, receive from up
    sbuf_dn = field.temperature.data(field.nx, 0);
    MPI_Isend(sbuf_dn, field.ny + 2, MPI_DOUBLE, parallel.ndown, 0, MPI_COMM_WORLD, parallel.requests + 2);
    rbuf_up = field.temperature.data(0, 0);
    MPI_Irecv(rbuf_up, field.ny + 2, MPI_DOUBLE, parallel.nup, 0, MPI_COMM_WORLD, parallel.requests + 3);
}

// Exchange the boundary values
void exchange(Field& field, ParallelData& parallel)
{
    // You can utilize the data() method of the Matrix class to obtain pointer
    // to element, e.g. field.temperature.data(i, j)
    
    // exchange_send_recv(field, parallel);
    // exchange_sendrecv(field, parallel);
    exchange_nonblocking(field, parallel);
}

// Update the temperature values using five-point stencil */
void evolve(Field& curr, Field& prev, ParallelData& parallel, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  // handle the inner part
  for (int i = 2; i < curr.nx; i++) {
    for (int j = 1; j < curr.ny + 1; j++) {
            curr(i, j) = prev(i, j) + a * dt * (
	       ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	       ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
               );
    }
  }

  // wait for the boundary data to arrive
  MPI_Waitall(4, parallel.requests, MPI_STATUS_IGNORE);

  std::vector<int> xbound = {1, curr.nx};

  for (const auto& i: xbound)
  {
    for (int j = 1; j < curr.ny + 1; j++)
    {
         curr(i, j) = prev(i, j) + a * dt * (
	         ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	         ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
            );
    }
  }
}
