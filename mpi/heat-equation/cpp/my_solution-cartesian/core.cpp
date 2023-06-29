// Main solver routines for heat equation solver

#include <mpi.h>

#include "heat.hpp"

//! Sendrecv version
static void exchange_sendrecv(Field& field, const ParallelData parallel)
{
    double* sbuf;
    double* rbuf;
    MPI_Status status;

    // Send to up, receive from down
    sbuf = &field.temperature(1, 0);
    rbuf = &field.temperature(field.nx + 1, 0);
    MPI_Sendrecv(sbuf, field.ny + 2, MPI_DOUBLE, parallel.nup, 0, rbuf, field.ny + 2, MPI_DOUBLE, parallel.ndown, 0, parallel.comm, &status);

    // Send to down, receive from up
    sbuf = &field.temperature(field.nx, 0);
    rbuf = &field.temperature(0, 0);
    MPI_Sendrecv(sbuf, field.ny + 2, MPI_DOUBLE, parallel.ndown, 0, rbuf, field.ny + 2, MPI_DOUBLE, parallel.nup, 0, parallel.comm, &status);
}

// Exchange the boundary values
void exchange(Field& field, const ParallelData parallel)
{
    // You can utilize the data() method of the Matrix class to obtain pointer
    // to element, e.g. field.temperature.data(i, j)
    exchange_sendrecv(field, parallel);
}

// Update the temperature values using five-point stencil */
void evolve(Field& curr, const Field& prev, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  for (int i = 1; i < curr.nx + 1; i++) {
    for (int j = 1; j < curr.ny + 1; j++) {
            curr(i, j) = prev(i, j) + a * dt * (
	       ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	       ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
               );
    }
  }
}
