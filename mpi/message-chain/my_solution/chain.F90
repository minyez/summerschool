program basic
  use mpi_f08
  use iso_fortran_env, only : REAL64

  implicit none
  integer, parameter :: size = 10000000
  integer :: rc, myid, ntasks
  integer :: message(size)
  integer :: receiveBuffer(size)
  type(mpi_status) :: status

  real(REAL64) :: t0, t1

  integer :: source, destination

  call mpi_init(rc)
  call mpi_comm_rank(MPI_COMM_WORLD, myid, rc)
  call mpi_comm_size(MPI_COMM_WORLD, ntasks, rc)

  message = myid
  receiveBuffer = -1

  ! TODO: Set source and destination ranks
  source = myid - 1
  destination = myid + 1
  ! TODO: Treat boundaries with MPI_PROC_NULL
  if (myid .eq. 0) source = MPI_PROC_NULL
  if (myid .eq. ntasks - 1) destination = MPI_PROC_NULL

  ! Start measuring the time spent in communication
  call mpi_barrier(mpi_comm_world, rc)
  t0 = mpi_wtime()

  ! TODO: Send messages
  call MPI_Send(message, size, MPI_INT, destination, 1, MPI_COMM_WORLD, rc);

  write(6,'(A10,I3,A20,I8,A,I3,A,I3)') 'Sender: ', myid, &
          ' Sent elements: ', size, &
          '. Tag: ', myid+1, '. Receiver: ', destination

  ! TODO: Receive messages
  call MPI_Recv(receiveBuffer, size, MPI_INT, source, 1, MPI_COMM_WORLD, status, rc);

  write(6,'(A10,I3,A,I3)') 'Receiver: ', myid, &
          ' First element: ', receiveBuffer(1)

  ! Finalize measuring the time and print it out
  t1 = mpi_wtime()
  call flush(6)
  call mpi_barrier(mpi_comm_world, rc)

  call print_ordered(t1 - t0)

  call mpi_finalize(rc)

contains

  subroutine print_ordered(t)
    implicit none
    real(REAL64) :: t

    integer i

    if (myid == 0) then
       write(6, '(A20, I3, A, F6.3)') 'Time elapsed in rank', myid, ':', t
       do i=1, ntasks-1
           call mpi_recv(t, 1, MPI_DOUBLE_PRECISION, i, 11,  &
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE, rc)
           write(6, '(A20, I3, A, F6.3)') 'Time elapsed in rank', i, ':', t
       end do
    else
       call mpi_send(t, 1, MPI_DOUBLE_PRECISION, 0, 11,  &
                         MPI_COMM_WORLD, rc)
    end if
  end subroutine print_ordered

end program basic
