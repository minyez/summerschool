program datatype1
  use mpi_f08
  implicit none

  integer :: i, j
  integer :: array(8,6)
  integer :: rank, ierr, dst, src
  type(mpi_datatype) :: vector

  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)

  ! Initialize arrays
  array = 0
  if (rank == 0) then
     array = reshape([ ((i*10 + j, i=1,8), j=1,6) ], [8, 6] )
  end if

  ! Print data on rank 0
  if (rank == 0) then
     write(*,*) 'Data on rank', rank
     do i=1,8
        write(*,'(*(I3))') array(i, :)
     end do
  end if

  ! Create datatype
  call MPI_TYPE_VECTOR(6, 1, 8, MPI_INTEGER, vector, ierr)
  call MPI_TYPE_COMMIT(vector, ierr)

  ! Send data from rank 0 to rank 1
  src = MPI_PROC_NULL
  dst = MPI_PROC_NULL
  if (rank .eq. 0) then
    dst = 1
  endif

  if (rank .eq. 1) then
    src = 0
  endif

  ! Send data from rank 0 to rank 1
  call MPI_SENDRECV(array, 2, vector, dst, 0, array, 2, vector, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)

  ! Free datatype
  call MPI_TYPE_FREE(vector, ierr)

  ! Print received data
  if (rank == 1) then
     write(*,*) 'Received data on rank', rank
     do i=1,8
        write(*,'(*(I3))') array(i, :)
     end do
  end if

  call mpi_finalize(ierr)

end program datatype1
