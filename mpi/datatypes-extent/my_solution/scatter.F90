program datatype1
  use mpi_f08
  implicit none

  integer :: i, j
  integer :: array(8,6), array_recv(8,6)
  integer :: nprocs, rank, ierr, id
  integer(kind=mpi_address_kind) :: lb, extent
  type(mpi_datatype) :: vector
  type(mpi_datatype) :: vector_resized

  call mpi_init(ierr)
  call mpi_comm_size(MPI_COMM_WORLD, nprocs, ierr)
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

  if (nprocs > 6) then
    write(*,*) "Procs > 6, probably will lead to wrong results"
  end if

  ! Create datatype
  call MPI_TYPE_VECTOR(6, 1, 8, MPI_INTEGER, vector, ierr)
  call MPI_TYPE_GET_EXTENT(vector, lb, extent, ierr)
  lb = 0
  ! extent = sizeof(array(1, 1)) ! GNU extension
  extent = storage_size(array(1, 1)) / 8 ! Fortran 2008
  call MPI_TYPE_CREATE_RESIZED(vector, lb, extent, vector_resized, ierr)
  call MPI_TYPE_COMMIT(vector_resized, ierr)

  call MPI_SCATTER(array, 1, vector_resized, array_recv(rank+1, 1), 1, vector_resized, 0, MPI_COMM_WORLD, ierr)

  ! Free datatype
  call MPI_TYPE_FREE(vector, ierr)

  ! Print received data
  do id = 0, nprocs - 1
    if (rank == id) then
       write(*,*) 'Received data on rank', rank
       do i=1,8
          write(*,'(*(I3))') array_recv(i, :)
       end do
    end if
  end do

  call mpi_finalize(ierr)

end program datatype1
