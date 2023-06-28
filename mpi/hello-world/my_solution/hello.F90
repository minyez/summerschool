program hello
  use mpi
  implicit none

  integer :: info
  integer :: nprocs, myrank

  call MPI_INIT(info)

  call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, info)
  call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, info)

  ! TODO: say hello! in parallel
  write(*,*) 'Hello! from rank ', myrank

  call MPI_FINALIZE(info)

end program hello
