program datatype_struct
  use mpi_f08
  use iso_fortran_env, only : REAL64
  implicit none

  integer, parameter :: n = 1000, reps=10000

  type particle
     real :: coords(3)
     integer :: charge
     character(len=2) :: label
  end type particle

  type(particle) :: particles(n)

  integer :: i, ierror, myid

  type(mpi_datatype) :: particle_mpi_type, temp_type
  type(mpi_datatype) :: types(3)
  integer :: blocklen(3)
  integer(kind=MPI_ADDRESS_KIND) :: disp(3)
  integer(kind=MPI_ADDRESS_KIND) :: lb, extent

  real(REAL64) :: t1, t2

  call mpi_init(ierror)
  call mpi_comm_rank(MPI_COMM_WORLD, myid, ierror)

  ! Fill in some values for the particles
  if(myid == 0) then
    do i = 1, n
      call random_number(particles(i)%coords)
      particles(i)%charge = 54
      particles(i)%label = 'Xe'
    end do
  end if

  ! Define datatype for the struct
  blocklen = (/3, 1, 2/)
  types = (/MPI_REAL, MPI_INTEGER, MPI_CHARACTER/)
  call MPI_GET_ADDRESS(particles(1)%coords, disp(1), ierror)
  call MPI_GET_ADDRESS(particles(1)%charge, disp(2), ierror)
  call MPI_GET_ADDRESS(particles(1)%label, disp(3), ierror)
  disp(:) = disp(:) - disp(1)
  call MPI_TYPE_CREATE_STRUCT(3, blocklen, disp, types, particle_mpi_type, ierror)
  call MPI_TYPE_COMMIT(particle_mpi_type, ierror)

  ! Check extent
  call MPI_GET_ADDRESS(particles(1), disp(1), ierror)
  call MPI_GET_ADDRESS(particles(2), disp(2), ierror)
  call MPI_TYPE_GET_EXTENT(particle_mpi_type, lb, extent, ierror)
  if (disp(2) - disp(1) .ne. extent) then
    if (myid == 0) print*, "Resizing particle_mpi_type"
    temp_type = particle_mpi_type
    lb = 0
    extent = disp(2) - disp(1)
    call MPI_TYPE_CREATE_RESIZED(temp_type, lb, extent, particle_mpi_type, ierror)
    call MPI_TYPE_FREE(temp_type, ierror)
  end if

  ! Communicate using the created particletype
  ! Multiple sends are done for better timing
  t1 = mpi_wtime()
  if(myid == 0) then
     do i = 1, reps
       call MPI_SEND(particles, n, particle_mpi_type, 1, 0, MPI_COMM_WORLD, ierror)
     end do
  else if(myid == 1) then
     do i = 1, reps
       call MPI_RECV(particles, n, particle_mpi_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
     end do
  end if
  t2=mpi_wtime()

  write(*,*) "Time: ", myid, (t2-t1) / reps
  write(*,*) "Check:", myid, particles(n)%label, particles(n)%coords(1), &
                       particles(n)%coords(2), particles(n)%coords(3)

  ! Free datatype
  call MPI_TYPE_FREE(particle_mpi_type, ierror)

  call mpi_finalize(ierror)

end program datatype_struct
