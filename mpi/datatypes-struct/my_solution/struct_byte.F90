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

  integer(kind=MPI_ADDRESS_KIND) :: disp(2)
  integer :: total_bytes

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

  ! Check how large one particle type is
  call MPI_GET_ADDRESS(particles(1), disp(1), ierror)
  call MPI_GET_ADDRESS(particles(2), disp(2), ierror)
  total_bytes = (disp(2) - disp(1)) * n

  ! Communicate using the created particletype
  ! Multiple sends are done for better timing
  t1 = mpi_wtime()
  if(myid == 0) then
     do i = 1, reps
       call MPI_SEND(particles, total_bytes, MPI_BYTE, 1, 0, MPI_COMM_WORLD, ierror)
     end do
  else if(myid == 1) then
     do i = 1, reps
       call MPI_RECV(particles, total_bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
     end do
  end if
  t2=mpi_wtime()

  write(*,*) "Time: ", myid, (t2-t1) / reps
  write(*,*) "Check:", myid, particles(n)%label, particles(n)%coords(1), &
                       particles(n)%coords(2), particles(n)%coords(3)

  call mpi_finalize(ierror)

end program datatype_struct
