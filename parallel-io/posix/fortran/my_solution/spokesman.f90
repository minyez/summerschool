program pario
  use mpi_f08
  use, intrinsic :: iso_fortran_env, only : error_unit, output_unit
  implicit none

  integer, parameter :: datasize = 64, writer_id = 0
  integer :: rc, my_id, ntasks, localsize, i
  integer, dimension(:), allocatable :: localvector
  integer, dimension(datasize) :: fullvector

  call mpi_init(rc)
  call mpi_comm_size(mpi_comm_world, ntasks, rc)
  call mpi_comm_rank(mpi_comm_world, my_id, rc)

  if (ntasks > 64) then
     write(error_unit, *) 'Maximum number of tasks is 64!'
     call mpi_abort(MPI_COMM_WORLD, -1, rc)
  end if

  if (mod(datasize, ntasks) /= 0) then
     write(error_unit,*) 'Datasize (64) should be divisible by number of tasks'
     call mpi_abort(MPI_COMM_WORLD, -1, rc)
  end if

  localsize = datasize / ntasks
  allocate(localvector(localsize))

  localvector = [(i + my_id * localsize, i=1,localsize)]

  call single_writer()

  deallocate(localvector)
  call mpi_finalize(rc)

contains

  subroutine single_writer()
    implicit none

    character(len=10), parameter :: outfile = "data.txt"

    call MPI_GATHER(localvector, localsize, MPI_INTEGER, fullvector, localsize, MPI_INTEGER, 0, MPI_COMM_WORLD, rc)
    if (my_id == 0) then
      open(writer_id, file=outfile)
      do i = 0, 7
        write(writer_id, "(8I4)") fullvector(i*8+1:i*8+8)
      end do
      close(writer_id)
    end if

  end subroutine single_writer

end program pario
