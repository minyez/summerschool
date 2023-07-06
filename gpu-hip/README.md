# GPU programming IV-VI: exercises

## HIP Error Checking

- [X] [Wrapping HIP calls into an error checking macro](error-checking)

## Basics of HIP programming

- [X] [Hello world](hello-world)
- [X] [Kernel: saxpy](kernel-saxpy)
- [X] [Kernel: copy2d](kernel-copy2d)

## Streams, events, and synchronization

- [X] [Understanding asynchronicity using events](events)
- [X] [Investigating streams and events](streams)
- [ ] [Asynchronous operations (OpenMP)](../gpu-openmp/async-operations)

## Memory management

- [X] [Memory management strategies](memory-prefetch)
- [X] [Unified memory and structs](memory-struct)
- [X] [Memory performance strategies](memory-performance)
- [ ] [EXTRA: advanced c++ memory wrapping strategies](memory-advanced)

## Fortran and HIP

- [ ] [Hipfort: saxpy](hipfort)

## Multi-GPU programming

- [X] [Vector sum on two GPUs without MPI](vector-sum)
- [X] [Ping-pong with multiple GPUs and MPI](ping-pong)
- [X] [Peer to peer device access (HIP and OpenMP)](p2pcopy)
- [ ] [Bonus: Heat equation with HIP](heat-equation)
