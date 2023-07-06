a.out: ping-pong.cpp
	CC -xhip $< -o $@

run: a.out
	srun --account=project_465000536 --partition=dev-g --nodes=1 --ntasks-per-node=2 --cpus-per-task=1 --time=00:05:00 --gpus-per-node=2 ./$<
