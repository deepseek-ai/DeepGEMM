import argparse
import torch
import torch.multiprocessing as mp
import deep_gemm


def main(local_rank: int):
    torch.cuda.set_device(local_rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test lazy initialization')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    args = parser.parse_args()

    procs = [mp.Process(target=main, args=(i, ), ) for i in range(args.num_processes)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
