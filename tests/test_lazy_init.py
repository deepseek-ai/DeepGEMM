import torch
import torch.multiprocessing as mp
import deep_gemm


def main(local_rank: int):
    torch.cuda.set_device(local_rank)


if __name__ == '__main__':
    procs = [mp.Process(target=main, args=(i, ), ) for i in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
