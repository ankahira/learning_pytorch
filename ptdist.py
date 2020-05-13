import argparse
import torch.distributed as dist

parser = argparse.ArgumentParser(description='Testing distributed')
parser.add_argument('--distributed', action='store_true', help='enables distributed processes')
parser.add_argument('--local_rank', default=0, type=int, help='number of distributed processes')
parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')


def main():
    print("we are here")
    opt = parser.parse_args()
    if opt.distributed:
        dist.init_process_group(backend=opt.dist_backend, init_method='env://')

    print("Initialized Rank:", dist.get_rank())


if __name__ == '__main__':
    main()