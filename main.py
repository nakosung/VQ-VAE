import argparse
import os
from torch.backends import cudnn
from data_loader import get_loader
from solver import Solver


def str2bool(v):
    return v.lower() in ('true',)


def main(config):
    # Create directories if not exist.
    os.makedirs(config.log_path, exist_ok=True)
    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.sample_path, exist_ok=True)

    if config.mode == 'sample':
        config.batch_size = config.sample_size

    # Data loader
    data_loader = get_loader(config.image_path, config.image_size,
                             config.batch_size, config.num_workers)
    # Solver
    solver = Solver(data_loader, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'sample':
        solver.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--k_dim', type=int, default=256)
    parser.add_argument('--code_dim', type=int, default=16)
    parser.add_argument('--g_conv_dim', type=int, default=64)

    # Training settings
    parser.add_argument('--total_step', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--trained_model', type=int, default=None)

    parser.add_argument('--vq_beta', type=float, default=0.25)

    # Test settings
    parser.add_argument('--step_for_sampling', type=int, default=200000)
    parser.add_argument('--sample_size', type=int, default=32)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    parser.add_argument('--log_path', type=str, default='./output/logs')
    parser.add_argument('--model_save_path', type=str, default='./output/models')
    parser.add_argument('--sample_path', type=str, default='./output/samples')
    parser.add_argument('--image_path', type=str, default='./data')

    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=200)
    parser.add_argument('--model_save_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
