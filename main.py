import argparse
import base64
from io import BytesIO
import os
import numpy as np
from PIL import Image
import torch
from torch.backends import cudnn
from data_loader import get_loader
from solver import Solver
import nsml

INFER_STR = ['inference', 'infer']
OUTPUT_FORMAT = 'png'


def str2bool(v):
    return v.lower() in ('true')


def main(config, scope):

    # Create directories if not exist.
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)

    if config.mode == 'sample':
        config.batch_size = config.sample_size

    # Data loader
    data_loader = get_loader(config.image_path, config.image_size,
                              config.batch_size, config.num_workers)
    # Solver
    solver = Solver(data_loader, config)

    def load(filename, *args):
        solver.load(filename)

    def save(filename, *args):
        solver.save(filename)

    def infer(input):
        result = solver.infer(input)
        # convert tensor to dataurl
        data_url_list = [''] * input
        for idx, sample in enumerate(result):
            numpy_array = np.uint8(sample.cpu().numpy()*255)
            image = Image.fromarray(np.transpose(numpy_array, axes=(1, 2, 0)), 'RGB')
            temp_out = BytesIO()
            image.save(temp_out, format=OUTPUT_FORMAT)
            byte_data = temp_out.getvalue()
            data_url_list[idx] = u'data:image/{format};base64,{data}'.\
                format(format=OUTPUT_FORMAT,
                       data=base64.b64encode(byte_data).decode('ascii'))
        return data_url_list

    def evaluate(test_data, output):
        pass

    def decode(input):
        return input

    nsml.bind(save, load, infer, evaluate, decode)

    if config.pause:
        nsml.paused(scope=scope)
    
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
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    
    # Training settings
    parser.add_argument('--total_step', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=32)
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
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample', *INFER_STR])
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    
    parser.add_argument('--log_path', type=str, default='./celebA/logs')
    parser.add_argument('--model_save_path', type=str, default='./celebA/models')
    parser.add_argument('--sample_path', type=str, default='./celebA/samples')
    parser.add_argument('--image_path', type=str, default=nsml.DATASET_PATH)
    
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=200)
    parser.add_argument('--model_save_step', type=int, default=1000)

    # nsml setting
    parser.add_argument('--pause', type=int, default=0)
    
    config = parser.parse_args()
    print(config)
    main(config,scope=locals())
