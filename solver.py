import torch
import torch.nn as nn
import os
from torchvision.utils import save_image
from model import Generator


class Solver(object):

    def __init__(self, data_loader, config):
        # Data loader
        self.data_loader = data_loader

        # Model hyper-parameters
        self.image_size = config.image_size
        self.z_dim = config.z_dim
        self.k_dim = config.k_dim
        self.g_conv_dim = config.g_conv_dim
        self.code_dim = config.code_dim

        # Training settings
        self.total_step = config.total_step
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.trained_model = config.trained_model
        self.use_tensorboard = config.use_tensorboard

        self.vq_beta = config.vq_beta

        # Path and step size
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.model_save_path = config.model_save_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Test setting
        self.step_for_sampling = config.step_for_sampling
        self.sample_size = config.sample_size

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.trained_model:
            self.load_trained_model()

    def build_model(self):
        self.G = Generator(self.image_size, self.z_dim, self.g_conv_dim,
                           k_dim=self.k_dim, code_dim=self.code_dim).to(self.device)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.lr, [self.beta1, self.beta2])

    def load_trained_model(self):
        self.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.trained_model)))
        print('loaded trained models (step: {})..!'.format(self.trained_model))

    def load(self, filename):
        S = torch.load(filename, map_location=self.device, weights_only=True)
        self.G.load_state_dict(S['G'])

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def train(self):
        reconst_loss = nn.L1Loss()

        data_iter = iter(self.data_loader)
        iter_per_epoch = len(self.data_loader)

        # Fixed inputs for sampling
        fixed_x = next(data_iter)
        save_image(self.denorm(fixed_x), os.path.join(self.sample_path, 'fixed_x.png'))
        fixed_x = fixed_x.to(self.device)

        # Start with trained model
        start = (self.trained_model + 1) if self.trained_model else 0

        for step in range(start, self.total_step):
            # Schedule learning rate
            alpha = (step - start) / (self.total_step - start)
            lr = self.lr * (1 / 100 ** alpha)
            self.update_lr(lr)

            # Reset data_iter for each epoch
            if (step + 1) % iter_per_epoch == 0:
                data_iter = iter(self.data_loader)

            x = next(data_iter).to(self.device)

            # ================== Train G ================== #
            out, loss_e1, loss_e2 = self.G(x)
            loss_rec = reconst_loss(out, x)

            loss = loss_rec + loss_e1 + self.vq_beta * loss_e2
            self.reset_grad()
            loss.backward()
            self.g_optimizer.step()

            # Print out log info
            if (step + 1) % self.log_step == 0:
                print("[{}/{}] loss_rec: {:.4f}, loss_codebook: {:.4f}".format(
                    step + 1, self.total_step, loss_rec.item(), loss_e1.item()))

                if self.use_tensorboard:
                    self.logger.scalar_summary('loss/loss_rec', loss_rec.item(), step + 1)
                    self.logger.scalar_summary('loss/loss_codebook', loss_e1.item(), step + 1)
                    self.logger.scalar_summary('misc/lr', lr, step + 1)

            # Sample images
            if (step + 1) % self.sample_step == 0:
                with torch.no_grad():
                    reconst, _, _ = self.G(fixed_x)
                save_image(self.denorm(reconst),
                           os.path.join(self.sample_path, 'reconst_{}.png'.format(step + 1)))

            # Save checkpoints
            if (step + 1) % self.model_save_step == 0:
                self.save(os.path.join(
                    self.model_save_path, '{}_G.pth'.format(step + 1)))

    def save(self, filename):
        torch.save({'G': self.G.state_dict()}, filename)

    @torch.no_grad()
    def infer(self, sample_size):
        z = torch.randn(sample_size, self.z_dim).to(self.device)
        self.G.eval()
        fake = self.G.decode(z)
        return self.denorm(fake)

    def sample(self):
        data_iter = iter(self.data_loader)

        x = next(data_iter)
        z = torch.randn(self.sample_size, self.z_dim).to(self.device)

        save_image(self.denorm(x), 'real.png')
        x = x.to(self.device)

        self.G.eval()
        S = torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.step_for_sampling)),
            map_location=self.device, weights_only=True)
        self.G.load_state_dict(S['G'])

        with torch.no_grad():
            reconst, _, _ = self.G(x)
            fake = self.G.decode(z)
        save_image(self.denorm(reconst), 'reconst.png')
        save_image(self.denorm(fake), 'fake.png')
