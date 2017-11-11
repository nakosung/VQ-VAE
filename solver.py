import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from torchvision.utils import save_image
from model import Generator
import nsml

class Solver(object):
    
    def __init__(self, data_loader, config):
        # Data loader새
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
        
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()
        
        # Start with trained model
        if self.trained_model:
            self.load_trained_model()
        
    def build_model(self):
        # model and optimizer
        self.G = Generator(self.image_size, self.z_dim, self.g_conv_dim, k_dim=self.k_dim, code_dim=self.code_dim)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.lr, [self.beta1, self.beta2])
        
        if torch.cuda.is_available():
            self.G.cuda()
    
    def load_trained_model(self):
        self.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.trained_model)))
        print('loaded trained models (step: {})..!'.format(self.trained_model))

    def load(self,filename):
        S = torch.load(filename)        
        self.G.load_state_dict(S['G'])
         
    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)
        
    def update_lr(self, lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = lr
    
    def reset_grad(self):
        self.g_optimizer.zero_grad()
        
    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)
    
    def to_np(self, x):
        return x.data.cpu().numpy()
    
    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)
        
    def detach(self, x):
        return Variable(x.data)
    
    def train(self):
        # Reconst loss
        reconst_loss = nn.L1Loss()
        
        # Data iter
        data_iter = iter(self.data_loader)
        iter_per_epoch = len(self.data_loader)
        
        # Fixed inputs for sampling
        fixed_x = next(data_iter)
        save_image(self.denorm(fixed_x), os.path.join(self.sample_path, 'fixed_x.png'))
        fixed_x = self.to_var(fixed_x)
        
        # Start with trained model
        if self.trained_model:
            start = self.trained_model + 1
        else:
            start = 0
            
        for step in range(start, self.total_step):
            # Schedule learning rate
            alpha = (step - start) / (self.total_step - start)
            lr = self.lr * (1/100 ** alpha)

            self.update_lr(lr)
            
            # Reset data_iter for each epoch
            if (step+1) % iter_per_epoch == 0:
                data_iter = iter(self.data_loader)
               
            x = self.to_var(next(data_iter))
                        
            # ================== Train G ================== #
            # Train with real images (VQ-VAE)
            out, loss_e1, loss_e2 = self.G(x)
            loss_rec = reconst_loss(out, x)
            
            loss = loss_rec + loss_e1 + self.vq_beta * loss_e2
            self.reset_grad()

            # For decoder
            loss.backward(retain_graph=True)

            # For encoder
            self.G.bwd()

            self.g_optimizer.step()
            
            # Print out log info
            if (step+1) % self.log_step == 0:
                print("[{}/{}] loss: {:.4f}, loss_e1: {:.4f}". \
                      format(step+1, self.total_step, loss_rec.data[0], loss_e1.data[0]))
                      
                if self.use_tensorboard:
                    info = {
                        'loss/loss_rec': loss_rec.data[0],
                        'loss/loss_ee': loss_e1.data[0],
                        'misc/lr': lr
                    }
                    
                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, step+1, scope=locals())
                
            # Sample images
            if (step+1) % self.sample_step == 0:
                reconst, _, _ = self.G(fixed_x)

                def np(tensor):
                    return tensor.cpu().numpy()

                self.logger.images_summary('recons',np(self.denorm(reconst.data)),step+1)

            # Save check points
            if (step+1) % self.model_save_step == 0:
                nsml.save(step)
    
    def save(self, filename):
        G = self.G.state_dict()
        torch.save({'G':G}, filename)

    def infer(self, sample_size):
        # Data iter
        data_iter = iter(self.data_loader)
        
        # Inputs for sampling
        z = self.to_var(torch.randn(sample_size, self.z_dim)) 
        
        # Load trained params
        self.G.eval()  
        
        # Sampling
        fake = self.G.decode(z)

        return self.denorm(fake.data)
        
    def sample(self):
        # Data iter
        data_iter = iter(self.data_loader)
        
        # Inputs for sampling
        x = next(data_iter)
        z = self.to_var(torch.randn(self.sample_size, self.z_dim)) 
        
        save_image(self.denorm(x), 'real.png')
        x = self.to_var(x)
        
        # Load trained params
        self.G.eval()  
        S = torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.step_for_sampling)))
        self.G.load_state_dict(S['G'])
        
        # Sampling
        reconst, _, _ = self.G(x)
        fake = self.G.decode(z)
        save_image(self.denorm(reconst.data), 'reconst.png')