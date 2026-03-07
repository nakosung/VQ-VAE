from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def images_summary(self, tag, images, step):
        """Log a batch of images. images: (N, C, H, W) tensor."""
        self.writer.add_images(tag, images, step)

    def close(self):
        self.writer.close()
