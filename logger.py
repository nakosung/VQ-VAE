# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import numpy as np
import scipy.misc 
import nsml
import visdom

class Logger(object):    
    def __init__(self, log_dir):
        self.last = None
        self.viz = nsml.Visdom(visdom=visdom)

    def scalar_summary(self, tag, value, step, scope=None):
        if self.last and self.last['step'] != step:
            nsml.report(**self.last,scope=scope)
            self.last = None
        if self.last is None:
            self.last = {'step':step,'iter':step,'epoch':1}
        self.last[tag] = value            

    def images_summary(self, tag, images, step):
        """Log a list of images."""
        self.viz.images(
            images,            
            opts=dict(title='%s/%d' % (tag, step), caption='%s/%d' % (tag, step)),
        )            
        
    def histo_summary(self, tag, values, step, bins=1000):
        pass