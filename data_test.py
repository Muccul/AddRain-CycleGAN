import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

import visdom

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)


    viz = visdom.Visdom()
    x = next(iter(dataset))
    viz.images(x['A'], nrow=4, win='de', opts=dict(title='de'))
    viz.images(x['B'], nrow=4, win='rain', opts=dict(title='rain'))

    print(x.keys())
    print(x['A'].shape)
    print(x['B'])