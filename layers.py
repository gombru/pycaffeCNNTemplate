import caffe

import numpy as np
from PIL import Image
from PIL import ImageOps

import random


class customDataLayer(caffe.Layer):

    def setup(self, bottom, top):

        # config
        params = eval(self.param_str)
        self.dir = params['dir']
        self.train = params['train']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.batch_size = params['batch_size']
        self.resize_w = params['resize_w']
        self.resize_h = params['resize_h']
        self.crop_w = params['crop_w']
        self.crop_h = params['crop_h']
        self.crop_margin = params['crop_margin']
        self.mirror = params['mirror']
        self.rotate = params['rotate']
        self.HSV_prob = params['HSV_prob']
        self.HSV_jitter = params['HSV_jitter']



        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.labels = [int(i.split(' ', 1)[1]) for i in self.indices]
        self.indices = [i.split(' ', 1)[0] for i in self.indices]

	# ATTENTION: We may want to randomize also in validation and test
        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        self.idx = np.arange(self.batch_size)
        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            for x in range(0,self.batch_size):
                self.idx[x] = random.randint(0, len(self.indices) - 1)
        else:
            for x in range(0, self.batch_size):
                self.idx[x] = x


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = np.zeros((self.batch_size, 3, self.crop_w, self.crop_h))
        self.label = np.zeros((self.batch_size, 1))
        for x in range(0,self.batch_size):
            self.data[x,] = self.load_image(self.indices[self.idx[x]])
            self.label[x,] = self.labels[self.idx[x]]

        # reshape tops to fit
        top[0].reshape(*self.data.shape)
        top[1].reshape(self.batch_size, 1)



    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        self.idx = np.arange(self.batch_size)

        # pick next input
        if self.random:
            for x in range(0,self.batch_size):
                self.idx[x] = random.randint(0, len(self.indices) - 1)

        else:
            for x in range(0,self.batch_size):
                self.idx[x] = self.idx[x] + self.batch_size

            if self.idx[self.batch_size-1] == len(self.indices):
                for x in range(0, self.batch_size):
                    self.idx[x] = x


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/images/{}'.format(self.dir, idx)).resize((self.resize_w, self.resize_h), Image.ANTIALIAS)

        if( im.size.__len__() == 2):
            im_gray = im
            im = Image.new("RGB", im_gray.size)
            im.paste(im_gray)

        if self.train: #Data Aumentation
            if(self.rotate is not 0):
                im = self.rotate_image(im)

            if self.crop_h is not self.resize_h or self.crop_h is not self.resize_h:
                im = self.random_crop(im)

            if(self.mirror and random.randint(0, 1) == 1):
                im = self.mirror_image(im)

            if(self.HSV_prob is not 0):
                im = self.saturation_value_jitter_image(im)

        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_



    #DATA AUMENTATION

    def random_crop(self,im):
        # Crops a random region of the image that will be used for training. Margin won't be included in crop.
        margin = 2
        left = random.randint(margin,self.resize_w - self.crop_w - 1 - margin)
        top = random.randint(margin,self.resize_h - self.crop_h - 1 - margin)
        im = im.crop((left, top, left + self.crop_w, top + self.crop_h))
        return im

    def mirror_image(self, im):
        return ImageOps.mirror(im)

    def rotate_image(self, im):
        return im.rotate(random.randint(-self.rotate, self.rotate))

    def saturation_value_jitter_image(self,im):
        if(random.randint(0, int(1/self.HSV_prob)) == 0):
            return im
        im = im.convert('HSV')
        data = np.array(im)  # "data" is a height x width x 3 numpy array
        data[:, :, 1] = data[:, :, 1] * random.uniform(1 - self.HSV_jitter, 1 + self.HSV_jitter)
        data[:, :, 2] = data[:, :, 2] * random.uniform(1 - self.HSV_jitter, 1 + self.HSV_jitter)
        im = Image.fromarray(data, 'HSV')
        im = im.convert('RGB')
        return im



