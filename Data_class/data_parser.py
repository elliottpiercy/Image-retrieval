import numpy as np
from os import listdir
from os.path import isfile, join
import cv2


class data():
    
    def __init__(self,images, train, validation, test):
        
        if train + validation + test != 1:
            raise ValueError("Data split must sum to 1.")
            
        self.images = images
        self.train = self.train(images[:int(len(images)*train)],np.arange(int(len(images)*train)))
        
        self.validation = self.validation(images[int(len(images)*train):int(len(images)*train)+int(len(images)*validation)],
                                          np.arange(int(len(images)*validation)))
        
        self.test = self.test(images[int(len(images)*train)+int(len(images)*validation):],
                              np.arange(int(len(images)*test)))
        
    
    # Class for training set variables batch method
    class train():
    
        def __init__(self,images,labels):
            self.images = images
            self.labels = labels
            self.num_examples = images.shape[0]
            self.batch_count = 0


        def next_batch(self,batch_size):
            next_batch = self.images[self.batch_count:self.batch_count + batch_size]
            self.batch_count += batch_size
            return next_batch   
        
        
    # Class for validation set variables
    class validation():
    
        def __init__(self,images,labels):
            self.images = images
            self.labels = labels
            self.num_examples = images.shape[0]

        
    # Class for test set variables 
    class test():
            
        def __init__(self,images,labels):
            self.images = images
            self.labels = labels
            self.num_examples = images.shape[0]
        
        
        
        
        
imgs = get_images()
satellite = data(imgs, train = 0.7, validation = 0.15, test = 0.15)