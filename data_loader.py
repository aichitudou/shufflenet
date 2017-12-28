import numpy as np

import scipy.misc
from scipy.misc import imread
import os

data_path = '/home/cdpt/dataset/tiny-imagenet-200'


class DataLoader:
    """Data Loader class. As a simple case, the model is tried on TinyImageNet. For larger datasets,
    you may need to adapt this class to use the Tensorflow Dataset API"""

    def __init__(self, batch_size, shuffle=False):
        self.X_train = None
        self.X_mean = None
        self.y_train = None
        self.train_data_len = 0

        self.X_val = None
        self.y_val = None
        self.val_data_len = 0

        self.X_test = None
        self.y_test = None
        self.test_data_len = 0

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.class_name = None

    def load_tiny_imagenet(self):

	# First load wnids
        wnids_file = os.path.join(data_path, 'wnids.txt')

        with open(wnids_file, 'r') as f:

            wnids = [x.strip() for x in f]


	# Map wnids to integer labels
        wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

	# Use words.txt to get names for each class
        words_file = os.path.join(data_path, 'words.txt')
        with open(words_file, 'r') as f:
            wnid_to_words = dict(line.split('\t') for line in f)

            for wnid, words in wnid_to_words.items():
                wnid_to_words[wnid] = [w.strip() for w in words.split(',')]

        class_names = [wnid_to_words[wnid] for wnid in wnids]

	# Next load training data
        X_train = []
        y_train = []

        for i, wnid in enumerate(wnids):
            if (i+1) % 20 == 0:
                print ('loading training data for synset %d / %d' % (i+1, len(wnids)))
            boxes_file = os.path.join(data_path, 'train', wnid, '%s_boxes.txt' % wnid)

            with open(boxes_file, 'r') as f:
                filenames = [x.split('\t')[0] for x in f]
            num_images = len(filenames)

            X_train_block = np.zeros((num_images,3,64,64),dtype=np.float32)

            y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)

            for j, img_file in enumerate(filenames):
                img_file = os.path.join(data_path, 'train', wnid, 'images', img_file)
                img = imread(img_file)
                
                if img.ndim == 2:
                    ## grayscale file
                    img.shape=(64,64,1)

                X_train_block[j] = img.transpose(2,0,1)

            X_train.append(X_train_block)
            y_train.append(y_train_block)


	# We need to concatenate all training data
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)


	#print (np.array(y_train).shape)

	# Next load validation data
        val_anno_file = os.path.join(data_path, 'val', 'val_annotations.txt')
        with open(val_anno_file, 'r') as f:
            img_files = []
            val_wnids = []

            for line in f:
                # Select only validation images in chosen wnids set
                if line.split()[1] in wnids:
                    img_file, wnid = line.split('\t')[:2]
                    img_files.append(img_file)
                    val_wnids.append(wnid)

                num_val = len(img_files)
                y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])

                X_val = np.zeros((num_val, 3,64,64),dtype=np.float32)

                for i, img_file in enumerate(img_files):
                    img_file =  os.path.join(data_path, 'val', 'images', img_file)
                    img = imread(img_file)
                    if img.ndim == 2:
                        img.shape=(64,64,1)

                    X_val[i] = img.transpose(2,0,1)
        return class_names, X_train, y_train, X_val, y_val


    def load_vehicles(self):

        X_train =[]
        y_train =[]
        
        train_file = os.path.join('data','train_random.txt')
        val_file = os.path.join('data', 'val_random.txt')
       

        with open(train_file, 'r') as f:

            train_img = [x for x in f]

        train_imgnames = [x.split(' ')[0] for x in train_img]
        y_train = [int(x.strip().split(' ')[1]) for x in train_img]
        
        for img_file in train_imgnames:
            img = imread(img_file)
            X_train.append(img)
        
        X_val=[]
        y_val=[]
        with open(val_file, 'r') as f:
            val_img = [x for x in f]

        val_imgnames = [x.split(' ')[0] for x in val_img]
        y_val = [int(x.strip().split(' ')[1]) for x in val_img]

        for img_file in val_imgnames:
            img = imread(img_file)
            X_val.append(img)
         

        return np.array(X_train,dtype=np.float32), np.array(y_train), np.array(X_val,dtype=np.float32), np.array(y_val)
    
    
    def load_data(self):
        # This method is an example of loading a dataset. Change it to suit your needs..
        import matplotlib.pyplot as plt
        #class_names, X_train, y_train, X_val, y_val = self.load_tiny_imagenet()        
        X_train, y_train, X_val, y_val = self.load_vehicles()
	# For going in the same experiment as the paper. Resizing the input image data to 224x224 is done.
       
        self.X_train = X_train
        self.y_train = y_train

        
        self.X_val = X_val
        self.y_val = y_val

        self.train_data_len = self.X_train.shape[0]
        self.val_data_len = self.X_val.shape[0]
        img_height = 64
        img_width = 64
        num_channels = 3
        return img_height, img_width, num_channels, self.train_data_len, self.val_data_len

    def generate_batch(self, type='train'):
        """Generate batch from X_train/X_test and y_train/y_test using a python DataGenerator"""
        if type == 'train':
            # Training time!
            new_epoch = True
            start_idx = 0
            mask = None
            while True:
                if new_epoch:
                    start_idx = 0
                    if self.shuffle:
                        mask = np.random.choice(self.train_data_len, self.train_data_len, replace=False)
                    else:
                        mask = np.arange(self.train_data_len)
                    new_epoch = False

                # Batch mask selection
                X_batch = self.X_train[mask[start_idx:start_idx + self.batch_size]]
                y_batch = self.y_train[mask[start_idx:start_idx + self.batch_size]]
                start_idx += self.batch_size

                # Reset everything after the end of an epoch
                if start_idx >= self.train_data_len:
                    new_epoch = True
                    mask = None
                yield X_batch, y_batch
        elif type == 'test':
            # Testing time!
            start_idx = 0
            while True:
                # Batch mask selection
                X_batch = self.X_test[start_idx:start_idx + self.batch_size]
                y_batch = self.y_test[start_idx:start_idx + self.batch_size]
                start_idx += self.batch_size

                # Reset everything
                if start_idx >= self.test_data_len:
                    start_idx = 0
                yield X_batch, y_batch
        elif type == 'val':
            # Testing time!
            start_idx = 0
            while True:
                # Batch mask selection
                X_batch = self.X_val[start_idx:start_idx + self.batch_size]
                y_batch = self.y_val[start_idx:start_idx + self.batch_size]
                start_idx += self.batch_size

                # Reset everything
                if start_idx >= self.val_data_len:
                    start_idx = 0
                yield X_batch, y_batch
        else:
            raise ValueError("Please select a type from \'train\', \'val\', or \'test\'")




