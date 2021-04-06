""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_images

FLAGS = flags.FLAGS

class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size 
        self.num_samples_per_class = num_samples_per_class #in 1-shot we have just one image per class, in 5 shot we have 5 images per class and so on 
        self.num_classes = 1  # by default 1 (only relevant for classification problems)
        
        if 'omniglot' in FLAGS.datasource:
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (28, 28))
            self.dim_input = np.prod(self.img_size)
            self.dim_output = self.num_classes
            
            # data that is pre-resized using PIL with lanczos filter
            data_folder = config.get('data_folder', './data/omniglot_resized')
              
            character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
            
            #Each family corresponds to one alphabet among 50, each alphabet has a different number of characters 
            #Each folder in character_folders  is related to a specific alphabet and a specific character and contains 20 images. 
                
            random.seed(1)
            random.shuffle(character_folders)
            
            #The meta-train character folders are the first 1200 character_folders. 
            #The meta-val character folders are the next 100 character_folders
            
            num_val = 100
            num_train = config.get('num_train', 1200) - num_val
            self.metatrain_character_folders = character_folders[:num_train]
            
            if FLAGS.test_set:
            #The meta-test character folders are the remaining ones (323)
                self.metaval_character_folders = character_folders[num_train+num_val:]
            else:
                self.metaval_character_folders = character_folders[num_train:num_train+num_val]
                
            #for the omniglot dataset a data augmentation with rotations is performed to reduce overfitting 
            #(all character images are not so different even if the characters have been written by 20 different persons)
            self.rotations = config.get('rotations', [0, 90, 180, 270])
            
        elif FLAGS.datasource == 'miniimagenet':
            
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size)*3
            self.dim_output = self.num_classes
            
            #minimagenet dataset is divided into train.csv, val.csv and test.csv 
            
            metatrain_folder = config.get('metatrain_folder', './data/miniImagenet/train')
            
            if FLAGS.test_set:
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/test')
            else:
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/val')
            #preparing train folders
            metatrain_folders = [os.path.join(metatrain_folder, label) \
                for label in os.listdir(metatrain_folder) \
                if os.path.isdir(os.path.join(metatrain_folder, label)) \
                ]
            #preparing validation folders
            metaval_folders = [os.path.join(metaval_folder, label) \
                for label in os.listdir(metaval_folder) \
                if os.path.isdir(os.path.join(metaval_folder, label)) \
                ]
            
            self.metatrain_character_folders = metatrain_folders
            self.metaval_character_folders = metaval_folders
            self.rotations = config.get('rotations', [0])
        else:
            raise ValueError('Unrecognized data source')


    def make_data_tensor(self, train=True):
        if train:
            folders = self.metatrain_character_folders #omniglot
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            #in 5way omniglot we have metabatch size = 32 so 200000/32 = 6250 iterations
            #in 5way minimagenet we have metabatch size = 4 so 200000/4 = 500000 iterations 
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600

        # make list of files
        print('Generating filenames')
        
        all_filenames = []
        
        for _ in range(num_total_batches): #for each of the 200000 tasks
            #from characters folders sample n-way=num_classes classes randomly
            sampled_character_folders = random.sample(folders, self.num_classes)
            random.shuffle(sampled_character_folders 
            labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
            # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        
        if FLAGS.datasource == 'miniimagenet':
            image = tf.image.decode_jpeg(image_file, channels=3)
            image.set_shape((self.img_size[0],self.img_size[1],3)) # tensorflow format: N*H*W*C
            image = tf.reshape(image, [self.dim_input]) #reshape(image,[84*84*3])
            image = tf.cast(image, tf.float32) / 255.0  #convert to range (0,1)
        else:
            
            image = tf.image.decode_png(image_file)
            image.set_shape((self.img_size[0],self.img_size[1],1)) # tensorflow format: N*H*W*C
            image = tf.reshape(image, [self.dim_input]) #reshape(image,[84*84*1])
            image = tf.cast(image, tf.float32) / 255.0 #convert to range (0,1)
            image = 1.0 - image  # invert 
            
        num_preprocess_threads = 1 # TODO - enable this to be set to >1
        min_queue_examples = 256
        
        examples_per_batch = self.num_classes * self.num_samples_per_class #5*16
        batch_image_size = self.batch_size  * examples_per_batch #
        print('Batching images')
        images = tf.train.batch(
                [image],
                batch_size = batch_image_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_image_size,
                )
        all_image_batches, all_label_batches = [], []
        print('Manipulating image data to be right shape')
        for i in range(self.batch_size):
            image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch]

            if FLAGS.datasource == 'omniglot':
                # omniglot augments the dataset by rotating digits to create new classes
                # get rotation per class (e.g. 0,1,2,0,0 if there are 5 classes)
                rotations = tf.multinomial(tf.log([[1., 1.,1.,1.]]), self.num_classes)
            label_batch = tf.convert_to_tensor(labels)
            new_list, new_label_list = [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)

                true_idxs = class_idxs*self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch,true_idxs))
                if FLAGS.datasource == 'omniglot': # and FLAGS.train:
                    new_list[-1] = tf.stack([tf.reshape(tf.image.rot90(
                        tf.reshape(new_list[-1][ind], [self.img_size[0],self.img_size[1],1]),
                        k=tf.cast(rotations[0,class_idxs[ind]], tf.int32)), (self.dim_input,))
                        for ind in range(self.num_classes)])
                new_label_list.append(tf.gather(label_batch, true_idxs))
            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

