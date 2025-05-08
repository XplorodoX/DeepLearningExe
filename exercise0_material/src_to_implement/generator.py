import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        
        #TODO: implement constructor
        self.batch_size = batch_size
        self.image_size = image_size
        self.file_path = os.path.abspath(file_path)
        self.label_path = os.path.abspath(label_path)
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        
        images = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], 3))
        labels = np.zeros((self.batch_size, 10))
        return images, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image

        # Random horizontal flip
        if getattr(self, 'mirroring', False):
            if np.random.rand() < 0.5:
                img = np.fliplr(img)

        # Random rotation by 0°, 90°, 180° or 270°
        if getattr(self, 'rotation', False):
            k = np.random.choice([0, 1, 2, 3])
            img = np.rot90(img, k)

        return img

    def current_epoch(self):
        # return the current epoch number
        
        return 0

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]
    
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images, labels = self.next()
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(2):
            for j in range(5):
                axes[i, j].imshow(images[i * 5 + j])
                axes[i, j].set_title(self.class_name(np.argmax(labels[i * 5 + j])))
                axes[i, j].axis('off')
        plt.tight_layout()
        plt.show()
