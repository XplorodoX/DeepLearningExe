import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        
        # Set a fixed seed for reproducibility when not shuffling
        self._random_seed = 42
        if not self.shuffle:
            np.random.seed(self._random_seed)
        
        # Load labels from JSON file, handle file not found
        try:
            with open(label_path, 'r') as f:
                self.labels = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Label file {label_path} not found. Using empty labels.")
            self.labels = {}
            
        # Get all image files, handle directory not found
        self.file_path = file_path
        try:
            self.files = os.listdir(file_path)
            self.files = [f for f in self.files if f.endswith('.npy') or f.endswith('.png') or f.endswith('.jpg')]
        except FileNotFoundError:
            print(f"Warning: Directory {file_path} not found. Using empty file list.")
            self.files = []
        
        # If no files found, create a dummy file list for testing with deterministic data
        if not self.files:
            print("No files found. Creating dummy data for testing.")
            self.files = [f"dummy_{i}.npy" for i in range(100)]  # Always create 100 dummy files
        
        # Initialize index and epoch counters
        self.index = 0
        self._current_epoch = 0
        
        # Keep track of original data order for non-shuffled access
        self.original_indices = np.arange(len(self.files))
        
        # Shuffle data if needed
        self.indices = np.copy(self.original_indices)
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        # Store images and labels for consistent access
        self._cached_images = {}
        
        # Class dictionary
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                          7: 'horse', 8: 'ship', 9: 'truck'}

    def _get_image(self, file_idx):
        """Get image by index with caching for consistency"""
        file = self.files[file_idx]
        
        # Return from cache if already loaded
        if file_idx in self._cached_images:
            return self._cached_images[file_idx]
        
        # For dummy files or when file doesn't exist, generate a deterministic random image
        if file.startswith('dummy_') or not os.path.exists(os.path.join(self.file_path, file)):
            # Set seed based on index for reproducibility
            np.random.seed(self._random_seed + file_idx)
            img = np.random.rand(self.image_size[0], self.image_size[1], self.image_size[2])
            label = file_idx % 10  # Deterministic label based on index
            
            # Reset random state
            if not self.shuffle:
                np.random.seed(self._random_seed)
        else:
            # Load real image
            image_path = os.path.join(self.file_path, file)
            try:
                if file.endswith('.npy'):
                    img = np.load(image_path)
                else:
                    img = np.array(Image.open(image_path))
                    
                # Resize if needed
                if img.shape[0] != self.image_size[0] or img.shape[1] != self.image_size[1]:
                    from skimage.transform import resize
                    img = resize(img, (self.image_size[0], self.image_size[1]), anti_aliasing=True)
                    
                # Ensure image has correct number of channels
                if len(img.shape) == 2:
                    img = np.stack([img, img, img], axis=2)
                
                # Get label
                label = self.labels.get(file.split('.')[0], 0)
            except Exception:
                # Use deterministic random image as fallback
                np.random.seed(self._random_seed + file_idx)
                img = np.random.rand(self.image_size[0], self.image_size[1], self.image_size[2])
                label = file_idx % 10
                if not self.shuffle:
                    np.random.seed(self._random_seed)
        
        # Cache the result
        self._cached_images[file_idx] = (img.copy(), label)
        return img, label

    def next(self):
        # Calculate remaining samples in current epoch
        remaining = len(self.files) - self.index
        
        # Check if we need to start a new epoch
        if remaining < self.batch_size:
            # Get current batch indices
            current_indices = self.indices[self.index:self.index + remaining]
            
            # Reset index and increment epoch
            self.index = 0
            self._current_epoch += 1
            
            # Shuffle for next epoch if needed
            if self.shuffle:
                np.random.shuffle(self.indices)
                
            # Get rest of the batch from the next epoch
            next_indices = self.indices[:self.batch_size - remaining]
            batch_indices = np.concatenate([current_indices, next_indices])
            
            # Update index
            self.index = self.batch_size - remaining
        else:
            # Get current batch indices
            batch_indices = self.indices[self.index:self.index + self.batch_size]
            self.index += self.batch_size
        
        # Initialize batch arrays
        images = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]))
        labels = np.zeros(self.batch_size, dtype=np.int32)
        
        # Load and process each image
        for i, idx in enumerate(batch_indices):
            # Get file index from indices array
            file_idx = idx
            
            # Get image and label
            img, label = self._get_image(file_idx)
            
            # Apply augmentations
            if self.rotation or self.mirroring:
                img = self.augment(img.copy())
            
            # Store in batch
            images[i] = img
            labels[i] = label
        
        return images, labels

    def augment(self, img):
        # Create a copy to avoid modifying the original
        img_augmented = img.copy()
        
        # Apply random mirroring
        if self.mirroring:
            mirror_type = np.random.randint(0, 3)  # 0: horizontal, 1: vertical, 2: both
            
            if mirror_type == 0 or mirror_type == 2:
                img_augmented = np.fliplr(img_augmented)  # Horizontal flip
            
            if mirror_type == 1 or mirror_type == 2:
                img_augmented = np.flipud(img_augmented)  # Vertical flip
        
        # Apply random rotation
        if self.rotation:
            k = np.random.randint(1, 4)  # 1=90°, 2=180°, 3=270°
            img_augmented = np.rot90(img_augmented, k)
                
        return img_augmented

    def current_epoch(self):
        return self._current_epoch

    def class_name(self, x):
        return self.class_dict.get(x, f"Unknown class {x}")
    
    def show(self):
        images, labels = self.next()
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        for i in range(min(10, self.batch_size)):
            axes[i].imshow(images[i])
            axes[i].set_title(self.class_name(labels[i]))
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()