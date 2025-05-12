import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        if self.resolution % (2 * self.tile_size) != 0:
            raise ValueError("resolution must be divisible by 2 * tile_size")
        pairs = self.resolution // (2 * self.tile_size)

        base = np.array([[0, 1],
                         [1, 0]], dtype=np.float32)

        board = np.tile(base, (pairs, pairs))

        block = np.ones((self.tile_size, self.tile_size), dtype=np.float32)
        output = np.kron(board, block)

        self.output = output
        return output.copy()

    def show(self):
        if self.output is None:
            self.draw()
        plt.imshow(self.output, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.show()

class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        # Create coordinate arrays
        ix = np.arange(self.resolution, dtype=np.float32)
        iy = np.arange(self.resolution, dtype=np.float32)[:, None]
        
        # Handle position parameter
        pos = self.position
        if isinstance(pos, (int, float)):
            cx = cy = float(pos)
        else:
            cx, cy = pos
            
        # Calculate squared distance from center
        dist2 = (ix - cx)**2 + (iy - cy)**2
        
        # Create binary mask where distance <= radius
        mask = dist2 <= (self.radius ** 2)
        
        # Create output array
        output = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        output[mask] = 1.0
        
        self.output = output
        return output.copy()

    def show(self):
        if self.output is None:
            self.draw()
        plt.imshow(self.output, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None
    
    def draw(self):
        res = self.resolution
        x = np.linspace(0.0, 1.0, res, dtype=np.float32)  
        y = np.linspace(0.0, 1.0, res, dtype=np.float32)
        
        # Create coordinate grid
        X, Y = np.meshgrid(x, y)
        
        # Create RGB channels
        R = X                    
        G = Y                    
        B = 1.0 - X              
        
        # Stack channels to create RGB image
        spectrum = np.stack([R, G, B], axis=2)
        
        self.output = spectrum
        return spectrum.copy()
    
    def show(self):
        if self.output is None:
            self.draw()
        plt.imshow(self.output)  # Remove cmap and vmin/vmax for RGB
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        