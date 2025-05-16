import numpy as np
import matplotlib.pyplot as plt

# Class to generate a checkerboard pattern
class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution  # Size of the image (resolution x resolution)
        self.tile_size = tile_size    # Size of each square tile
        self.output = None            # Will hold the final pattern

    def draw(self):
        # Ensure resolution is divisible by twice the tile size
        if self.resolution % (2 * self.tile_size) != 0:
            raise ValueError("resolution must be divisible by 2 * tile_size")
        
        # Calculate how many 2x2 blocks fit into the image
        pairs = self.resolution // (2 * self.tile_size)

        # Base 2x2 checker tile (black and white)
        base = np.array([[0, 1],
                         [1, 0]], dtype=np.float32)

        # Tile the base to fill the entire board area
        board = np.tile(base, (pairs, pairs))

        # Create a tile_size x tile_size block of ones to scale up each tile
        #
        block = np.ones((self.tile_size, self.tile_size), dtype=np.float32)

        # Expand each checker tile to the appropriate size using the Kronecker product
        output = np.kron(board, block)

        self.output = output
        return output.copy()

    def show(self):
        # Draw pattern if not already drawn
        if self.output is None:
            self.draw()
        # Display the checkerboard pattern
        plt.imshow(self.output, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.show()

# Class to generate a binary image of a filled circle
class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution  # Size of the image
        self.radius = radius          # Radius of the circle
        self.position = position      # Position of the center (can be scalar or (x, y))
        self.output = None            # Will hold the final image

    def draw(self):
        # Create coordinate grids for x and y axes
        ix = np.arange(self.resolution, dtype=np.float32)
        iy = np.arange(self.resolution, dtype=np.float32)[:, None]

        # Handle both scalar and tuple positions
        pos = self.position
        if isinstance(pos, (int, float)):
            cx = cy = float(pos)  # If single value, assume square center
        else:
            cx, cy = pos          # Unpack center coordinates
        
        # Compute squared distance from the center for each pixel
        dist2 = (ix - cx)**2 + (iy - cy)**2

        # Mask pixels within the radius (creates a filled circle)
        mask = dist2 <= (self.radius ** 2)

        # Create an empty black image
        output = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        output[mask] = 1.0  # Set pixels inside the circle to white

        self.output = output
        return output.copy()

    def show(self):
        # Draw image if not already created
        if self.output is None:
            self.draw()
        # Display the circle image
        plt.imshow(self.output, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Class to generate a smooth RGB spectrum across the image
class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution  # Image resolution
        self.output = None            # Will hold the RGB image

    def draw(self):
        res = self.resolution
        # Generate normalized coordinates from 0 to 1
        x = np.linspace(0.0, 1.0, res, dtype=np.float32)
        y = np.linspace(0.0, 1.0, res, dtype=np.float32)

        # Create coordinate grid
        X, Y = np.meshgrid(x, y)

        # Define RGB color channels
        R = X             # Red increases from left to right
        G = Y             # Green increases from top to bottom
        B = 1.0 - X       # Blue decreases from left to right

        # Combine channels into one RGB image
        spectrum = np.stack([R, G, B], axis=2)

        self.output = spectrum
        return spectrum.copy()

    def show(self):
        # Draw image if not already created
        if self.output is None:
            self.draw()
        # Display the RGB spectrum image
        plt.imshow(self.output)
        plt.axis('off')
        plt.tight_layout()
        plt.show()