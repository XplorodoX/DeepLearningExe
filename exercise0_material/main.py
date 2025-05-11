from pattern import Checker, Circle

def main():
    # Create a checkerboard pattern
    checker = Checker(resolution=512, tile_size=16)
    checker.draw()
    checker.show()

    # Create a circle pattern
    circle = Circle(resolution=512, position=(256, 256), radius=100)
    circle.draw()
    circle.show()

if __name__ == "__main__":
    main()