from Preprocessing import Preprocessing


def main():
    image = Preprocessing('something.png')
    image.grayscale()
    image.save('え.png')


if __name__ == "__main__":
    main()
