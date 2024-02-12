from Preprocessing import Preprocessing


def main():
    image = Preprocessing('C:/Users/prszy/Desktop/something.png')
    image.grayscale()
    image.save('C:/Users/prszy/Desktop/え.png')

if __name__ == "__main__":
    main()
