#pickle_reader dp ogarniecia co sie znajduje w plikach .p
import pickle
import cv2
def main(filename):
    objects = []
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects
if __name__ == '__main__':
    #filename="full_CNN_labels.p"
    filename = "full_CNN_train.p"
    objects = main(filename)
    print(len((objects)))
    print(len((objects[0])))
    print(len((objects[0][0])))

    for i in (objects[0]):
        print((i))
        cv2.namedWindow("pickle", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("pickle", 1280,720)
        cv2.imshow("pickle", i)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


