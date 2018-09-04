import cv2
import sys
import os.path

def detect(filename, outfolder, cascade_file = "lbpcascade_animeface.xml", ):

    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier (cascade_file)
    image   = cv2.imread            (filename, cv2.IMREAD_COLOR)
    gray    = cv2.cvtColor          (image, cv2.COLOR_BGR2GRAY)
    gray    = cv2.equalizeHist      (gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor  = 1.1,
                                     minNeighbors = 5,
                                     minSize      = (24, 24))
    x = 0
    y = 0
    w = 0
    h = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #print(x, y, w, h)

    img = filename.split('/')[-1].split('.')[0] + ".png"

    cv2.imwrite("./" + outfolder + "/" + img, image)
    if x or y or w or h:
        return True
    else:
        return False
def main():

    direct = "./training_data/miku/"

    cropped = 0
    all     = 0

    for filename in os.listdir(direct):
        #print(filename)
        if detect(direct + filename, "out"):
            cropped += 1
        all += 1

        print(cropped/all)
    print("cropped: ", cropped, "\nall: ", all)
main()