import cv2
import numpy as np
from tensorflow.keras.models import load_model


def intializePredectionModel(modelName):
    '''
    loads the convolutional neural network(CNN) model
    output- CNN model
    '''
    model = load_model(modelName)
    return model


def preProcess(img):
    '''
    preProcess the image to be put into the CNN model
    Converts the image to gray scale, add gaussian blur and
    apply the adaptive threshold.
    input- image
    output- image
    '''
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThreshold


def reorder(myPoints):
    '''
    reorder the points for Warp Perspective
    '''
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def biggestContour(contours):
    '''
    finds the biggest contour and assumes that its the sudoku puzzle
    input-
        contours- array of contour
    output-
        biggest- array([]) (the biggest contour)
        max_area- int (maximum area of the contour)
    '''
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def splitBoxes(img):
    '''
    Splits the image vertically and horizontally into 9 parts each (9*9=81)
    input:
        img- image
    output:
        boxes- array of images
    '''
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes


def getPredection(boxes, model):
    '''
    gets the prediction of the images in the box according to the model
    input:
        boxes: array of images
        model: CNN model
    output:
        result: an array of int (the array of predicted values)
    '''
    result = []
    for image in boxes:
        # prepare the images for the model

        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img, (32, 32))
        img = img / 255
        img = img.reshape(1, 32, 32, 1)

        # get the prediction for the image
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=1)
        probabilityValue = np.amax(predictions)

        # add the prediction to the result if its propbabilty value is greater
        # than 0.7 else add a 0 to the result
        if probabilityValue > 0.7:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result


def displayNumbers(img, numbers, color=(0, 255, 0)):
    '''
    display the numbers in an image
    input:
        img: image
        numbers: array of int ([int])
        color: color of the number (RGB format)
    output:
        img- image
    '''
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y*9)+x] != 0:
                cv2.putText(img, str(numbers[(y*9)+x]),
                            (x*secW+int(secW/2)-10, int((y+0.8)*secH)),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2,
                            cv2.LINE_AA)
    return img


def drawGrid(img):
    '''
    modifies the image and draws a grid in the image
    input:
        img- image
    output:
        img- image
    '''
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range(0, 9):
        pt1 = (0, secH*i)
        pt2 = (img.shape[1], secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i, img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)
        cv2.line(img, pt3, pt4, (255, 255, 0), 2)
    return img
