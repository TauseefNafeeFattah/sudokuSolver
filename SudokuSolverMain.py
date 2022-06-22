import cv2
import pandas as pd
import numpy as np
from sudokuSolverMainHelper import intializePredectionModel, preProcess
from sudokuSolverMainHelper import reorder, biggestContour, splitBoxes
from sudokuSolverMainHelper import getPredection, displayNumbers, drawGrid
from pprint import pprint
from sudokuSolvingCode import solve_sudoku

# change the number in the pathImage to test different images
# currently there are only 3 sudoku images in test_images folder
pathImage = 'test_images/1.jpg'
heightImg = 450
widthImg = 450
# get the CNN model
model = intializePredectionModel('typed.h5')

# read the whole sudoku image and resize it
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))

# create a blank image for testing and debugging
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
imgThreshold = preProcess(img)

# find all the contours of the image

# copy the image for display purpose
imgContours = img.copy()
imgBigContour = img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
# draw the contours
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

# find the biggest contour and use it as sudoku
biggest, maxArea = biggestContour(contours)

if biggest.size != 0:
    biggest = reorder(biggest)

    cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 25)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg],
                      [widthImg, heightImg]])
    # get the sudoku image and convert it to gray for easier processing
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # wrap Perspective the sudoku image
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

    # split the image and find each digit available
    imgSolvedDigits = imgBlank.copy()
    imgSolvedDigitsOnly = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)

    # get prediction for all the digits
    numbers = getPredection(boxes, model)

    # display the found numbers in a sudoku format for visuals
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers,
                                       color=(255, 0, 255))

    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    # convert the sudoku into an array and solve it
    board = np.array_split(numbers, 9)
    try:
        solve_sudoku(board)
    except Exception as e:
        print(e)
    pprint(board)

    # flatten it and display the solved solutions
    flatlist = []
    for arr in board:
        for item in arr:
            flatlist.append(item)

    # display the solved sudoku
    solvedNumbers = flatlist
    imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers,
                                     color=(255, 0, 255))
    # display only the solution numbers
    solvedNumbersOnly = flatlist*posArray
    imgSolvedDigitsOnly = displayNumbers(imgSolvedDigitsOnly,
                                         solvedNumbersOnly,
                                         color=(255, 0, 255))

    # display the sudoku by overlaying it with the original image
    pts2 = np.float32(biggest)
    pts1 = np.float32(
        [[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(
        imgSolvedDigitsOnly, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)

    # draw a grid in the solved image so that its easier to view
    imgSolvedDigits = drawGrid(imgSolvedDigits)
    cv2.imshow('', img)  # display the original image
    cv2.imshow('', imgWarpColored)  # display the extracted sudoku image
    cv2.imshow('', imgSolvedDigits)  # display the solved sudoku image
    cv2.waitKey(0)
    cv2.imshow('', inv_perspective)  # display the overlayed solution
    cv2.waitKey(0)
