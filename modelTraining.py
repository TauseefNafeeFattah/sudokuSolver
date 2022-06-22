import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


# prepocess function for the images used in training
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


def Model():
    '''
    Creates the CNN model
    '''
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1,
                      input_shape=(imageDimensions[0], imageDimensions[1], 1),
                      activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# parameters
path = 'myData'
testRatio = 0.2
valRatio = 0.2
imageDimensions = (32, 32, 3)
batch_size = 50
epochsVal = 10

# importing data from folders
count = 0
images = []
classNo = []
myList = os.listdir(path)
noOfClasses = len(myList)
print("Total Classes Detected:" + str(noOfClasses))

print("Importing Classes")

for i in range(0, noOfClasses):
    myImageList = os.listdir(path+"/"+str(i))
    for j in myImageList:
        curImg = cv2.imread(path+"/"+str(i)+"/"+j)
        curImg = cv2.resize(curImg, (32, 32))
        images.append(curImg)
        classNo.append(i)
print("Total images in the images list =" + str(len(images)))
print("Total classes in the classNo list =" + str(len(classNo)))

# convert images and classNo to numpy array
images = np.array(images)
classNo = np.array(classNo)

# split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
                                    images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(
                                    X_train, y_train, test_size=valRatio)


# processes the image to be used in training
X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

# reshape the images
X_train = X_train.reshape(
                X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(
                X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(
                X_validation.shape[0], X_validation.shape[1],
                X_validation.shape[2], 1)

# image augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

# convert y_train, y_test, y_validation to one hot encoding of matrices
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

# create the model
model = Model()

# train the model
history = model.fit(
            dataGen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train)//batch_size,
            epochs=epochsVal,
            validation_data=(X_validation, y_validation),
            shuffle=1)

# plot the result for easier viewing
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# evaluate the model using test images
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ' + str(score[0]))
print('Test Accuracy = ' + str(score[1]))

# save the trained model
model.save('typed.h5')
