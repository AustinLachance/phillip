from __future__ import division, print_function, absolute_import

# import tflearn
import tensorflow as tf
from enum import Enum
import h5py

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, TimeDistributed, Dense, Conv2D, MaxPooling2D, Flatten, LSTM, Dropout, Reshape, Activation, Convolution2D
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint
import cv2
import os
import numpy as np
# X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from tensorflow.python.keras.models import Sequential

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# List of move classes
class Moves(Enum):
  # Single Inputs
  null = 0
  A = 1
  B = 2
  Z = 3
  Shield = 4
  Left = 5
  Right = 6
  Up = 7
  UpLeft = 8
  UpRight = 9
  UpTilt = 10
  Down = 11
  DownLeft = 12
  DownRight = 13
  DownTilt = 14
  LeftSmash = 15
  RightSmash = 16
  UpSmash = 17
  DownSmash = 18
  Jump = 19
  # Combined Inputs
  LeftA = 20
  RightA = 21
  UpA = 22
  DownA = 23
  UpTiltA = 24
  DownTiltA = 25
  LeftB = 26
  RightB = 27
  UpB = 28
  UpLeftB = 29
  UpRightB = 30
  DownB = 31
  LeftJump = 32
  RightJump = 33
  JumpZ = 34
  LeftJumpZ = 35
  RightJumpZ = 36
  LeftShield = 37
  RightShield = 38

# Dictionary mapping moves to their index
moveDict = {item: val.value for item, val in Moves.__members__.items()}


# Time distributed alexnet with additional lstm layer after the last pooling layer
def buildCRNN(framesPerSequence, imageW, imageH, channels):
  # Building 'AlexNet'
  # model = Sequential()
  print("TESTING")
  print(framesPerSequence)
  print(imageW)
  print(imageH)
  print(channels)

  # construct input to network
  inputs = Input(shape=(framesPerSequence, imageW, imageH, channels))

  # Send each frame in a frame sequence through the CNN layers in parallel
  conv1 = TimeDistributed(Conv2D(96, (7, 7), strides=(2, 2), activation='relu'))(inputs)
  pool1 = TimeDistributed(MaxPooling2D((3, 3), strides=(3, 3)))(conv1)
  conv2 = TimeDistributed(Conv2D(256, (5,5), strides=(1, 1), activation='relu'))(pool1)
  pool2 = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(conv2)
  conv3 = TimeDistributed(Conv2D(512, (3,3), strides=(1, 1), activation='relu'))(pool2)
  conv4 = TimeDistributed(Conv2D(512, (3,3), strides=(1, 1), activation='relu'))(conv3)
  conv5 = TimeDistributed(Conv2D(512, (3,3), strides=(1, 1), activation='relu'))(conv4)
  pool3 = TimeDistributed(MaxPooling2D((3, 3), strides=(3, 3)))(conv5)

  # Flatten and RNN netowrk
  flat1 = Flatten(name='flatten')(pool3)
  # lstm = LSTM(4096)(flatCNN)

  # Fully connected layers and output
  full1 = Dense(4096, activation='tanh',name='full1')(flat1)
  drop1 = Dropout(0.5)(full1)
  full2 = Dense(4096, activation='tanh',name='full2')(drop1)
  drop2 = Dropout(0.5)(full2)
  crnnOut = Dense(39, activation='softmax', name='crnnOut')(drop2)

  # Construct model and compile with rms optimizer
  crnn = Model(inputs, crnnOut)

  # rmsprop = RMSprop(lr=0.0001)
  crnn.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  # crnn.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])

  return crnn

# Add randomization to the data sequences?
# Might be too slow. Preprocess data into sequences?
# Use when the data is a set of frames. Data shape (numFrames, ImageW, ImageH, channels). Manually create sequences
def createBatchMakeSequences(data, labels, numSamples, numFrames, startIndex):

  # Check that enough data exists to make the batch
  if(startIndex + (numFrames * numSamples) >= len(data)):
    print("Warning: not enough data to create another full batch")
    return

  # Initialize size of training data
  x_train = np.zeros((numSamples, numFrames, imageW, imageH, channels))

  # Get the labels
  y_train = labels[startIndex:startIndex + numFrames*numSamples]

  # Get the training data for the batch
  for i in range (0, numSamples):
    x_train[i] = data[startIndex:startIndex+numFrames]
    startIndex = startIndex + numFrames

  return (x_train, y_train)


# Use when the data has been pre-divided into sequences of frames. Data shape (numSamples, numFrames, ImageW, ImageH, channels)
def createBatchPreSequenced(data, labels, numSamples, numFrames, startIndex):
  # Check that enough data exists to make the batch
  if(startIndex + (numFrames * numSamples) >= len(data)):
    print("Warning: not enough data to create another full batch")
    return

  # Initialize size of training data
  x_train = np.zeros((numSamples, numFrames, imageW, imageH, channels))

  # Get the training data for the batch
  x_train = data[startIndex:startIndex + numSamples]

  # Get the labels
  y_train = labels[startIndex:startIndex + numFrames*numSamples]

  return x_train, y_train


# Input shape (numFrames, imageW, imageH, channels)
# Output shape (numSequences, numFrames, imagew, imageH, channels)
# For frames 1-10 with 3 frames per sequence, generates sequences 1,2,3 ; 4, 5, 6; 7, 8, 9; ... 
def makeSequencesSeparate(data, framesPerSequence, start):
  numFrames = len(data)
  maxSequences = numFrames/framesPerSequence
  if(numFrames%framesPerSequence != 0):
    maxSequences += 1

  x_train = []
  y_train = []

  # Generate sequences
  for i in range(0, maxSequences):
    x_train.append(data[start:start+framesPerSequence])
    y_train.append(labels[start+framesPerSequence])

  return x_train, y_train


# Input shape => data = (numFrames, imageW, imageH, channels), labels = (numFrames)
# Output shape => x_train = (numSequences, numFrames, imagew, imageH, channels), y_train = (numSequences)
# For frames 1-10 with 3 frames per sequence, generates sequences 1,2,3 ; 2, 3, 4; 3, 4, 5; ... 
def makeSequencesRedundant(data, labels, framesPerSequence, start, numSequences):
  # numFrames = len(data)
  # print(numFrames)
  # maxSequences = numFrames - framesPerSequence

  # Initialize output
  # x_train = np.zeros((maxSequences, numFrames, imageW, imageH, channels))
  # y_train = np.zeros(maxSequences)

  # For testing, just add the names of the frame images not the images themselves
  # x_train = np.empty((maxSequences, framesPerSequence), dtype = str)
  # y_train = np.empty(maxSequences, dtype = str)
  x_train = []
  y_train = np.zeros((numSequences, 39))

  # Generate sequences
  for i in range(0, numSequences):
    x_train.append(data[start:start+framesPerSequence])
    y_train[i][moveDict[labels[start+framesPerSequence]]] = 1.0
    start+=1

  return x_train, y_train


def trainModel(model, x_train, y_train, batchSize):
  x_train, y_train = createBatchPreSequenced()
  model.fit(self, x_train, y_train, batch_size=batchSize, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)



# Input: Directory with images of frames, # of first image, # of last image
# Output: Array of images size lastImage - firstImage
def getImageData(framesDirectory, firstImage, lastImage, imageW, imageH):

  data = []
  for imageNum in range(firstImage, lastImage):

    # Construct image path name
    imageNumStr = str(imageNum)
    fileName = framesDirectory + "/IMG_" + imageNumStr.zfill(4) + '.png'

    # Open and read in image of frame
    img = cv2.imread(fileName, 1)

    # Crop the image
    img = img[0:1800, 344:2532]

    # Downsize image
    imgResized = cv2.resize(img, (imageW, imageH), interpolation=cv2.INTER_LINEAR)

    # Size test print
    # size = img.shape
    # print(size)

    data.append(imgResized)

  return data


# Input: labels file name
# Output: array of labels from file
def getLabels(fileName):
  with open(fileName) as labels_file:
      labels = labels_file.readlines()

  # you may also want to remove whitespace characters like `\n` at the end of each line
  labels = [x.strip() for x in labels]
  return labels

# numSeq, numFrames, imgW, imgH, channels = x_train.shape

# print(x_train.shape)

# timesteps=10;
# number_of_samples=250;
# nb_samples=number_of_samples;
# frame_row=32;
# frame_col=32;
# channels=3;

# nb_epoch=50;
# batch_size=timesteps;


# data= np.random.random((250,timesteps,frame_row,frame_col,channels))
# label=np.random.random((250,1))

# X_train=data[0:200,:]
# y_train=label[0:200]

# X_test=data[200:,:]
# y_test=label[200:,:]
# print('created data')
#%%

# model=Sequential();                          

# model.add(TimeDistributed(Convolution2D(32, 3, 3), input_shape=X_train.shape[1:]))
# model.add(TimeDistributed(Activation('relu')))
# model.add(TimeDistributed(Convolution2D(32, 3, 3)))
# model.add(TimeDistributed(Activation('relu')))
# model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
# model.add(TimeDistributed(Dropout(0.25)))

# model.add(TimeDistributed(Flatten()))
# model.add(TimeDistributed(Dense(512)))
                
                
# model.add(TimeDistributed(Dense(35, name="first_dense" )))
        
# model.add(LSTM(20, return_sequences=True, name="lstm_layer"));
         
# #%%
# model.add(TimeDistributed(Dense(1), name="time_distr_dense_one"))
# model.add(GlobalAveragePooling1D(name="global_avg"))


# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# data = getImageData("game5_frames", 1, 3419, 150, 150)
# print("Got Images")
# labels = getLabels("game5_labels")
# print("Got Labels")

# numFrames = len(data)
# framesPerSequence = 4

# # Total number of sequences
# maxSequences = numFrames - framesPerSequence
# batchSize = 25
# stepPerEpoch = int(maxSequences/batchSize)

# imgW = 150
# imgH = 150
# channels = 3

# model = buildCRNN(framesPerSequence, imgW, imgH, channels)
# print(model.summary())
# modelPath = "crnn_noLSTM.h5"
# checkpoint = ModelCheckpoint(modelPath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
# callbacks_list = [checkpoint]

# numEpochs = 5

# for epoch in range(0, numEpochs):
#   for step in range(0, stepPerEpoch):
#     x_train, y_train = makeSequencesRedundant(data, labels, framesPerSequence, step*batchSize, batchSize)
    
#     # Convert to numpy arrays
#     x_train = np.array(x_train)
#     y_train = np.array(y_train)
#     print("Epoch " + str(epoch))

#     # model.train_on_batch(x_train, y_train)
#     model.fit(x=x_train, y=y_train, batch_size=25, epochs=1, verbose=2, callbacks=callbacks_list, validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)

# x_test, y_test = makeSequencesRedundant(data, labels, framesPerSequence, batchSize, batchSize)
# x_test = np.array(x_test)
# model.load_weights(modelPath)
# results = model.predict_on_batch(x_test)
# print(y_test)
# print(results)
# x_test, y_test = makeSequencesRedundant(data, labels, framesPerSequence, )

# modelPath = "crnn_noLSTM.h5"
# checkpoint = ModelCheckpoint(modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

# print(nb_epoch)

# model.fit(X_train, y_train,
#           batch_size=batch_size,
#           epochs=nb_epoch,
#           validation_data=(X_test, y_test))
# model.fit(x=x_train, y=y_train, batch_size=32, epochs=1, verbose=2, callbacks=callbacks_list, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)






