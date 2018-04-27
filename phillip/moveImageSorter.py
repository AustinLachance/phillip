import numpy as np
import cv2
import os
from enum import Enum


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

moveLst = [item for item, val in Moves.__members__.items()]
print(moveLst)

# Dont allow angled A inputs or downangled B inputs
angled = ["DownLeft", "DownRight", "UpLeft", "UpRight"]


labels = []
moves = {}

# Images are expected in the format "IMG_(image_number).png"

# image_number of first image and last image in the images folder
firstImage = 1
lastImage = 3419


# Name of the images directory and the desired name for the generated labels file
labelFileName = 'game5_labels'
imageDirectory = 'game5_frames'

f= open(labelFileName,"w+")


for imageNum in range(firstImage, lastImage):

	# Construct image path name
	imageNumStr = str(imageNum)
	fileName = imageDirectory + "IMG_" + imageNumStr.zfill(4) + '.png'

	img = cv2.imread(fileName,1)
	# reshape
	img = img[0:1800,344:2532]
	# [rows, columns, colors]
	size = img.shape

	# ratio = 9.767
	# im_resized = cv2.resize(img, (224, 184), interpolation=cv2.INTER_LINEAR)
	# hud = img[0:203,0:415]

	# figure out which parts are active
	className = ''

	# get analog
	analog = img[85:198,60:165]

	buttonCount = 0
	leftStickCount = 0
	rightStickCount = 0


	# check right
	for x in range(75,100):
		if(analog[60,x][0] > 200):
			className += 'Right'
			leftStickCount += 1
			break

	# check right tilt
	# for x in range(70,80):
	# 	if(analog[60,x][0] > 200):
	# 		className += 'RightTilt'
	# 		leftStickCount += 1
	# 		break

	# check left
	for x in range(5,30):
		if(analog[60,x][0] > 200):
			className += 'Left'
			leftStickCount += 1
			break

	# check left tilt
	# for x in range(25,35):
	# 	if(analog[60,x][0] > 200):
	# 		className += 'LeftTilt'
	# 		leftStickCount += 1
	# 		break

	# check up
	for y in range(10,20):
		if(analog[y,55][0] > 200):
			className += 'Up'
			leftStickCount += 1
			break

	# check up tilt
	for y in range(30,40):
		if(analog[y,55][0] > 200):
			className += 'UpTilt'
			leftStickCount += 1
			break

	# check down
	for y in range(95,105):
		if(analog[y,55][0] > 200):
			className += 'Down'
			leftStickCount += 1
			break

	# check down tilt
	for y in range(75,85):
		if(analog[y,55][0] > 200):
			className += 'DownTilt'
			leftStickCount += 1
			break

	# check up left
	if(analog[22,20][0] > 200):
		className += 'UpLeft'
		leftStickCount += 1

	# check up right
	if(analog[22,88][0] > 200):
		className += 'UpRight'
		leftStickCount += 1

	# check down left
	if(analog[94,20][0] > 200):
		className += 'DownLeft'
		leftStickCount += 1

	# check down right
	if(analog[94,88][0] > 200):
		className += 'DownRight'
		leftStickCount += 1

	# get cStick
	cStick = img[85:198,323:428]

	# check right
	for x in range(90,100):
		if(cStick[60,x][0] < 15 and cStick[60,x][1] > 200):
			className += 'RightSmash'
			rightStickCount += 1
			break

	# check left
	for x in range(5,15):
		if(cStick[60,x][0] < 15 and cStick[60,x][1] > 200):
			className += 'LeftSmash'
			rightStickCount += 1
			break

	# check up
	for y in range(10,20):
		if(cStick[y,55][0] < 15 and cStick[y,55][1] > 200):
			className += 'UpSmash'
			rightStickCount += 1
			break

	# check down
	for y in range(95,105):
		if(cStick[y,55][0] < 15 and cStick[y,55][1] > 200):
			className += 'DownSmash'
			rightStickCount += 1
			break

	# get A button
	# off = [ 11 193   6]
	aBtn = img[138:188,242:291]
	if aBtn[30,30][0] > 252:

		# Don't allow angled A presses (i.e. DownLeftA)
		if(className in angled):
			if(len(className) <= 7):
				className = 'Up'
			else:
				className = 'Down'

		className += 'A'
		buttonCount += 1

	# get B button
	# off = [  9   0 197]
	bBtn = img[165:205,198:240]
	if bBtn[30,30][0] > 252:

		# Don't allow down angled B inputs (i.e. DownLeftB)
		if((className in angled) and (len(className) > 7)):
			className = 'Down'

		className += 'B'
		buttonCount += 1

	# get jump
	# off = [98 84 91]
	xBtn = img[138:188,296:328]
	yBtn = img[103:133,242:291]
	if xBtn[15,30][0] > 252 or yBtn[15,30][0] > 252:
		className += 'Jump'
		buttonCount += 1

	# get Z button
	# off = [201   0 101]
	zBtn = img[103:133,296:328]
	if zBtn[15,15][0] > 252:
		if('Jump' not in className):
			className += 'JumpZ'
		else:
			className += 'Z'
		buttonCount += 1

	# get shield
	# off = [113 103 102]
	lBmp = img[56:91,90:155]
	rBmp = img[56:91,242:307]
	if lBmp[20,40][0] > 252 or rBmp[20,40][0] > 252:
		className += 'Shield'
		buttonCount += 1

	if className == '':
		className = 'null'

	if className[:9] == "UpUpTilt":
		print("Replacing " + className + " with " + "Up" + className[9:])
		className = "Up" + className[9:]

	# if((buttonCount > 2) or (leftStickCount > 1) or (rightStickCount > 1) or (buttonCount + leftStickCount + rightStickCount > 3)):
	if(className not in moveLst):
		print("Class Name Error for " + str(fileName) + ": " + str(className))
		className = 'null'

	# labels.append(className)

	# Write the label to the labels file
	f.write(className + "\n")

	if className in moves:
		moves[className] += 1
	else:
		moves[className] = 1

# Close labels file
f.close()

# Print each documented move and its number of occurrences
for move, count in moves.items():
	print(move + ": " + str(count))




