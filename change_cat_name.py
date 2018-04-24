from os import listdir
import os

origDir = "C:/Users/t_tor/Unsynced/coco/annotations_boat_old/"
targetDir =  "C:/Users/t_tor/Unsynced/coco/annotations_boat/"

if not os.path.exists(targetDir):
    os.makedirs(targetDir)

catOld = '9'
catNew = 'boat'
nameTagOld = '<name>' + catOld + '</name>'
nameTagNew = '<name>' + catNew + '</name>'
		
nFiles = len(listdir(origDir))
progress = 0
for filename in listdir(origDir):
	progress += 1
	print(str(progress) + '/' + str(nFiles))
	f1 = open(origDir+filename, 'r')
	f2 = open(targetDir+filename, 'w')
	for line in f1:
		f2.write(line.replace(nameTagOld,nameTagNew))
	f2.close()
	f1.close()
		