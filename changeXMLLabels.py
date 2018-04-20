import fileinput
from os import listdir

parentDir = 'C:/Users/t_tor/Unsynced/CV&DL/labels/'
targetDir = parentDir + 'all/'
subdirs = ['n02858304_boat_bbox', 'n04158807_sea_boat_bbox','n04194289_ship_bbox', 'n04530566_vessel_bbox']
oldLabels = ['n02858304', 'n04158807', 'n04194289', 'n04530566']
i = 0

for dir in subdirs:
	label = oldLabels[i]
	i += 1
	path = parentDir + dir + '/'
	string = '<name>' + label + '</name>'
	for filename in listdir(path):
		if filename != '.idea':
			print(path + filename)
			f1 = open(path+filename, 'r')
			f2 = open(targetDir+filename, 'w')
			for line in f1:
				   f2.write(line.replace(string,'<name>boat</name>'))
			f1.close()
			f2.close()