inFile = "C:/Users/t_tor/Unsynced/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/boat_test.txt"
outFile =  "C:/Users/t_tor/Unsynced/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/boat_test_fixed.txt"


f1 = open(inFile, 'r')
f2 = open(outFile, 'w')
for line in f1:
	if not '-1' in line:
		string = line.split(' ')[0]	
		f2.write(string + '\n')
f2.close()
f1.close()
		