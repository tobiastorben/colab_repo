import os

dir = "C:/Users/t_tor/Unsynced/complete_dataset/jpeg/"

for filename in os.listdir(dir):
	fixed = filename.replace('.jpg', '.jpeg')
	os.rename(dir+filename, dir + fixed)