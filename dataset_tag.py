import glob
import cv2
import time
sort = 0    #data type
count = 0
for i in glob.glob("cap/*.jpg"):										#path to source and get all .jpg file name
	k = cv2.waitKey(1)
	img = cv2.imread(i)
	cv2.imwrite("training_data/"+str(sort)+"_"+str(count)+".jpg",img)	#path to destination and rename as tag
	count += 1
