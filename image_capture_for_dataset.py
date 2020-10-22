import cv2
import numpy as np
import time

#cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
#for pi camera
cap = cv2.VideoCapture(0) #for usb camera or camera of laptop
time.sleep(0.1)
count = 0
px,py,pw,ph = 0,0,0,0
while True:
	_, frame = cap.read()
	cv2.imshow("read",frame)
	k = cv2.waitKey(5)
	time.sleep(0.01)
	if k == ord('q'):  #press "q" to quit
		break
	elif k == ord('f'):  #press "f" to capture image
		_, frame = cap.read()
		draw = frame.copy()
		gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
		#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		canny = cv2.Canny(gray, 120,200)

		contours,_ = cv2.findContours(canny, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #"_,contours,_" if error occcur
		for contour in contours:
			area = cv2.contourArea(contour)
			if area > 100:															#filter out the undesired object(small object) 
				x,y,w,h = cv2.boundingRect(contour)
				if abs(x-px)<10 and abs(y-py)<10 and abs(w-pw)<10 and abs(h-ph)<10: #filter out the undesired object(same object)
					pass
				else:
					pre = area
					cut_img = frame[y:y+h, x:x+w]			  			#cut detected object
					resize_img = cv2.resize(cut_img,(28,28))  			#resize image to 28*28
					cv2.imwrite("cap/"+str(count)+".jpg", resize_img)	#save the image ( cv2.imwrite( "path to your file" +str(count)+ ".jpg", resize_img)  )
					cv2.rectangle(draw, (x,y),(x+w,y+h),(0,255,0),1)
					cv2.imshow("capture",draw)							#show the detected object in another window
					px,py,pw,ph = x,y,w,h
					count += 1
		print(count)
		time.sleep(0.1)
cv2.destroyAllWindows()
cap.release()

