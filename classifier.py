from __future__ import print_function
import cv2
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import numpy as np
import time
import glob

train = 1
sort = 2

def add_layer(inputs, in_size, out_size, activation_function=None,):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b,)
	return outputs
    
def compute_accuracy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict={xs: v_xs})
	correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
	return result,y_pre

def save():
	saver = tf.train.Saver()
	saver.save(sess, '/classmodel', write_meta_graph=False)

def restore():
	saver = tf.train.Saver()
	saver.restore(sess, '/classmodel')

xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 2])


prediction = add_layer(xs, 784, 2,  activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))       # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.compat.v1.Session()

if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
	init = tf.initialize_all_variables()
else:
	init = tf.global_variables_initializer()
sess.run(init)

if train:
	count = 0
	dataset_x = np.array([100,784])
	dataset_y = np.array([100,1])
	for i in glob.glob("training_data/*.jpg"):
		img = cv2.imread(i)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		dataset_x[count] = np.reshape(gray,(1,784))
		dataset_y[count] = i[14:15]
		count += 1
		
	for i in range (10):
		batch_x[i] = np.vsplit(arr, 10)
		batch_y[i] = np.vsplit(arr, 10)
		sess.run(train_step, feed_dict={xs: batch_x[i], ys: batch_y[i]})
		print(compute_accuracy(dataset_x, dataset_y))
	save()
else:
	restore()
	cap = cv2.VideoCapture(0)
	while True:
		start = time.time()
		_, frame = cap.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (3, 3), 0)
		canny = cv2.Canny(blurred, 120,200)
		
		_,contours,_ = cv2.findContours(canny, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		for contour in contours:
			area = cv2.contourArea(contour)
			if(area>200):
				x,y,w,h = cv2.boundingRect(contour)
				#canny = cv2.rectangle(canny, (x,y),(x+w,y+h),255,2)
				cut_img = gray[y:y+h, x:x+w]
				resize_img = cv2.resize(cut_img,(28,28))
				x = np.reshape(resize_img,(1,784))
				result = sess.run(prediction, feed_dict={xs: x})
		if time.time()-start > 1/60:
			time.sleep(0.001)
		
	cap.release()
	videoframe.release()
	cv2.destroyAllWindows()
