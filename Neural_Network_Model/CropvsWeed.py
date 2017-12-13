
#Pre-processing

import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = '/media/holy_cow/Penseive/ML_CropvsWeed/train'
#TEST_DIR = ''
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'CropvsWeed-{}-{}.model'.format(LR,'6ConvBasic10epochs')

def label_img(img):
	word_label = img.split('.')[-3]
	if word_label == 'crop' : return [1,0]
	elif word_label == 'weed': return [0,1]

def create_train_data():
	training_data = []
	for img in tqdm(os.listdir(TRAIN_DIR)):
		try:
			label = label_img(img)
			path = os.path.join(TRAIN_DIR,img)
			img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
			training_data.append([np.array(img), np.array(label)])

		except Exception as e:
			print(str(e)) 

	shuffle(training_data)
	np.save('train_data.npy', training_data)
	return training_data

#Processing
def process_test_data():
	testing_data = []
	for img in tqdm(os.listdir(TEST_DIR)):
		try:
			path = os.path.join(TEST_DIR, img)
			img_num = img.split('.')[0]
			img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
			testing_data.append([np.array(img), img_num])

		except Exception as e:
			print(str(e)) 

	np.save('test_data.npy',testing_data)
	return testing_data

train_data = create_train_data()
#if already done training,
#train_data = np.load('train_data.npy')

#NeuralNetwork
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf

tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('Yay, Model loaded..!!')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

#Training
model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#tensorboard --logdir=/media/holy_cow/Penseive/ML_CropvsWeed/Neural_Network_Model/log

model.save(MODEL_NAME)

'''

#Predicting/Output
import matplotlib.pyplot as plt

test_data = process_test_data()
#if already done,
#test_data = np.load('test_data.npy')

fig = plt.figure()

for num, data in enumerate(test_data[:12]):
	img_num = data[1]
	img_data = data[0]

	y = fig.add_subplot(3,4,num+1)
	orig = img_data
	data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)

	model_out = model.predict([data])[0]

	if np.argmax(model_out) == 1: str_label='crop'
	else: str_label='weed'

	y.imshow(orig, cmap='gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)

plt.show()


'''

#To export result as csv
'''

with open('submission_file.csv','w') as f:
    f.write('id,label\n')
            
with open('submission_file.csv','a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num,model_out[1]))

'''


