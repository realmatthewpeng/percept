import time

import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

def load_train_dataset():
	X = np.load('Out/prima75scoreboard/traindata.npz')
	Y = np.load('Out/prima75scoreboard/trainlabels.npz')
	trainX = X['data']
	trainY = Y['data']
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	return trainX, trainY

def load_test_dataset():
	X = np.load('Out/prima75scoreboard/testdata.npz')
	Y = np.load('Out/prima75scoreboard/testlabels.npz')
	testX = X['data']
	testY = Y['data']
	# reshape dataset to have a single channel
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	testY = to_categorical(testY)
	return testX, testY
 
# define cnn model
def define_basic_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 
def train_model(model, trainX: np.ndarray, trainY: np.ndarray, outdir = 'test'):
	model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=2)
	model.save(f'Out/{outdir}/final_model.h5')

def eval_model(model, testX: np.ndarray, testY: np.ndarray):
	_, acc = model.evaluate(testX, testY, verbose=1)
	print('> %.3f' % (acc * 100.0))
 
def main():
	start_time = time.time()

	model = define_basic_model()
	trainX, trainY = load_train_dataset()
	testX, testY = load_test_dataset()

	train_model(model, trainX, trainY)

	trained_model = load_model('Out/prima75scoreboard/final_model.h5')

	eval_model(trained_model, testX, testY)

	end_time = time.time()
	execution_time = end_time - start_time
	print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()