
# evaluate the deep model on the test dataset
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import time
 
# load train and test dataset
def load_dataset():
	# load dataset
	X = np.load('Out/prima75scoreboard/testdata.npz')
	Y = np.load('Out/prima75scoreboard/testlabels.npz')
	testX = X['data']
	testY = Y['data']
	# reshape dataset to have a single channel
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	testY = to_categorical(testY)
	return testX, testY
 
# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	testX, testY = load_dataset()
	# load model
	model = load_model('Out/prima75scoreboard/final_model.h5')
	# evaluate model on test dataset
	_, acc = model.evaluate(testX, testY, verbose=1)
	print('> %.3f' % (acc * 100.0))
      
def main():
    start_time = time.time()
    run_test_harness()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()