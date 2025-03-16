import logging
import pickle

import numpy as np
import pulse2percept as p2p
import torchvision
import yaml
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from torchvision import transforms

import image_preprocessor as ip
from models.resnet import resnet56, resnet1202
from models.vggnet import vggnet
from models.resnet2 import resnet20


def load_config(yaml_file):
    with open(yaml_file) as file:
        raw_content = yaml.load(file,Loader=yaml.FullLoader) # nested dictionary
    return raw_content

percept_model_argtypes = {
    'xrange': tuple,
    'yrange': tuple,
    'xystep': tuple
}

def get_percept_model(cfg):
    for param, type in percept_model_argtypes.items():
        if param in cfg['percept_model_args'] and cfg['percept_model_args'][param] is not None:
            if param == 'xystep':
                try:
                    cfg['percept_model_args'][param] = type(cfg['percept_model_args'][param])
                except:
                    pass
            else:
                cfg['percept_model_args'][param] = type(cfg['percept_model_args'][param])

    if cfg['percept_model'] == 'scoreboard':
        return p2p.models.ScoreboardModel(**cfg['percept_model_args'])
    elif cfg['percept_model'] == 'axon':
        return p2p.models.AxonMapModel(**cfg['percept_model_args'])
    else:
        raise NotImplementedError
    
def get_implant(cfg):
    has_args = (('implant_args' in cfg) and (cfg['implant_args'] is not None))
    if cfg['implant'] == 'PRIMA75':
        if has_args:
            return p2p.implants.PRIMA75(**cfg['implant_args'])
        else:
            return p2p.implants.PRIMA75()
    if cfg['implant'] == 'PRIMA55':
        if has_args:
            return p2p.implants.PRIMA55(**cfg['implant_args'])
        else:
            return p2p.implants.PRIMA55()
    if cfg['implant'] == 'PRIMA40':
        if has_args:
            return p2p.implants.PRIMA40(**cfg['implant_args'])
        else:
            return p2p.implants.PRIMA40()
    if cfg['implant'] == 'PRIMA':
        if has_args:
            return p2p.implants.PRIMA(**cfg['implant_args'])
        else:
            return p2p.implants.PRIMA()
    if cfg['implant'] == 'ArgusII':
        if has_args:
            return p2p.implants.ArgusII(**cfg['implant_args'])
        else:
            return p2p.implants.ArgusII()
    if cfg['implant'] == 'AlphaAMS':
        if has_args:
            return p2p.implants.AlphaAMS(**cfg['implant_args'])
        else:
            return p2p.implants.AlphaAMS()
    else:
        raise NotImplementedError

def get_dataset(cfg, test_only = False):
    if cfg['dataset'] == 'MNIST':
        return get_MNIST_dataset(test_only)
    elif cfg['dataset'] == 'Fashion':
        return get_Fashion_dataset(test_only)
    elif cfg['dataset'] == 'cifar10':
        if not test_only:
            raise NotImplementedError
        return get_cifar10_dataset()
    elif cfg['dataset'] == 'cifar100':
        if not test_only:
            raise NotImplementedError
        return get_cifar100_dataset()
    elif cfg['dataset'] == 'EMNIST':
        if not test_only:
            raise NotImplementedError
        return get_EMNIST_dataset()
    else:
        raise NotImplementedError
    
def get_EMNIST_dataset():

    with open('./datasets/EMNIST/emnist_test.pkl', 'rb') as file:
        data = pickle.load(file)

    test_images = []
    test_labels = []

    for i in range(1000):
        if i % 100 == 0: logging.debug(f"creating test image {i}")
        test_images.append(data['data'][i]/255)
        test_labels.append(data['labels'][i])

    return [], [], test_images, test_labels

def get_cifar10_dataset():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])  
    testset = torchvision.datasets.CIFAR10(root='./datasets/', train=False, download=True, transform=transform)

    test_images = []
    test_labels = []

    for i in range(1000):
        if i % 100 == 0: logging.debug(f"creating test image {i}")
        image, label = testset[i]
        test_images.append(image)
        test_labels.append(label)

    return [], [], test_images, test_labels

def get_cifar100_dataset():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])  
    testset = torchvision.datasets.CIFAR100(root='./datasets/', train=False, download=True, transform=transform)

    test_images = []
    test_labels = []

    for i in range(1000):
        if i % 100 == 0: logging.debug(f"creating test image {i}")
        image, label = testset[i]
        test_images.append(image)
        test_labels.append(label)

    return [], [], test_images, test_labels

def get_MNIST_dataset(test_only = False):
    trainset = torchvision.datasets.MNIST(root='./datasets/', train=True, download=True)

    train_images = []
    train_labels = []

    if (not test_only):
        for i in range(len(trainset)):
            if i % 1000 == 0: logging.debug(f"creating train image {i}")
            image, label = trainset[i]
            train_images.append(np.array(image)/255)
            train_labels.append(label)

    testset = torchvision.datasets.MNIST(root='./datasets/', train=False, download=True)

    test_images = []
    test_labels = []

    for i in range(len(testset)):
        if i % 1000 == 0: logging.debug(f"creating test image {i}")
        image, label = testset[i]
        test_images.append(np.array(image)/255)
        test_labels.append(label)

    return train_images, train_labels, test_images, test_labels

def get_Fashion_dataset(test_only = False):
    trainset = torchvision.datasets.FashionMNIST(root='./datasets/', train=True, download=True)

    train_images = []
    train_labels = []

    if (not test_only):
        for i in range(len(trainset)):
            if i % 1000 == 0: logging.debug(f"creating train image {i}")
            image, label = trainset[i]
            train_images.append(np.array(image)/255)
            train_labels.append(label)

    testset = torchvision.datasets.FashionMNIST(root='./datasets/', train=False, download=True)

    test_images = []
    test_labels = []

    for i in range(len(testset)):
        if i % 1000 == 0: logging.debug(f"creating test image {i}")
        image, label = testset[i]
        test_images.append(np.array(image)/255)
        test_labels.append(label)

    return train_images, train_labels, test_images, test_labels

def get_classifier(cfg):
    if cfg['classifier'] == 'basic_cnn':
        return get_basic_cnn_classifier()
    else:
        raise NotImplementedError
    
def get_basic_cnn_classifier():
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

def get_processed_dataset(test_only = False, test_X_path = 'Out/testdata.npz', test_Y_path = 'Out/testlabels.npz', xdim=28, ydim=28, outdir = 'test'):
    trainX = []
    trainY = []
    if (not test_only):
        X = np.load(f'Out/{outdir}/traindata.npz')
        Y = np.load(f'Out/{outdir}/trainlabels.npz')
        trainX = X['data']
        trainY = Y['data']
        # reshape dataset to have a single channel
        trainX = trainX.reshape((trainX.shape[0], xdim, ydim, 1))
        # one hot encode target values
        trainY = to_categorical(trainY)

    test_X_path = f'Out/{outdir}/testdata.npz'
    test_Y_path = f'Out/{outdir}/testlabels.npz'

    X = np.load(test_X_path)
    Y = np.load(test_Y_path)
    testX = X['data']
    testY = Y['data']
    # reshape dataset to have a single channel
    testX = testX.reshape((testX.shape[0], xdim, ydim, 1))
    # one hot encode target values
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY

def get_trained_classifier(cfg, outdir = 'test'):
    if cfg['classifier'] == 'basic_cnn':
        trained_model = load_model(f'Out/{outdir}/final_model.h5')
        return trained_model
    else:
        raise NotImplementedError
    
def get_pretrained_classifier(path):
    if path == 'resnet20':
        trained_model = resnet20()
        return trained_model
    if path == 'resnet56':
        trained_model = resnet56()    
        return trained_model
    if path == 'resnet1202':
        trained_model = resnet1202()    
        return trained_model
    if path == 'vggnet':
        trained_model = vggnet()
        return trained_model
    trained_model = load_model(path)
    trained_model.compile()
    return trained_model

def get_image_preprocessor(cfg):
    if cfg is not None:
        return ip.ImagePreprocessor(**cfg)
    else:
        return None