import io
import logging
import time

import numpy as np
import pandas as pd
import pulse2percept as p2p
from PIL import Image

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)

def generate_percept(train_images: list, train_labels: list, test_images: list, test_labels: list, implant: p2p.implants.ProsthesisSystem, model: p2p.models.Model):

    model.build()

    train_processed = []

    for i in range(len(train_images)):
        if i % 1000 == 0: logging.debug(f"processing train image {i}")
        stim = p2p.stimuli.ImageStimulus(train_images[i])

        implant.stim = stim.resize(implant.shape)
        percept = model.predict_percept(implant)

        frame = percept.max(axis='frames')
        train_processed.append(frame)

    np.savez_compressed("Out/traindata.npz", data=train_processed)
    np.savez_compressed("Out/trainlabels.npz", data=train_labels)


    test_processed = []

    for i in range(len(test_images)):
        if i % 1000 == 0: logging.debug(f"processing test image {i}")
        stim = p2p.stimuli.ImageStimulus(test_images[i])

        implant.stim = stim.resize(implant.shape)
        percept = model.predict_percept(implant)

        frame = percept.max(axis='frames')
        test_processed.append(frame)

    np.savez_compressed("Out/testdata.npz", data=test_processed)
    np.savez_compressed("Out/testlabels.npz", data=test_labels)

def main():
    start_time = time.time()

    train_df = pd.read_parquet("datasets/MNIST/train.parquet")
    train_images = []
    train_labels = []

    for i in range(train_df.shape[0]):
        if i % 1000 == 0: logging.debug(f"creating train image {i}")
        image = Image.open(io.BytesIO(train_df["image"].iloc[i]['bytes']))
        train_labels.append(train_df["label"].iloc[i])
        img_array = np.array(image)
        train_images.append(img_array)
    logging.debug(f"total train images: {len(train_images)}")


    test_df = pd.read_parquet("datasets/MNIST/test.parquet")
    test_images = []
    test_labels = []

    for i in range(test_df.shape[0]):
        if i % 1000 == 0: logging.debug(f"creating test image {i}")
        image = Image.open(io.BytesIO(test_df["image"].iloc[i]['bytes']))
        test_labels.append(test_df["label"].iloc[i])
        img_array = np.array(image)
        test_images.append(img_array)
    logging.debug(f"total test images: {len(test_images)}")

    implant = p2p.implants.PRIMA75(z=0)
    model = p2p.models.ScoreboardModel(xrange=(-2, 2), yrange=(-2, 2), xystep=0.15, rho=20)

    generate_percept(train_images, train_labels, test_images, test_labels, implant, model)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()