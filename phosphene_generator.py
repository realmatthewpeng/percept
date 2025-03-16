import io
import logging
import os
import time

import numpy as np
import pandas as pd
import pulse2percept as p2p
import torch
from PIL import Image

import image_preprocessor as ip

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)

def generate_percept(train_images: list, train_labels: list, test_images: list, test_labels: list, 
                     implant: p2p.implants.ProsthesisSystem, model: p2p.models.Model, image_preprocessor: ip.ImagePreprocessor = None,
                     outdir = "test"):

    if not os.path.exists("Out/"):
        os.makedirs("Out/")

    if not os.path.exists(f"Out/{outdir}"):
        os.makedirs(f"Out/{outdir}")

    model.build()

    train_processed = []

    for i in range(len(train_images)):
        if i % 1000 == 0: logging.debug(f"processing train image {i}")

        if type(train_images[i]) == torch.Tensor:
            train_images[i] = train_images[i].numpy().transpose((1, 2, 0))

        stim = p2p.stimuli.ImageStimulus(train_images[i])

        if image_preprocessor is not None:
            stim = image_preprocessor.process_image(stim)

        implant.stim = stim.resize(implant.shape)
        percept = model.predict_percept(implant)

        frame = percept.max(axis='frames')
        # Normalize to [0, 1]
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
        train_processed.append(frame)

    if len(train_processed) != 0:
        np.savez_compressed(f"Out/{outdir}/traindata.npz", data=train_processed)
        np.savez_compressed(f"Out/{outdir}/trainlabels.npz", data=train_labels)

    test_processed = []

    for i in range(len(test_images)):
        if i % 100 == 0: logging.debug(f"processing test image {i}")

        if type(test_images[i]) == torch.Tensor:
            test_images[i] = test_images[i].numpy().transpose((1, 2, 0))

        stim = p2p.stimuli.ImageStimulus(test_images[i])

        if image_preprocessor is not None:
            stim = image_preprocessor.process_image(stim)

        implant.stim = stim.resize(implant.shape)
        percept = model.predict_percept(implant)

        frame = percept.max(axis='frames')
        # Normalize to [0, 1]
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
        test_processed.append(frame)

    if len(test_processed) != 0:
        np.savez_compressed(f"Out/{outdir}/testdata.npz", data=test_processed)
        np.savez_compressed(f"Out/{outdir}/testlabels.npz", data=test_labels)

    return test_processed[0].shape[0], test_processed[0].shape[1]

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