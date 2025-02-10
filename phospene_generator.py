import io
import time

import numpy as np
import pandas as pd
import pulse2percept as p2p
from PIL import Image


def generate_percept():
    train_df = pd.read_parquet("datasets/MNIST/train.parquet")

    implant = p2p.implants.PRIMA75()
    model = p2p.models.ScoreboardModel(xrange=(-2, 2), yrange=(-2, 2), xystep=0.15, rho=20)
    model.build()

    processed = []
    labels = []

    for i in range(train_df.shape[0]):
        if i % 1000 == 0: print(i)
        image = Image.open(io.BytesIO(train_df["image"].iloc[i]['bytes']))
        labels.append(train_df["label"].iloc[i])
        img_array = np.array(image)
        stim = p2p.stimuli.ImageStimulus(img_array)

        implant.stim = stim.resize(implant.shape)
        percept = model.predict_percept(implant)

        frame = percept.max(axis='frames')
        processed.append(frame)

    print(len(processed))
    print(len(labels))

    np.savez_compressed("Out/traindata.npz", data=processed)
    np.savez_compressed("Out/trainlabels.npz", data=labels)

    test_df = pd.read_parquet("datasets/MNIST/test.parquet")

    processed = []
    labels = []

    for i in range(test_df.shape[0]):
        if i % 1000: print(i)
        image = Image.open(io.BytesIO(test_df["image"].iloc[i]['bytes']))
        labels.append(test_df["label"].iloc[i])
        img_array = np.array(image)
        stim = p2p.stimuli.ImageStimulus(img_array)

        implant.stim = stim.resize(implant.shape)
        percept = model.predict_percept(implant)

        frame = percept.max(axis='frames')
        processed.append(frame)

    print(len(processed))
    print(len(labels))

    np.savez_compressed("Out/testdata.npz", data=processed)
    np.savez_compressed("Out/testlabels.npz", data=labels)

def main():
    start_time = time.time()
    generate_percept()
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()