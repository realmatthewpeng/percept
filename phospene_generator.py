import io
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulse2percept as p2p
from PIL import Image


def generate_percept():
    train_df = pd.read_parquet("datasets/MNIST/train.parquet")
    logging.info(f"train_df shape = {train_df.shape}")

    implant = p2p.implants.PRIMA75()
    model = p2p.models.ScoreboardModel(xrange=(-2.5, 2), yrange=(-2, 2), rho=20, xystep=0.05)
    model.build()

    processed = []

    for i in range(10):
        image = Image.open(io.BytesIO(train_df["image"].iloc[0]['bytes']))
        img_array = np.array(image)
        stim = p2p.stimuli.ImageStimulus(img_array)

        implant.stim = stim.resize(implant.shape)
        percept = model.predict_percept(implant)

        frame = percept.max(axis='frames')
        processed.append(frame)

    print(len(processed))
    print((processed[0] == processed[1]).all())

def main():
    start_time = time.time()
    generate_percept()
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()