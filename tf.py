import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import *
from torch.optim import *
from torchvision.models import *
from sklearn.model_selection import *
from sklearn.metrics import f1_score,accuracy_score,precision_score
import wandb
import nltk
from nltk.stem.porter import *
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn import svm
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import warnings
import os
warnings.filterwarnings("ignore")
PROJECT_NAME = "Natural-Language-Processing-with-Disaster-Tweets"
np.random.seed(55)
stemmer = PorterStemmer()
device = "cuda"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

class TensorFlow_Data_Loader:
    def __init__(
        self,
        data: pd.DataFrame = pd.read_csv("./Modelling/dataset/data/train.csv"),
        test: pd.DataFrame = pd.read_csv("./Modelling/dataset/data/test.csv"),
    ):
        self.data = data
        self.data = self.data.sample(frac=1)
        self.test = test
        self.X = np.array(self.data["keyword"])
        self.y = np.array(self.data["target"])
        print(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.0625, shuffle=True
        )

    def create(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def create_test(self):
        return np.array(self.test["keyword"])

    def create_submission(self, model):
        preds = model.predict(self.create_test())
        print(preds)
        ids = self.test["id"]
        submission = {"id": [], "target": []}
        for pred, id in tqdm(zip(preds, ids)):
            submission["id"].append(id)
            submission["target"].append(int(pred))
        submission = pd.DataFrame(submission)
        return submission

class TensorFlow_Modelling:
    def train(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        name="BaseLine",
        model="https://tfhub.dev/google/nnlm-en-dim50/2",
        trainable=True,
        tl_model_output=16,
        output=1,
        activation="relu",
        optimizer="adam",
        criterion=tf.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name="accuracy")],
    ):
        wandb.init(project=PROJECT_NAME, name=name, sync_tensorboard=True)
        hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=trainable)
        model = tf.keras.Sequential()
        model.add(hub_layer)
        model.add(tf.keras.layers.Dense(tl_model_output, activation=activation))
        model.add(tf.keras.layers.Dense(output))
        model.compile(
            optimizer=optimizer,
            loss=criterion,
            metrics=metrics,
        )
        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1,
        )
        wandb.log({
            "Accuracy":model.evaluate(X_train, y_train)[0],
            "Loss":model.evaluate(X_train, y_train)[1],
             "Val Accuracy":model.evaluate(X_test, y_test)[0],
            "Val Loss":model.evaluate(X_test, y_test)[1],
        })
        return model

models = [
    "https://tfhub.dev/google/nnlm-en-dim50/2",
    "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2",
    "https://tfhub.dev/digitalepidemiologylab/covid-twitter-bert/2",
    "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3",
    "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1",
]
models = ["https://tfhub.dev/google/nnlm-en-dim50/2"]
for model in models:
    tdl = TensorFlow_Data_Loader()
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = tdl.create()
    tm = TensorFlow_Modelling()
    model = tm.train(X_train, X_test, y_train, y_test, model=model)
    submission = tdl.create_submission(model)
    submission.to_csv(f"./save/{model}-TensorFlow-BaseLine.csv", index=False)
