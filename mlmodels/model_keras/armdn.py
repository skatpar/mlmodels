import pandas as pd
import os
import numpy as np
import math
import tensorflow as tf
import keras.regularizers as reg
import matplotlib.pyplot as plt
import mdn
from keras.models import Sequential
from keras import Model
from keras import layers
from keras.layers import Dense, Dropout, Input, LSTM, Concatenate, Layer
from keras.callbacks import History, EarlyStopping
from keras.models import model_from_json
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as Keras
from mlmodels.util import save_keras, load_keras


# Less Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
VERBOSE = False


class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        lstm_h_list = model_pars["lstm_h_list"]
        OUTPUT_DIMS = model_pars["timesteps"]
        N_MIXES = model_pars["n_mixes"]
        learning_rate = compute_pars["learning_rate"]
        dense_neuron = model_pars["dense_neuron"]
        timesteps = model_pars["timesteps"]
        last_lstm_neuron = model_pars["last_lstm_neuron"]
        print("creating model....")
        model = Sequential()
        for ind, hidden in enumerate(lstm_h_list):
            model.add(LSTM(units=hidden, return_sequences=True,
                           name=f"LSTM_{ind+1}",
                           input_shape=(timesteps, 1),
                           recurrent_regularizer=reg.l1_l2(l1=0.01, l2=0.01)))
        model.add(LSTM(units=last_lstm_neuron, return_sequences=False,
                       name=f"LSTM_{len(lstm_h_list) + 1}",
                       input_shape=(timesteps, 1),
                       recurrent_regularizer=reg.l1_l2(l1=0.01, l2=0.01)))
        model.add(Dense(dense_neuron, input_shape=(-1, lstm_h_list[-1]),
                        activation='relu'))
        model.add(mdn.MDN(OUTPUT_DIMS, N_MIXES))
        adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                    decay=0.0)
        model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES),
                      optimizer=adam)
        self.model = model
        print("Model created, use model.summary() for model sructure")


def get_dataset(data_pars=None, **kw):
    path = os.path.abspath(os.path.dirname(__file__)) \
           + "/.." + data_pars["path"]
    df = pd.read_csv(path)
    return df


def os_package_root_path(filepath, sublevel=0, path_add=""):
    """
       get the module package root folder
    """
    from pathlib import Path
    path = Path(filepath).parent
    for i in range(1, sublevel + 1):
        path = path.parent

    path = os.path.join(path.absolute(), path_add)
    return path


def save(model=None, session=None, save_pars={}):
    path = save_pars["outpath"]
    os.makedirs(path, exist_ok=True)
    log("### Model saved at ######")
    log(path)
    model.model.save(path + "/armdn.h5")


def load(load_pars={}):
    print(load_pars)
    model0 = load_keras(load_pars['path'])
    model = Model()
    model.model = model0
    session = None
    return model, session


def get_params(choice=0, data_path="dataset/", **kw):
    if choice == 0:
        log("#### Path params   ##########################################")
        data_path = os_package_root_path(__file__, sublevel=1,
                                         path_add=data_path)
        out_path = os.getcwd() + "/keras_armdn/"
        os.makedirs(out_path, exist_ok=True)
        log(data_path, out_path)
        train_data_path = data_path + "timeseries/milk.csv"
        data_pars = {"train_data_path": train_data_path,
                     "train": False,
                     "prediction_length": 12,
                     "save_fig": "./series.png"}
        log("#### Model params ####")
        model_pars = {"lstm_h_list": [300, 200, 24], "last_lstm_neuron": 12,
                      "timesteps": 12, "dropout_rate": 0.1, "n_mixes": 3,
                      "dense_neuron": 10,
                      }
        compute_pars = {"batch_size": 32, "clip_gradient": 100, "ctx": None,
                        "epochs": 5000, "learning_rate": 0.05,
                        "patience": 50
                        }
        outpath = out_path + "result"
        out_pars = {"outpath": outpath}
    return model_pars, data_pars, compute_pars, out_pars


def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)


def get_dataset(data_params):
    df = pd.read_csv(data_params["train_data_path"])
    x_train = df["milk_production_pounds"].iloc[:-12]
    x_train = x_train.values.reshape(-1,
                                     data_params["prediction_length"],
                                     1)
    y_train = df["milk_production_pounds"].iloc[:-12].shift().fillna(0)
    y_train = y_train.values.reshape(-1, data_params["prediction_length"],
                                     1)
    x_test = df.iloc[-12:]["milk_production_pounds"]
    x_test = x_test.values.reshape(-1,
                                   data_params["prediction_length"],
                                   1)
    y_test = df.iloc[-12:]["milk_production_pounds"].shift().fillna(0)
    y_test = y_test.values.reshape(-1,
                                   data_params["prediction_length"],
                                   1)
    return x_train, y_train, x_test, y_test


def fit(model=None, data_pars={}, compute_pars={}, out_pars={},   **kw):
    """
    """
    batch_size = compute_pars['batch_size']
    epochs = compute_pars['epochs']
    patience = compute_pars["patience"]

    sess = None
    log("#### Loading dataset   #############################################")
    x_train, y_train, x_test, y_test = get_dataset(data_pars)

    early_stopping = EarlyStopping(monitor='loss', patience=patience,
                                   mode='min')
    model.model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[early_stopping]
                    )
    return model


def predict(model=None, model_pars=None, data_pars=None, **kwargs):
    pred = model.model.predict(kwargs["x_test"])

    y_samples = np.apply_along_axis(mdn.sample_from_output, 1, pred,
                                    data_pars["prediction_length"],
                                    model_pars["n_mixes"], temp=1.0)
    return y_samples.reshape(-1, 1)


def test(data_path="dataset/"):
    log("#### Loading params   ##############################################")
    model_pars, data_pars,
    compute_pars, out_pars = get_params(choice=0, data_path=ata_path)
    log("#### Model init, fit   #############################################")
    model = Model(model_pars=model_pars, data_pars=data_pars,
                  compute_pars=compute_pars)
    log("### Model created ###")
    log(model.model.summary())
    fit(model=model, data_pars=data_pars, compute_pars=compute_pars)
    # for prediction
    df = pd.read_csv(data_pars["train_data_path"])
    x_test = df.iloc[-12:]["milk_production_pounds"]
    x_test = x_test.values.reshape(-1, data_pars["prediction_length"], 1)
    y_test = df.iloc[-12:]["milk_production_pounds"].shift()
    y_test = y_test.values.reshape(-1, data_pars["prediction_length"], 1)

    log("#### Predict   ####")
    y_pred = predict(model=model, model_pars=model_pars,
                     data_pars=data_pars, x_test=x_test)

    plt.plot(y_test.reshape(-1, 1), "blue", label="actual", alpha=0.7)
    plt.plot(y_pred, "red", label="predicted", alpha=0.7)
    plt.xlabel("Month")
    plt.xlabel("milk demand")
    plt.legend(loc="upper left")
    plt.savefig(out_pars["outpath"] + "/armdn_out.png", dpi=100)
    save(model=model, session=None, save_pars=out_pars)


if __name__ == "__main__":
    VERBOSE = True
    test()
