# -*- coding: utf-8 -*-
"""
Gluon

"""
import os
import pandas as pd

from mlmodels.util import env_pip_check
#env_pip_check({"requirement": "pip/requirements_tabular.txt",
#               "import":   ["gluonts", "mxnet"] })


from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer


from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor

import matplotlib.pyplot as plt
import json




from mlmodels.model_gluon.util import (
    _config_process, fit, get_dataset, load, metrics,
    plot_predict, plot_prob_forecasts, predict, save)


from mlmodels.util import os_package_root_path, path_norm


VERBOSE = False





########################################################################################################################
#### Model defintion
class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, **kwargs):
        ## Empty model for Seaialization
        if model_pars is None and compute_pars is None:
            self.model = None

        else:
            self.compute_pars = compute_pars
            self.model_pars = model_pars

            m = self.compute_pars
            trainer = Trainer(batch_size=m['batch_size'], clip_gradient=m['clip_gradient'], ctx=m["ctx"],
                              epochs=m["epochs"],
                              learning_rate=m["learning_rate"], init=m['init'],
                              learning_rate_decay_factor=m['learning_rate_decay_factor'],
                              minimum_learning_rate=m['minimum_learning_rate'], hybridize=m["hybridize"],
                              num_batches_per_epoch=m["num_batches_per_epoch"],
                              patience=m['patience'], weight_decay=m['weight_decay']
                              )

            ##set up the model
            m = self.model_pars
            self.model = DeepAREstimator(prediction_length=m['prediction_length'], freq=m['freq'],
                                         num_layers=m['num_layers'],
                                         num_cells=m["num_cells"],
                                         cell_type=m["cell_type"], dropout_rate=m["dropout_rate"],
                                         use_feat_dynamic_real=m["use_feat_dynamic_real"],
                                         use_feat_static_cat=m['use_feat_static_cat'],
                                         use_feat_static_real=m['use_feat_static_real'],
                                         scaling=m['scaling'], num_parallel_samples=m['num_parallel_samples'],
                                         trainer=trainer)







########################################################################################################################
def get_params(choice="", data_path="dataset/timeseries/", config_mode="test", **kw):
    

    if choice == "json":
        return _config_process(data_path, config_mode=config_mode)


    if choice == "test01" :
        log("#### Path params   ###################################################")
        data_path  = path_norm( "dataset/timeseries" )   
        out_path   = path_norm( "ztest/model_gluon/gluon_deepar/" )   
        model_path = os.path.join(out_path , "model")


        data_pars = {"train_data_path": data_path + "/GLUON-train.csv" , 
                     "test_data_path":  data_path + "/GLUON-test.csv" , 
                     "train": False,
                     'prediction_length': 48, 'freq': '1H', 
                     "start": pd.Timestamp("01-01-1750", freq='1H'), 
                     "num_series": 37,
                     "save_fig": "./series.png", "modelpath": model_path}
        
        log("#### Model params   ################################################")
        model_pars = {"prediction_length": data_pars["prediction_length"], "freq": data_pars["freq"],
                      "num_layers": 2, "num_cells": 40, "cell_type": 'lstm', "dropout_rate": 0.1,
                      "use_feat_dynamic_real": False, "use_feat_static_cat": False, "use_feat_static_real": False,
                      "scaling": True, "num_parallel_samples": 100}


        compute_pars = {"batch_size": 32, "clip_gradient": 100, "ctx": None, "epochs": 1, "init": "xavier",
                        "learning_rate": 1e-3,
                        "learning_rate_decay_factor": 0.5, "hybridize": False, "num_batches_per_epoch": 10,
                        'num_samples': 100,
                        "minimum_learning_rate": 5e-05, "patience": 10, "weight_decay": 1e-08}

        out_pars = {"outpath": out_path + "result", 
                    "plot_prob": True, "quantiles": [0.1, 0.5, 0.9]}

    return model_pars, data_pars, compute_pars, out_pars







########################################################################################################################
def test(data_path="dataset/", choice=""):
    ### Local test

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=choice, data_path=data_path)
    print(model_pars, data_pars, compute_pars, out_pars)

    log("#### Loading dataset   #############################################")
    gluont_ds = get_dataset(data_pars)

    log("#### Model init, fit   #############################################")
    from mlmodels.models import module_load_full, fit, predict
    module, model = module_load_full("model_gluon.gluon_deepar", model_pars, data_pars, compute_pars)
    print(module, model)

    # model=m.model    ### WE WORK WITH THE CLASS (not the attribute GLUON )
    model = fit(module, model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)

    log("#### save the trained model  ######################################")
    save(model, data_pars["modelpath"])

    log("#### Predict   ####################################################")
    ypred = predict(module, model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    # print(ypred)

    log("#### metrics   ####################################################")
    metrics_val, item_metrics = metrics(ypred, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   #######################################################")
    plot_prob_forecasts(ypred, out_pars)
    plot_predict(item_metrics, out_pars)


if __name__ == '__main__':
    VERBOSE = True
    test(data_path="dataset/timeseries", choice="test01")



    from mlmodels.models import test_module
    # test(data_path="dataset/", choice="json")




