import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor

VERBOSE = False



from mlmodels.util import os_package_root_path, log


####################################################################################################
def _config_process(data_path, config_mode="test"):
    data_path = Path(os.path.realpath(
        __file__)).parent.parent / "model_gluon/gluon_deepar.json" if data_path == "dataset/" else data_path

    with open(data_path, encoding='utf-8') as config_f:
        config = json.load(config_f)
        config = config[config_mode]

    return config["model_pars"], config["data_pars"], config["compute_pars"], config["out_pars"]





def metrics(ypred, data_pars, compute_pars=None, out_pars=None, **kwargs):
    ## load test dataset
    data_pars['train'] = False
    test_ds = get_dataset(data_pars)

    forecasts = ypred["forecasts"]
    tss = ypred["tss"]

    ## evaluate
    evaluator = Evaluator(quantiles=out_pars['quantiles'])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
    metrics_dict = json.dumps(agg_metrics, indent=4)
    return metrics_dict, item_metrics


###############################################################################################################
### different plots and output metric
def plot_prob_forecasts(ypred, out_pars=None):
    forecast_entry = ypred["forecasts"][0]
    ts_entry = ypred["tss"][0]

    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()


def plot_predict(item_metrics, out_pars=None):
    item_metrics.plot(x='MSIS', y='MASE', kind='scatter')
    plt.grid(which="both")
    outpath = out_pars['outpath']
    plt.savefig(outpath)
    plt.clf()
    print('Saved image to {}.'.format(outpath))



###############################################################################################################
# save and load model helper function
class Model_empty(object):
    def __init__(self, model_pars=None, compute_pars=None):
        ## Empty model for Seaialization
        self.model = None


def save(model, path):
    if os.path.exists(path):
        model.model.serialize(Path(path))


def load(path):
    if os.path.exists(path):
        predictor_deserialized = Predictor.deserialize(Path(path))

    model = Model_empty()
    model.model = predictor_deserialized
    #### Add back the model parameters...

    return model




