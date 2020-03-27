# mlmodels : Model ZOO for Pytorch, Tensorflow, Keras, Gluon, sklearn, LightGBM models...

* Model ZOO with Lightweight Functional interface to wrap access to Recent and State o Art Deep Learning, ML models and Hyper-Parameter Search, cross platforms such as Tensorflow, Pytorch, Gluon, Keras, sklearn, light-GBM,...

* Logic follows sklearn : fit, predict, transform, metrics, save, load

* Goal is to transform Jupyter/research code into Semi-Prod (batch,..) code with minimal code change ... 

* Model list is available here : 
  https://github.com/arita37/mlmodels/blob/dev/README_model_list.md

* Why Functional interface instead of OOP ?
    Functional reduces the amount of code needed, focus more on the computing part (vs design part), 
    a bit easier maintenability for medium size project, good for scientific computing process.


*  Colab demo :
https://colab.research.google.com/drive/1sYbrXNZh9nTeizS-AuCA8RSu94B_B-RF


## Model List :

### Time Series:
Nbeats: 2019, Time Series NNetwork,  https://arxiv.org/abs/1905.10437

Amazon Deep AR: 2019, Time Series NNetwork,  https://arxiv.org/abs/1905.10437

Facebook Prophet 2017, Time Series prediction,

ARMDN Advanced Time series Prediction :  2019, Associative and Recurrent Mixture Density Networks for time series.

LSTM prediction 



### NLP :

Sentence Transformers :  2019, Embedding of full sentences using BERT,  https://arxiv.org/pdf/1908.10084.pdf

Transformers Classifier : Using Transformer for Text Classification, https://arxiv.org/abs/1905.05583

TextCNN Pytorch : 2016, Text CNN Classifier, https://arxiv.org/abs/1801.06287

TextCNN Keras : 2016, Text CNN Classifier, https://arxiv.org/abs/1801.06287

charCNN Keras : Text Character Classifier,


### TABULAR :
AutoML Gluon  :  2020, AutoML in Gluon, MxNet using LightGBM, CatBoost

All sklearn models, LighGBM



### VISION :
  Vision Models (pre-trained) :  
        alexnet, densenet121, densenet169, densenet201,
        densenet161, inception_v3, resnet18, resnet34, resnet50, resnet101, resnet152,
        resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, squeezenet1_0,
        squeezenet1_1, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,
        googlenet, shufflenet_v2_x0_5, shufflenet_v2_x1_0, mobilenet_v2"


A lot more...


......


https://github.com/arita37/mlmodels/blob/dev/README_model_list.md



######################################################################################

## ① Installation
Install as editable package (ONLY dev branch), in Linux
    
    conda create -n py36 python=3.6.5  -y
    source activate py36
    
    cd yourfolder
    git clone https://github.com/arita37/mlmodels.git mlmodels
    cd mlmodels
    git checkout dev 


    ### On Linux/MacOS    
    pip install numpy<1.17.0
    pip install -e .  -r requirements.txt
    pip install   -r requirements_fake.txt


    ### On Windows 
    VC 14   https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2019
    pip install  numpy<1.17.0
    pip install torch==1.0.1 -f https://download.pytorch.org/whl/torch_stable.html  
    pip install -e .  -r requirements_wi.txt  
    pip install   -r requirements_fake.txt


    ### No Deps
    # pip install -e .  --no-deps    


    ### Check this Colab for install :
https://colab.research.google.com/drive/1sYbrXNZh9nTeizS-AuCA8RSu94B_B-RF

    ##### Initialize
    Will copy template, dataset, example to your folder

    ml_models --init  /yourworkingFolder/


    ##### To test :
    ml_optim

    ##### To test model fitting
    ml_models
    

    

  
####  Dependencies

https://github.com/arita37/mlmodels/blob/dev/requirements.txt


#### Actual test runs

https://github.com/arita37/mlmodels/actions


![test_fast_linux](https://github.com/arita37/mlmodels/workflows/test_fast_linux/badge.svg)

![test_fast_windows](https://github.com/arita37/mlmodels/workflows/test_fast_windows/badge.svg?branch=dev)


![ All model testing (Linux) ](https://github.com/arita37/mlmodels/workflows/code_structure_linux/badge.svg)



#######################################################################################
## Usage in Jupyter

https://github.com/arita37/mlmodels/blob/dev/README_usage.md



#######################################################################################
## CLI tools: 

https://github.com/arita37/mlmodels/blob/dev/README_usage_CLI.md

  ```
  - ml_models    :  mlmodels/models.py
  - ml_optim     :  mlmodels/optim.py
  - ml_test      :  mlmodels/ztest.py

  ```


####################################################################################
## Model List

https://github.com/arita37/mlmodels/blob/dev/README_model_list.md




#######################################################################################
## How to add a new model

https://github.com/arita37/mlmodels/blob/dev/README_addmodel.md


 
  
#######################################################################################
## Index of functions/methods

https://github.com/arita37/mlmodels/blob/dev/README_index_doc.txt






####################################################################################
## In Jupyter 

### LSTM example in TensorFlow ([Example notebook](example/1_lstm.ipynb))

#### Import library and functions
```python
# import library
import mlmodels
```

#### Define model and data definitions
```python
model_uri    = "model_tf.1_lstm.py"
model_pars   =  {  "num_layers": 1,
                  "size": ncol_input, "size_layer": 128, "output_size": ncol_output, "timestep": 4,
                }
data_pars    =  {"data_path": "/folder/myfile.csv"  , "data_type": "pandas" }
compute_pars =  { "learning_rate": 0.001, }

out_pars     =  { "path": "ztest_1lstm/", "model_path" : "ztest_1lstm/model/"}
save_pars = { "path" : "ztest_1lstm/model/" }
load_pars = { "path" : "ztest_1lstm/model/" }

```


#### Load Parameters and Train
```python
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model
```


#### Inference
```python
metrics_val   =  module.fit_metrics( model, sess, data_pars, compute_pars, out_pars) # get stats
ypred         = module.predict(model, sess,  data_pars, compute_pars, out_pars)     # predict pipeline


```
---

### AutoML example in Gluon ([Example notebook](example/gluon_automl.ipynb))

#### Import library and functions
```python
# import library
import mlmodels
import autogluon as ag
```

#### Define model and data definitions
```python
model_uri = "model_gluon.gluon_automl.py"
data_pars = {"train": True, "uri_type": "amazon_aws", "dt_name": "Inc"}

model_pars = {"model_type": "tabular",
              "learning_rate": ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),
              "activation": ag.space.Categorical(*tuple(["relu", "softrelu", "tanh"])),
              "layers": ag.space.Categorical(
                          *tuple([[100], [1000], [200, 100], [300, 200, 100]])),
              'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1),
              'num_boost_round': 10,
              'num_leaves': ag.space.Int(lower=26, upper=30, default=36)
             }


compute_pars = {
    "hp_tune": True,
    "num_epochs": 10,
    "time_limits": 120,
    "num_trials": 5,
    "search_strategy": "skopt"
}

out_pars = {
    "out_path": "dataset/"
}

```


#### Load Parameters and Train
```python
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, model_pars=model_pars, compute_pars=compute_pars, out_pars=out_pars)      
```


#### Inference
```python
ypred       = module.predict(model, data_pars, compute_pars, out_pars)     # predict pipeline


```

---

### RandomForest example in Scikit-learn ([Example notebook](example/sklearn.ipynb))

#### Import library and functions
```python
# import library
import mlmodels
```

#### Define model and data definitions
```python
model_uri    = "model_sklearn.sklearn.py"

model_pars   = {"model_name":  "RandomForestClassifier", "max_depth" : 4 , "random_state":0}

data_pars    = {'mode': 'test', 'path': "../mlmodels/dataset", 'data_type' : 'pandas' }

compute_pars = {'return_pred_not': False}

out_pars    = {'path' : "../ztest"}
```


#### Load Parameters and Train
```python
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model
```


#### Inference
```python
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
```

---

### TextCNN example in keras ([Example notebook](example/textcnn.ipynb))

#### Import library and functions
```python
# import library
import mlmodels
```

#### Define model and data definitions
```python
model_uri    = "model_keras.textcnn.py"

data_pars    = {"path" : "../mlmodels/dataset/text/imdb.csv", "train": 1, "maxlen":400, "max_features": 10}

model_pars   = {"maxlen":400, "max_features": 10, "embedding_dims":50}
                       
compute_pars = {"engine": "adam", "loss": "binary_crossentropy", "metrics": ["accuracy"] ,
                        "batch_size": 32, "epochs":1, 'return_pred_not':False}

out_pars     = {"path": "ztest/model_keras/textcnn/"}

```


#### Load Parameters and Train
```python
from mlmodels.models import module_load

module        =  module_load( model_uri= model_uri )                           # Load file definition
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model
```


#### Inference
```python
data_pars['train'] = 0
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
```

---

### Using json config file for input ([Example notebook](example/1_lstm_json.ipynb), [JSON file](mlmodels/dataset/json/1_lstm.json))

#### Import library and functions
```python
# import library
import mlmodels
```

#### Load model and data definitions from json
```python
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_tf.1_lstm.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
    'choice':'json',
    'config_mode':'test',
    'data_path':'../mlmodels/dataset/json/1_lstm.json'
})
```

#### Load parameters and train
```python
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model
```

#### Check inference
```python
ypred       = module.predict(model, sess=sess,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
```

---

### Using Scikit-learn's SVM for Titanic Problem from json file ([Example notebook](example/sklearn_titanic_svm.ipynb), [JSON file](mlmodels/dataset/json/sklearn_titanic_svm.json))

#### Import library and functions
```python
# import library
import mlmodels
```

#### Load model and data definitions from json
```python
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_sklearn.sklearn.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
    'choice':'json',
    'config_mode':'test',
    'data_path':'../mlmodels/dataset/json/sklearn_titanic_svm.json'
})
```


#### Load Parameters and Train
```python
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model
```


#### Inference
```python
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
ypred
```

#### Check metrics
```python
import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv('../mlmodels/dataset/tabular/titanic_train_preprocessed.csv')['Survived'].values
roc_auc_score(y, ypred)
```

---

### Using Scikit-learn's Random Forest for Titanic Problem from json file ([Example notebook](example/sklearn_titanic_randomForest.ipynb), [JSON file](mlmodels/dataset/json/sklearn_titanic_randomForest.json))

#### Import library and functions
```python
# import library
import mlmodels
```

#### Load model and data definitions from json
```python
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_sklearn.sklearn.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars={
    'choice':'json',
    'config_mode':'test',
    'data_path':'../mlmodels/dataset/json/sklearn_titanic_randomForest.json'
})
```


#### Load Parameters and Train
```python
model         =  module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)             # Create Model instance
model, sess   =  module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model
```


#### Inference
```python
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
ypred
```

#### Check metrics
```python
import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv('../mlmodels/dataset/tabular/titanic_train_preprocessed.csv')['Survived'].values
roc_auc_score(y, ypred)
```

---

### Using Autogluon for Titanic Problem from json file ([Example notebook](example/gluon_automl_titanic.ipynb), [JSON file](mlmodels/dataset/json/gluon_automl.json))

#### Import library and functions
```python
# import library
import mlmodels
```

#### Load model and data definitions from json
```python
from mlmodels.models import module_load
from mlmodels.util import load_config

model_uri    = "model_gluon.gluon_automl.py"
module        =  module_load( model_uri= model_uri )                           # Load file definition

model_pars, data_pars, compute_pars, out_pars = module.get_params(
    choice='json',
    config_mode= 'test',
    data_path= '../mlmodels/dataset/json/gluon_automl.json'
)
```


#### Load Parameters and Train
```python
model         =  module.Model(model_pars=model_pars, compute_pars=compute_pars)             # Create Model instance
model   =  module.fit(model, model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)          # fit the model
model.model.fit_summary()
```


#### Check inference
```python
ypred       = module.predict(model,  data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)     # predict pipeline
```

#### Check metrics
```python
model.model.model_performance

import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv('../mlmodels/dataset/tabular/titanic_train_preprocessed.csv')['Survived'].values
roc_auc_score(y, ypred)
```

---
---












