# Documentation
List of Functions/Methods

https://github.com/arita37/mlmodels/blob/dev/README_index_doc.py

# 
# 
___________________________________________________________________________________________
# VScode Online Editor (pre-installed)

https://github.com/arita37/mlmodels/issues/101


# 
# 
___________________________________________________________________________________________
# Steps for: Pull Request, A Fix or Adding a New Model:


### Coding Style
   - Please don't exceed 100 characters per line.
   - "Pretty Code is better than strict PEP8, which is ugly to read...". [Read More](https://realpython.com/python-pep8/)
   - Please use [Black](https://github.com/psf/black) formatter Python package.



### 0) Read the examples :

  [Issue#102](https://github.com/arita37/mlmodels/issues/102)
  [Issue#100](https://github.com/arita37/mlmodels/pull/100)



    
### 1) Fork from arita37/mlmodels. Create a new branch from DEV branch (modelXXX, e.g., model_deepar) to work on.
    git checkout -b myfeat


### 2) TEST : Change in these files where needed with your MODEL_NAME and BRANCH NAME :

  [test_specific_model](https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_specific_model.yml)

  [pullrequest.json] (https://github.com/arita37/mlmodels/blob/dev/pullrequest.json)


### 3) TEST : After each commit or PullRequest, automatic test is run on Github :
  Please check here :
  https://github.com/arita37/mlmodels/actions



### 4) Create  mlmodels/model_XXXX/yyyyy.py   
  Template
  https://github.com/arita37/mlmodels/blob/dev/mlmodels/template/model_xxx.py

  Example :
  https://github.com/arita37/mlmodels/blob/dev/mlmodels/model_keras/textcnn.py  
  https://github.com/arita37/mlmodels/blob/dev/mlmodels/model_tch/transformer_sentence.py


  Please re-use existing functions
  https://github.com/arita37/mlmodels/blob/dev/mlmodels/util.py
  
     from mlmodels.util import os_package_root_path, log, 
                            path_norm, get_model_uri, path_norm_dict

     ### Use path_norm to normalize your path.
     data_path = path_norm("dataset/text/myfile.txt")
        --> FULL_ PATH   /home/ubuntu/mlmodels/dataset/text/myfile.txt


     ### Use path_norm to normalize your path.
     data_path = path_norm("ztest/text/myfile.txt")
        --> FULL_ PATH   /home/ubuntu/mlmodels/ztest/text/myfile.txt


     data_path = path_norm("ztest/text/myfile.txt")
        --> FULL_ PATH   /home/ubuntu/mlmodels/ztest/text/myfile.txt




### 4) Create  mlmodels/model_XXXX/yyyy.json , following this template :
  
  https://github.com/arita37/mlmodels/blob/dev/mlmodels/template/models_config.json


### 5)  Pull Request  arita37/mlmodels DEV ---> YourBranch  (or git pull --all )
   You need to MERGE with recent changes in dev INTO your Branch to reduce conflicts at final steps.



### 6) Run/Test on your local machine
    source activate py36
    cd mlmodels
    python model_XXXX/yyyy.py  



## 7)  Pull Request  arita37/mlmodels DEV ---> YourBranch  (or git pull --all )
   You need to MERGE with recent changes in dev INTO your Branch to reduce conflicts at final steps.




### 8) Check on Github your test runs
https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_custom_model




### 9)  Send Pull Request






# 
# 
# ___________________________________________________________________________________________
# Manual Installation
    ### On Linux/MacOS
    pip install numpy<=1.17.0
    pip install -e .  -r requirements.txt
    pip install   -r requirements_fake.txt


    ### On Windows
    VC 14   https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2019
    pip install numpy<=1.17.0
    pip install torch==1..1 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -e .  -r requirements_wi.txt
    pip install   -r requirements_fake.txt


    ### No Deps
    # pip install -e .  --no-deps




___________________________________________________________________________________________
# How to add a new model
### Source code structure as below
- `docs`: documentation
- `mlmodels`: interface wrapper for pytorch, keras, gluon, tf, transformer NLP for train, hyper-params searchi.
    + `model_xxx`: folders for each platform with same interface defined in template folder
    + `dataset`: store dataset files for test runs.
    + `template`: template interface wrapper which define common interfaces for whole platforms
    + `ztest`: testing output for each sample testing in `model_xxx`
- `ztest`: testing output for each sample testing in `model_xxx`

###  How to define a custom model
#### 1. Create a file `mlmodels\model_XXXX\mymodel.py` , XXX: tch: pytorch, tf:tensorflow, keras:keras, .... 
- Declare below classes/functions in the created file:

      Class Model()                                                  :   Model definition
            __init__(model_pars, data_pars, compute_pars)            :   
                                  
      def fit(model, data_pars, model_pars, compute_pars, out_pars ) : Train the model
      def fit_metric(model, data_pars, compute_pars, out_pars )         : Measure the results
      def predict(model, sess, data_pars, compute_pars, out_pars )   : Predict the results


      def get_params(choice, data_path, config_mode)                                               : returnparameters of the model
      def get_dataset(data_pars)                                     : load dataset
      def test()                                                     : example running the model     
      def test_api()                                                 : example running the model in global settings  

      def save(model, session, save_pars)                            : save the model
      def load(load_pars)                                            : load the trained model


- *Infos* 
     ```
     model :         Model(model_pars), instance of Model() object
     sess  :         Session for TF model  or optimizer in PyTorch
     model_pars :    dict containing info on model definition.
     data_pars :     dict containing info on input data.
     compute_pars :  dict containing info on model compute.
     out_pars :      dict containing info on output folder.
     save_pars/load_pars : dict for saving or loading a model
     ```

#### 2. Write your code and create test() to test your code.  **
- Declare model definition in Class Model()
```python
    self.model = DeepFM(linear_cols, dnn_cols, task=compute_pars['task']) # mlmodels/model_kera/01_deectr.py
    # Model Parameters such as `linear_cols, dnn_cols` is obtained from function `get_params` which return `model_pars, data_pars, compute_pars, out_pars`
```        
- Implement pre-process data in function `get_dataset` which return data for both training and testing dataset
Depend on type of dataset, we could separate function with datatype as below example
```python    
    if data_type == "criteo":
        df, linear_cols, dnn_cols, train, test, target = _preprocess_criteo(df, **kw)

    elif data_type == "movie_len":
        df, linear_cols, dnn_cols, train, test, target = _preprocess_movielens(df, **kw)
```
- Call fit/predict with initialized model and dataset
```python
    # get dataset using function get_dataset
    data, linear_cols, dnn_cols, train, test, target = get_dataset(**data_pars)
    # fit data
     model.model.fit(train_model_input, train[target].values,
                        batch_size=m['batch_size'], epochs=m['epochs'], verbose=2,
                        validation_split=m['validation_split'], )
    # predict data
    pred_ans = model.model.predict(test_model_input, batch_size= compute_pars['batch_size'])
```
- Calculate metric with predict output
```python
    # input of metrics is predicted output and ground truth data
    def metrics(ypred, ytrue, data_pars, compute_pars=None, out_pars=None, **kwargs):
```
- *Example* 
    https://github.com/arita37/mlmodels/tree/dev/mlmodels/template
    https://github.com/arita37/mlmodels/blob/dev/mlmodels/model_gluon/gluon_deepar.py
    https://github.com/arita37/mlmodels/blob/dev/mlmodels/model_gluon/gluon_deepar.json


#### 3. Create JSON config file inside  /model_XXX/mymodel.json  **
- Separate configure for staging development environment such as testing and production phase
then for each staging, declare some specific parameters for model, dataset and also output
- *Example*
```json
    {
        "test": {

              "hypermodel_pars":   {
             "learning_rate": {"type": "log_uniform", "init": 0.01,  "range" : [0.001, 0.1] },
             "num_layers":    {"type": "int", "init": 2,  "range" :[2, 4] },
             "size":    {"type": "int", "init": 6,  "range" :[6, 6] },
             "output_size":    {"type": "int", "init": 6,  "range" : [6, 6] },

             "size_layer":    {"type" : "categorical", "value": [128, 256 ] },
             "timestep":      {"type" : "categorical", "value": [5] },
             "epoch":         {"type" : "categorical", "value": [2] }
           },

            "model_pars": {
                "learning_rate": 0.001,     
                "num_layers": 1,
                "size": 6,
                "size_layer": 128,
                "output_size": 6,
                "timestep": 4,
                "epoch": 2
            },

            "data_pars" :{
              "path"            : 
              "location_type"   :  "local/absolute/web",
              "data_type"   :   "text" / "recommender"  / "timeseries" /"image",
              "data_loader" :  "pandas",
              "data_preprocessor" : "mlmodels.model_keras.prepocess:process",
              "size" : [0,1,2],
              "output_size": [0, 6]              
            },


            "compute_pars": {
                "distributed": "mpi",
                "epoch": 10
            },
            "out_pars": {
                "out_path": "dataset/",
                "data_type": "pandas",
                "size": [0, 0, 6],
                "output_size": [0, 6]
            }
        },
    
        "prod": {
            "model_pars": {},
            "data_pars": {}
        }
    }
```


 
#######################################################################################

## ③ CLI tools: package provide below tools
https://github.com/arita37/mlmodels/blob/dev/README_usage.md

```
- ml_models    :  mlmodels/models.py
- ml_optim     :  mlmodels/optim.py
- ml_test      :  mlmodels/ztest.py



```
   





#######################################################################################
### ④ Interface

models.py 
```
   module_load(model_uri)
   model_create(module)
   fit(model, module, session, data_pars, out_pars   )
   metrics(model, module, session, data_pars, out_pars)
   predict(model, module, session, data_pars, out_pars)
   save(model, path)
   load(model)
```

optim.py
```
   optim(modelname="model_tf.1_lstm.py",  model_pars= {}, data_pars = {}, compute_pars={"method": "normal/prune"}
       , save_folder="/mymodel/", log_folder="", ntrials=2) 

   optim_optuna(modelname="model_tf.1_lstm.py", model_pars= {}, data_pars = {}, compute_pars={"method" : "normal/prune"},
                save_folder="/mymodel/", log_folder="", ntrials=2) 
```

#### Generic parameters 
```
   Define in models_config.json
   model_params      :  Relative to model definition 
   compute_pars      :  Relative to  the compute process
   data_pars         :  Relative to the input data
   out_pars          :  Relative to outout data
```
   Sometimes, data_pars is required to setup the model (ie CNN with image size...)
   






   #######################################################################################
   ### ⑥ Naming convention

   ### Function naming
   ```
   pd_   :  input is pandas dataframe
   np_   :  input is numpy
   sk_   :  inout is related to sklearn (ie sklearn model), input is numpy array
   plot_


   col_ :  function name for column list related.
   ```


   #####################################################################################






