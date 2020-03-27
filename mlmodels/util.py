# -*- coding: utf-8 -*-
import os
import re

# import toml
from pathlib import Path
import json


"""
from mlmodels.util import (os_package_root_path, log, path_norm
    config_load_root, config_path_pretrained, config_path_dataset, config_set
    )
"""


####################################################################################################
class to_namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

    def get(self, key):
        return self.__dict__.get(key)


def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)


####################################################################################################
def os_folder_copy(src, dst):
    """Copy a directory structure overwriting existing files"""
    import shutil
    for root, dirs, files in os.walk(src):
        if not os.path.isdir(root):
            os.makedirs(root)

        for file in files:
            rel_path = root.replace(src, '').lstrip(os.sep)
            dest_path = os.path.join(dst, rel_path)

            if not os.path.isdir(dest_path):
                os.makedirs(dest_path)

            shutil.copyfile(os.path.join(root, file), os.path.join(dest_path, file))



def os_get_file(folder=None, block_list=[], pattern=r'*.py'):
    # Get all the model.py into folder
    folder = os_package_root_path() if folder is None else folder
    # print(folder)
    module_names = get_recursive_files3(folder, pattern)
    print(module_names)


    NO_LIST = []
    NO_LIST = NO_LIST + block_list

    list_select = []
    for t in module_names:
        t = t.replace(folder, "").replace("\\", ".").replace(".py", "")

        flag = False
        for x in NO_LIST:
            if x in t: flag = True

        if not flag:
            list_select.append(t)

    return list_select


def model_get_list(folder=None, block_list=[]):
    # Get all the model.py into folder
    folder = os_package_root_path(__file__) if folder is None else folder
    # print(folder)
    module_names = get_recursive_files(folder, r'/*model*/*.py')

    NO_LIST = ["__init__", "util", "preprocess"]
    NO_LIST = NO_LIST + block_list

    list_select = []
    for t in module_names:
        t = t.replace(folder, "").replace("\\", ".")

        flag = False
        for x in NO_LIST:
            if x in t: flag = True

        if not flag:
            list_select.append(t)

    return list_select


def get_recursive_files2(folderPath, ext):
    results = os.listdir(folderPath)
    outFiles = []
    for file in results:
        if os.path.isdir(os.path.join(folderPath, file)):
            outFiles += get_recursive_files(os.path.join(folderPath, file), ext)
        elif re.match(ext, file):
            outFiles.append( folderPath + "/" + file)

    return outFiles


def get_recursive_files3(folderPath, ext):
    results = os.listdir(folderPath)
    outFiles = []
    for file in results:
        if os.path.isdir(os.path.join(folderPath, file)):
            outFiles += get_recursive_files(os.path.join(folderPath, file), ext)
        elif re.match(ext, file):
            outFiles.append(file)
    return outFiles



def get_model_uri(file):
  return Path(os.path.abspath(file)).parent.name + "." + os.path.basename(file).replace(".py", "")



def get_recursive_files(folderPath, ext='/*model*/*.py'):
    import glob
    files = glob.glob(folderPath + ext, recursive=True)
    return files




def path_norm(path=""):
    root = os_package_root_path(__file__, 0)

    if len(path) == 0 or path is None:
        path = root

    tag_list = [ "model_", "//model_",  "dataset", "template", "ztest"  ]


    for t in tag_list :
        if path.startswith(t) :
            path = os.path.join(root, path)
            return path
    return path


def path_norm_dict(ddict):
    for k,v in ddict.items():
        if "path" in k :
            ddict[k] = path_norm(v)
    return ddict


"""
def os_module_path():
    import inspect
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    # sys.path.insert(0, parent_dir)
    return parent_dir
"""

def os_package_root_path(filepath="", sublevel=0, path_add=""):
    """
       get the module package root folder
    """
    from pathlib import Path
    import mlmodels, os, inspect 

    path = Path(inspect.getfile(mlmodels)).parent
    print( path )

    # path = Path(os.path.realpath(filepath)).parent
    for i in range(1, sublevel + 1):
        path = path.parent

    path = os.path.join(path.absolute(), path_add)
    return path


def os_file_current_path():
    import inspect
    val = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # return current_dir + "/"
    # Path of current file
    # from pathlib import Path

    # val = Path().absolute()
    val = str(os.path.join(val, ""))
    # print(val)
    return val


####################################################################################################
def test_module(model_uri="model_tf/1_lstm.py", data_path="dataset/", pars_choice="json", reset=True):
    ###loading the command line arguments
    # model_uri = "model_xxxx/yyyy.py"

    log("#### Module init   #################################################")
    from mlmodels.models import module_load
    module = module_load(model_uri)
    log(module)

    log("#### Loading params   ##############################################")
    param_pars = {"choice": pars_choice, "data_path": data_path}
    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)

    log("#### Run module test   ##############################################")
    from mlmodels.models import test_module as test_module_global
    test_module_global(model_uri, model_pars, data_pars, compute_pars, out_pars)


####################################################################################################
def config_load_root():
    import json
    path_user = os.path.expanduser('~')
    path_config = path_user + "/.mlmodels/config.json"

    ddict = json.load(open(path_config, mode='r'))
    return ddict


def config_path_pretrained():
    ddict = config_load_root()
    return ddict['model_trained']


def config_path_dataset():
    ddict = config_load_root()
    return ddict['dataset']


def config_set(ddict2):
    ddict = config_load_root()

    for k,x in ddict2.items():
      ddict[k] = x

    json.dump(ddict, open(ddict, mode='w'))
   


def params_json_load(path, config_mode="test"):
    import json
    pars = json.load(open(path, mode="rb"))
    pars = pars[config_mode]

    ### HyperParam, model_pars, data_pars,
    list_pars = []
    for t in ["hypermodel_pars", "model_pars", "data_pars", "compute_pars", "out_pars"]:
        pdict = pars.get(t)
        if pdict:
            list_pars.append(pdict)
        else:
            log("error in json, cannot load ", t)

    return tuple(list_pars)



def load_config(args, config_file, config_mode, verbose=0):
    ##### Load file dict_pars as dict namespace #############################
    import json
    print(config_file) if verbose else None

    try:
        pars = json.load(open(config_file, mode="r"))
        # print(arg.param_file, model_pars)

        pars = pars[config_mode]  # test / prod
        print(config_file, pars) if verbose else None

        ### Overwrite dict_pars from CLI input and merge with toml file
        for key, x in vars(args).items():
            if x is not None:  # only values NOT set by CLI
                pars[key] = x

        # print(model_pars)
        pars = to_namespace(pars)  # like object/namespace model_pars.instance
        return pars

    except Exception as e:
        print(e)
        return args


def val(x, xdefault):
    try:
        return x if x is not None else xdefault
    except:
        return xdefault


####################################################################################################
####################################################################################################
def env_conda_build(env_pars=None):
    if env_pars is None:
        env_pars = {'name': "test", 'python_version': '3.6.5'}

    p = env_pars
    cmd = f"conda create -n {p['name']}  python={p['python_version']}  -y"
    print(cmd)
    os.system(cmd)


def env_pip_requirement(env_pars=None):
    from time import sleep
    if env_pars is None:
        env_pars = {'name': "test", 'requirement': 'requirements.txt'}

    root_path = os_package_root_path(__file__)
    p = env_pars
    # cmd = f"source activate {p['name']}  &&  "
    cmd = ""
    cmd = cmd + f"  pip install -r  {root_path}/{p['requirement']} "

    print("Installing ", cmd)
    os.system(cmd)
    sleep(60)


def env_pip_check(env_pars=None):
    from importlib import import_module

    if env_pars is None:
        env_pars = {'name': "test", 'requirement': 'requirements.txt', "import": ['tensorflow', 'sklearn']}

    flag = 0
    try:
        for f in env_pars['import']:
            import_module(f)
    except:
        flag = 1

    if flag:
        env_pip_requirement(env_pars)


def env_build(model_uri, env_pars):
    from time import sleep

    model_uri2 = model_uri.replace("/", ".")
    root = os_package_root_path()
    model_path = os.path.join(root, env_pars["model_path"])

    env_pars['name'] = model_uri2
    env_pars['python_version'] = "3.6.5"
    env_pars['file_requirement'] = model_path + "/requirements.txt"

    env_conda_build(env_pars=env_pars)
    sleep(60)

    env_pip_requirement(env_pars=env_pars)


####################################################################################################
########## TF specific #############################################################################
def tf_deprecation():
    try:
        from tensorflow.python.util import deprecation
        deprecation._PRINT_DEPRECATION_WARNINGS = False
        print("Deprecaton set to False")
    except:
        pass






####################################################################################################
###########  Utils #################################################################################
class Model_empty(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None
                 ):
        ### Model Structure        ################################
        self.model = None


def os_path_split(path) :
  return str(Path( path ).parent), str(Path( path ).name)



def load(load_pars):
    p = load_pars

    if "model_keras" in p['model_uri']:
        path = os.path.abspath(p['path'] + "/../")
        name = os.path.basename(p['path']) if ".h5" in p['path'] else "model.h5"
        return load_keras(load_pars)


def save(model=None, session=None, save_pars=None):
    p = save_pars
    if  "model_keras" in p['model_uri'] :
       path = os.path.abspath( p['path'] + "/../")
       name = os.path.basename(p['path']) if ".h5" in p['path'] else "model.h5"
       save_keras( model, session, save_pars)




def load_tf(load_pars=""):
  """
  https://www.mlflow.org/docs/latest/python_api/mlflow.tensorflow.html#

 """
  import mlflow
  import tensorflow as tf

  path, filename = os_path_split( load_pars['path'] ) 
   
  model_uri = path + "/" + filename
  tf_graph = tf.Graph()
  tf_sess = tf.Session(graph=tf_graph)
  with tf_graph.as_default():
    signature_def = mlflow.tensorflow.load_model(model_uri=model_uri,
                                                 tf_sess=tf_sess)
    input_tensors = [tf_graph.get_tensor_by_name(input_signature.name)
                     for _, input_signature in signature_def.inputs.items()]
    output_tensors = [tf_graph.get_tensor_by_name(output_signature.name)
                      for _, output_signature in signature_def.outputs.items()]
  return input_tensors, output_tensors


def save_tf(model=None, sess=None, save_pars= None):
  import tensorflow as tf
  path, filename = os_path_split(save_pars['path'])
  os.makedirs(path, exist_ok=True)

  saver = tf.compat.v1.train.Saver()
  return saver.save(sess, path)



def load_tch(load_pars):
    import torch
    path, filename = load_pars['path'], load_pars['filename']

    path = path + "/" + filename if "." not in path else path
    model = Model_empty()
    model.model = torch.load(path)
    return model


def save_tch(model=None, optimizer=None, save_pars=None):
    import torch
    path, filename = os_path_split(save_pars['path'])
    os.makedirs(path, exist_ok=True)

    if save_pars.get('save_state') is not None:
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{path}/{filename}" )

    else:
        torch.save(model.model, f"{path}/{filename}")


def save_tch_checkpoint(model, optimiser, save_pars):
    import torch
    path = save_pars['checkpoint_name']
    torch.save({
        'grad_step': save_pars["grad_step"],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, path )



# def load(model, optimiser, CHECKPOINT_NAME='nbeats-fiting-checkpoint.th'):
def load_tch_checkpoint(model, optimiser, load_pars):
    import torch
    CHECKPOINT_NAME = load_pars['checkpoint_name']
    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_step = checkpoint['grad_step']
        print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
        return grad_step
    return 0



def load_pkl(load_pars):
    import cloudpickle as pickle
    return pickle.load(open( load_pars['path'], mode='rb') )


def save_pkl(model=None, save_pars=None):
  import cloudpickle as pickle
  path, filename = os_path_split(save_pars['path'])
  os.makedirs(path, exist_ok=True)
  return pickle.dump(model, open( f"{path}/{filename}" , mode='wb') )



def load_keras(load_pars):
    from keras.models import load_model
    path, filename = os_path_split(load_pars['path']  )

    path_file = path + "/" + filename if ".h5" not in path else path
    model = Model_empty()
    model.model = load_model(path_file)
    return model


def save_keras(model=None, session=None, save_pars=None, ):
    path, filename = os_path_split(save_pars['path']  )
    os.makedirs(path, exist_ok=True)
    model.model.save(str(Path(path) / filename))



def save_gluonts(model=None, session=None, save_pars=None):
    path = save_pars['path']
    os.makedirs(path, exist_ok=True)
    model.model.serialize(Path(path))



def load_gluonts(load_pars=None):
    from gluonts.model.predictor import Predictor
    predictor_deserialized = Predictor.deserialize(Path( load_pars['path'] ))

    model = Model_empty()
    model.model = predictor_deserialized
    return model








"""
def path_local_setup(current_file=None, out_folder="", sublevel=0, data_path="dataset/"):
    root = os_package_root_path(__file__, sublevel=0, path_add="")

    out_path = path_norm(out_folder)
    data_path = path_norm(data_path)

    model_path = f"{out_path}/model/"
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    log(data_path, out_path, model_path)
    return data_path, out_path, model_path
"""







