# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""


"""
import io
import os
import subprocess
import sys

from setuptools import find_packages, setup

######################################################################################
root = os.path.abspath(os.path.dirname(__file__))


##### check if GPU available  #########################################################
try :
  p = subprocess.Popen(["command -v nvidia-smi"], stdout=subprocess.PIPE, shell=True)
  out = p.communicate()[0].decode("utf8")
  gpu_available = len(out) > 0
except : pass


##### Version  #######################################################################
version ='0.31.1'
print("version", version)





######################################################################################
with open('requirements.txt') as fp:
    install_requires = fp.read()



######################################################################################
with open("README.md", "r") as fh:
    long_description = fh.read()


long_description =  """

```

ml_models --do                    
    "testall"     :  test all modules inside model_tf
    "test"        :  test a certain module inside model_tf


    "model_list"  :  #list all models in the repo          
    "fit"         :  wrap fit generic m    ethod
    "predict"     :  predict  using a pre-trained model and some data
    "generate_config"  :  generate config file from code source


ml_optim --do
  "test"      :  Test the hyperparameter optimization for a specific model
  "test_all"  :  TODO, Test all
  "search"    :  search for the best hyperparameters of a specific model


 


##### Include models :

mlmodels.model_dev.ml_mosaic.py
mlmodels.model_dev.mytest.py


mlmodels.model_flow.mlflow_run.py


mlmodels.model_gluon.gluon_automl.py
mlmodels.model_gluon.gluon_deepar.py
mlmodels.model_gluon.gluon_ffn.py
mlmodels.model_gluon.gluon_prophet.py


mlmodels.model_keras.00_template.py
mlmodels.model_keras.01_deepctr.py
mlmodels.model_keras.02_cnn.py
mlmodels.model_keras.preprocess.py


mlmodels.model_rank.LambdaRank.py
mlmodels.model_rank.load_mslr.py
mlmodels.model_rank.metrics.py
mlmodels.model_rank.RankNet.py


mlmodels.model_sklearn.model.py


mlmodels.model_tch.cnn_classifier.py
mlmodels.model_tch.mlp.py
mlmodels.model_tch.nbeats.py
mlmodels.model_tch.sentence_transformer.py
mlmodels.model_tch.transformer_classifier.py


mlmodels.model_tf.10_encoder_vanilla.py
mlmodels.model_tf.11_bidirectional_vanilla.py
mlmodels.model_tf.12_vanilla_2path.py
mlmodels.model_tf.13_lstm_seq2seq.py
mlmodels.model_tf.14_lstm_attention.py
mlmodels.model_tf.15_lstm_seq2seq_attention.py
mlmodels.model_tf.16_lstm_seq2seq_bidirectional.py
mlmodels.model_tf.17_lstm_seq2seq_bidirectional_attention.py
mlmodels.model_tf.18_lstm_attention_scaleddot.py
mlmodels.model_tf.19_lstm_dilated.py
mlmodels.model_tf.1_lstm.py
mlmodels.model_tf.20_only_attention.py
mlmodels.model_tf.21_multihead_attention.py
mlmodels.model_tf.22_lstm_bahdanau.py
mlmodels.model_tf.23_lstm_luong.py
mlmodels.model_tf.24_lstm_luong_bahdanau.py
mlmodels.model_tf.25_dnc.py
mlmodels.model_tf.26_lstm_residual.py
mlmodels.model_tf.27_byte_net.py
mlmodels.model_tf.28_attention_is_all_you_need.py
mlmodels.model_tf.29_fairseq.py
mlmodels.model_tf.2_encoder_lstm.py
mlmodels.model_tf.3_bidirectional_lstm.py
mlmodels.model_tf.4_lstm_2path.py
mlmodels.model_tf.50lstm attention.py
mlmodels.model_tf.5_gru.py
mlmodels.model_tf.6_encoder_gru.py
mlmodels.model_tf.7_bidirectional_gru.py
mlmodels.model_tf.8_gru_2path.py
mlmodels.model_tf.9_vanilla.py
mlmodels.model_tf.access.py
mlmodels.model_tf.addressing.py
mlmodels.model_tf.autoencoder.py
mlmodels.model_tf.dnc.py


```







"""



### Packages  ####################################################
packages = ["mlmodels"] + ["mlmodels." + p for p in find_packages("mlmodels")]


### CLI Scripts  #################################################
"""
scripts = [ "mlmodels/models.py",
            "mlmodels/optim.py",
            "mlmodels/cli_mlmodels",     
            ]


"""
scripts = [ "mlmodels/distri_torch_mpirun.sh",     
            ]



### CLI Scripts  #################################################   
entry_points={ 'console_scripts': [
               'ml_models = mlmodels.models:main',
               'ml_optim = mlmodels.optim:main',
               'ml_test = mlmodels.ztest:main'
              ] }


##################################################################   
setup(
    name="mlmodels",
    version=version,
    description="Generic model API, Model Zoo in Tensorflow, Keras, Pytorch, Hyperparamter search",
    keywords='Machine Learning Interface library',
    
    author="Kevin Noel",
    author_email="brookm291@gmail.com",
    url="https://github.com/arita37/mlmodels",
    
    install_requires=install_requires,
    python_requires='>=3.6',
    
    packages=packages,
    
    #### CLI
    scripts = scripts,
  
    ### CLI pyton
    entry_points= entry_points,
    
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,

    classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: ' +
          'Artificial Intelligence',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: ' +
          'Python Modules',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Environment :: Console',
          'Environment :: Web Environment',
          'Operating System :: POSIX',
          'Operating System :: MacOS :: MacOS X',
      ]
)





################################################################################
################################################################################
"""


https://packaging.python.org/tutorials/packaging-projects/


import io
import os
import subprocess
import sys

from setuptools import setup, find_packages

root = os.path.abspath(os.path.dirname(__file__))


# required packages for NLP Architect
with open('requirements.txt') as fp:
    install_requirements = fp.readlines()

# check if GPU available
p = subprocess.Popen(['command -v nvidia-smi'], stdout=subprocess.PIPE, shell=True)
out = p.communicate()[0].decode('utf8')
gpu_available = len(out) > 0

# Tensorflow version (make sure CPU/MKL/GPU versions exist before changing)
for r in install_requirements:
    if r.startswith('tensorflow=='):
        tf_version = r.split('==')[1]

# default TF is CPU
chosen_tf = 'tensorflow=={}'.format(tf_version)
# check system is linux for MKL/GPU backends
if 'linux' in sys.platform:
    system_type = 'linux'
    tf_be = os.getenv('NLP_ARCHITECT_BE', False)
    if tf_be and 'mkl' == tf_be.lower():
        chosen_tf = 'intel-tensorflow=={}'.format(tf_version)
    elif tf_be and 'gpu' == tf_be.lower() and gpu_available:
        chosen_tf = 'tensorflow-gpu=={}'.format(tf_version)

for r in install_requirements:
    if r.startswith('tensorflow=='):
        install_requirements[install_requirements.index(r)] = chosen_tf

with open('README.md', encoding='utf8') as fp:
    long_desc = fp.read()

with io.open(os.path.join(root, 'nlp_architect', 'version.py'), encoding='utf8') as f:
    version_f = {}
    exec(f.read(), version_f)
    version = version_f['NLP_ARCHITECT_VERSION']

setup(name='nlp-architect',
      version=version,
      description='Intel AI Lab\'s open-source NLP and NLU research library',
      long_description=long_desc,
      long_description_content_type='text/markdown',
      keywords='NLP NLU deep learning natural language processing tensorflow keras dynet',
      author='Intel AI Lab',
      packages=find_packages(exclude=['tests.*', 'tests', '*.tests', '*.tests.*',
                                      'examples.*', 'examples', '*.examples', '*.examples.*']),
      install_requires=install_requirements,
      scripts=['nlp_architect/nlp_architect'],
      include_package_data=True,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: ' +
          'Artificial Intelligence',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: ' +
          'Python Modules',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Environment :: Console',
          'Environment :: Web Environment',
          'Operating System :: POSIX',
          'Operating System :: MacOS :: MacOS X',
      ]
      )



import os
from io import open

from setuptools import find_packages, setup

packages = ['elfi'] + ['elfi.' + p for p in find_packages('elfi')]

# include C++ examples
package_data = {'elfi.examples': ['cpp/Makefile', 'cpp/*.txt', 'cpp/*.cpp']}

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

optionals = {'doc': ['Sphinx'], 'graphviz': ['graphviz>=0.7.1']}

# read version number
__version__ = open('elfi/__init__.py').readlines()[-1].split(' ')[-1].strip().strip("'\"")

setup(
    name='elfi',
    keywords='abc likelihood-free statistics',
    packages=packages,
    package_data=package_data,
    version=__version__,
    author='ELFI authors',
    author_email='elfi-support@hiit.fi',
    url='http://elfi.readthedocs.io',
    install_requires=requirements,
    extras_require=optionals,
    description='ELFI - Engine for Likelihood-free Inference',
    long_description=(open('docs/description.rst').read()),
    license='BSD',
    classifiers=[
        'Programming Language :: Python :: 3.5', 'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Mathematics', 'Operating System :: OS Independent',
        'Development Status :: 4 - Beta', 'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License'
    ],
    zip_safe=False)



"""
