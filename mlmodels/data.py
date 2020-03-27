
import os
import sys


from mlmodels.util import os_package_root_path, path_norm





def download_data(data_pars):
  """
  download_data({"from_path" :  "tabular",  
                        "out_path" :  path_norm("ztest/dataset/text/") } )

  Open URL
     https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AAAoFh0aO9RqwwROksGgasIha?dl=0


  """
  from cli_code.cli_download import Downloader

  folder = data_pars['from_path']  # dataset/text/

  urlmap = {
     "text" :    "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AADHrhC7rLkd42_CEqK6A9oYa/dataset/text?dl=1&subfolder_nav_tracking=1"
     ,"tabular" : "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AAAxZkJTGSumLADzj3B5wbA0a/dataset/tabular?dl=1&subfolder_nav_tracking=1"
     ,"pretrained" : "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AACL3LHW1USWrvsV5hipw27ia/model_pretrained?dl=1&subfolder_nav_tracking=1"

     ,"vision" : "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AAAM4k7rQrkjBo09YudYV-6Ca/dataset/vision?dl=1&subfolder_nav_tracking=1"
     ,"recommender": "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AABIb2JjQ6aQHwfq5CU0ypHOa/dataset/recommender?dl=1&subfolder_nav_tracking=1"

  }

  url = urlmap[folder]

  #prefix = "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/"
  #url= f"{prefix}/AADHrhC7rLkd42_CEqK6A9oYa/{folder}?dl=1&subfolder_nav_tracking=1"

  out_path = data_pars['out_path']

  zipname = folder.split("/")[-1]


  os.makedirs(out_path, exist_ok=True)
  downloader = Downloader(url)
  downloader.download(out_path)

  import zipfile
  with zipfile.ZipFile( out_path + "/" + zipname + ".zip" ,"r") as zip_ref:
      zip_ref.extractall(out_path)




####################################################################################
def import_data_tch(name="", mode="train", node_id=0, data_folder_root=""):
    import torch.utils.data.distributed
    from torchvision import datasets, transforms

    if name == "mnist" :
        data_folder = os.path.join( data_folder_root,  "data-%d" % node_id)
        dataset = datasets.MNIST(
            data_folder,
            train=True if mode =="train" else False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        return dataset


def import_data_fromfile(**kw):
   """
       data_pars["data_path"]
   
   """ 
   import pandas as pd
   import numpy as np

   m = kw
   if m.get("uri_type") in ["pickle", "pandas_pickle" ]:
       df = pd.read_pickle(m["data_path"])
       return df


   if m.get("uri_type") in ["csv", "pandas_csv" ]:
       df = pd.read_csv(m["data_path"])
       return df


   if m.get("uri_type") in [ "dask" ]:
       df = pd.read_csv(m["data_path"])
       return df


   if ".npz" in m['data_path'] :
       arr = import_to_numpy(data_pars, mode=mode, **kw)
       return arr


   if ".csv" in m['data_path'] or ".txt" in m['data_path']  :
       df = import_to_pandas(data_pars, mode=mode, **kw)
       return df


   if ".pkl" in m['data_path']   :
       df = pd.read_pickle(m["data_path"], **kw)
       return df




def import_data_dask(**kw):
  extension = kw['path'].split(".")[-1]

  if kw.get("use_dask", False):
    import dask.dataframe as dd
    if extension in [".csv", ".txt"]: 
       df = dd.read_csv(kw["data_path"])
    elif extension in [".pkl"]: 
       df = dd.read_pickle(kw["data_path"])
    elif extension in [".npz"]: 
       df = dd.read_pickle(m["data_path"])
    else: raise Exception(f"Not support extension {extension}")
  return df






def import_data():
  def sent_generator(TRAIN_DATA_FILE, chunksize):
      import pandas as pd
      reader = pd.read_csv(TRAIN_DATA_FILE, chunksize=chunksize, iterator=True)
      for df in reader:
          val3  = df.iloc[:, 3:4].values.tolist()
          val4  = df.iloc[:, 4:5].values.tolist()
          flat3 = [item for sublist in val3 for item in sublist]
          flat4 = [str(item) for sublist in val4 for item in sublist]
          texts = []
          texts.extend(flat3[:])
          texts.extend(flat4[:])

          sequences  = model.tokenizer.texts_to_sequences(texts)
          data_train = pad_sequences(sequences, maxlen=data_pars["MAX_SEQUENCE_LENGTH"])
          yield [data_train, data_train]

  model.model.fit(sent_generator(data_pars["train_data_path"], batch_size / 2),
                  epochs          = epochs,
                  steps_per_epoch = n_steps,
                  validation_data = (data_pars["data_1_val"], data_pars["data_1_val"]))






def get_dataset(data_pars) : 
  """
    path:
    is_local  : Local to the repo 
    data_type:
    train : 1/0
    data_source :  ams
  """
  dd = data_pars

  if not d.get('is_local') is None :
      dd['path'] = os_package_root_path(__file__, 0, dd['path'] )


  if dd['train'] :
     df = pd.read_csv(path) 



     ### Donwload from external


     ## Get from csv, local 


     ## Get from csv, external


     ### Get from external tool


