import os, json


import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import hub 

# import mlmodels.models as M

from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri, path_norm_dict
MODEL_URI = get_model_uri(__file__)

# CV datasets come in various formats, we should write a dataloader for each dataset
# I assume that the dataloader (itrator) will be ready and imported from another file

###########################################################################################################
###########################################################################################################
def _train(m, device, train_itr, criterion, optimizer, epoch, max_epoch):
    m.train()
    corrects, train_loss = 0.0,0.0
    for i,batch in enumerate(train_itr):
        print(i)
        if i == 10: break
        image, target = batch[0], batch[1]
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad()
        logit = m(image)
        
        loss = criterion(logit, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()
    
    size = len(train_itr)
    train_loss /= size 
    accuracy = 100.0 * corrects/size
  
    return train_loss, accuracy
    
def _valid(m, device, test_itr, criterion):
    m.eval()
    corrects, test_loss = 0.0,0.0
    for i,batch in enumerate(test_itr):
        print(i)
        if i == 10: break
        image, target = batch[0], batch[1]
        image, target = image.to(device), target.to(device)
        
        logit = m(image)
        loss = criterion(logit, target)

        
        test_loss += loss.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()
    
    size = len(test_itr)
    test_loss /= size 
    accuracy = 100.0 * corrects/size
    
    return test_loss, accuracy

def _get_device():
    # use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def get_config_file():
    return os.path.join(
        os_package_root_path(__file__, 1),
        'config', 'model_tch', 'Imagecnn.json')



###########################################################################################################
###########################################################################################################

 
#############
# functions #
#############

class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, out_pars=None):
        ### Model Structure        ################################
        if model_pars is None :
            self.model = None
            return self

        else:
            self.model = None

        _model      = model_pars['model']
        num_classes = model_pars['num_classes']
        pretrained  = bool(model_pars['pretrained'])
        assert _model in ['alexnet', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 
        'inception_v3', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
        'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
        'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn',
        'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'googlenet', 'shufflenet_v2_x0_5', 
        'shufflenet_v2_x1_0', 'mobilenet_v2'],\
        "Pretrained models are available for \
        the following models only: alexnet, densenet121, densenet169, densenet201,\
        densenet161, inception_v3, resnet18, resnet34, resnet50, resnet101, resnet152,\
        resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, squeezenet1_0,\
        squeezenet1_1, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,\
        googlenet, shufflenet_v2_x0_5, shufflenet_v2_x1_0, mobilenet_v2"


        self.model = hub.load('pytorch/vision', _model, pretrained=pretrained)

        if num_classes != 1000:
            fc_in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(fc_in_features, num_classes)




def get_params(param_pars=None, **kw):
    pp          = param_pars
    choice      = pp['choice']
    config_mode = pp['config_mode']
    data_path   = pp['data_path']
    if choice == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]

        ####Normalize path  : add /models/dataset/
        cf['data_pars'] = path_norm_dict(cf['data_pars'])
        cf['out_pars'] = path_norm_dict(cf['out_pars'])

        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")


def get_dataset(data_pars=None, **kw):
    data_path        = data_pars['data_path']
    train_batch_size = data_pars['train_batch_size']
    test_batch_size  = data_pars['test_batch_size']

    if data_pars['dataset'] == 'MNIST':
        train_loader = torch.utils.data.DataLoader( datasets.MNIST(data_pars['data_path'], train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Grayscale(num_output_channels=3),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=data_pars['train_batch_size'], shuffle=True)

        valid_loader = torch.utils.data.DataLoader( datasets.MNIST(data_pars['data_path'], train=False,
                        transform=transforms.Compose([
                            transforms.Grayscale(num_output_channels=3),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=data_pars['test_batch_size'], shuffle=True)
    else:
        print("Dataloader not implemented")
        exit
    return train_loader, valid_loader


def fit(model, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    model         = model.model
    lr            = compute_pars['learning_rate']
    epochs        = compute_pars["epochs"]
    criterion     = nn.CrossEntropyLoss()
    device        = _get_device()
    model.to(device)
    train_loss    = []
    train_acc     = []
    test_loss     = []
    test_acc      = []
    best_test_acc = -1

    optimizer     = optim.Adam(model.parameters(), lr=lr)
    train_iter, valid_iter = get_dataset(data_pars)
    for epoch in range(1, epochs + 1):
        #train loss
        tr_loss, tr_acc = _train(model, device, train_iter, criterion, optimizer, epoch, epochs)
        print( f'Train Epoch: {epoch} \t Loss: {tr_loss} \t Accuracy: {tr_acc}')

        ts_loss, ts_acc = _valid(model, device, valid_iter, criterion)
        print( f'Train Epoch: {epoch} \t Loss: {ts_loss} \t Accuracy: {ts_acc}')

        if ts_acc > best_test_acc:
            best_test_acc = ts_acc
            #save paras(snapshot)
            print("model saves at {}% accuracy".format(best_test_acc))

            os.makedirs(out_pars["checkpointdir"], exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(out_pars["checkpointdir"],  "best_accuracy"))

        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        test_loss.append(ts_loss)
        test_acc.append(ts_acc)
    return model, None


def predict(model, session=None, data_pars=None, compute_pars=None, out_pars=None):
    # get a batch of data
    _, valid_iter = get_dataset(data_pars=data_pars)
    device = _get_device()
    x_test = next(iter(valid_iter))[0].to(device)
    return model(x_test).detach().cpu().numpy()


def fit_metrics(model, data_pars=None, compute_pars=None, out_pars=None):
    pass


def save(model, path):
    return torch.save(model, path)


def load(path):
    return torch.load(path)


###########################################################################################################
###########################################################################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test

    log("#### Loading params   ##############################################")
    param_pars = {"choice":pars_choice,  "data_path":data_path,  "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)


    log("#### Loading dataset   #############################################")
    xtuple = get_dataset(data_pars)


    log("#### Model init, fit   #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)
    model, session = fit(model, data_pars, compute_pars, out_pars)


    log("#### Predict   #####################################################")
    ypred = predict(model, session, data_pars, compute_pars, out_pars)


    log("#### metrics   #####################################################")
    metrics_val = fit_metrics(model, data_pars, compute_pars, out_pars)
    print(metrics_val)


    log("#### Plot   ########################################################")


    log("#### Save/Load   ###################################################")
    save(model, session, out_pars)
    model2 = load( out_pars )
    #     ypred = predict(model2, data_pars, compute_pars, out_pars)
    #     metrics_val = metrics(model2, ypred, data_pars, compute_pars, out_pars)
    print(model2)



if __name__ == "__main__":
    test(data_path="dataset/json/Imagecnn.json", pars_choice="json", config_mode="test")