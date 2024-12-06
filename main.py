import os
import sys
sys.path.append(os.path.join(".", 'DANet'))
from DAN_Task import DANetClassifier, DANetRegressor
import argparse
import torch.distributed
import torch.backends.cudnn
from sklearn.metrics import accuracy_score, mean_squared_error
from lib.utils import normalize_reg_label
from qhoptim.pyt import QHAdam
from config.default import cfg
import pandas as pd
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
danet_args = {
    'seed': 42,
    'task': 'classification',
    'resume_dir': '',
    'logname': "danet_log"
}
danet_model_args = {
    'base_outdim': 64,
    'k': 5,
    'drop_rate': 0.1,
    'layer': 20
}
danet_fit_args = {
    'lr': 0.008,
    'max_epochs': 4000,
    'patience': 1500,
    'batch_size': 8192,
    'virtual_batch_size': 256,
    "weight_decay": 1e-5,  # 自己加的
    "schedule_step": 20  # 自己加的
}
def get_args():
    # parser = argparse.ArgumentParser(description='PyTorch v1.4, DANet Task Training')
    # parser.add_argument('-c', '--config', type=str, required=False, default='config/default.py', metavar="FILE", help='Path to config file')
    # parser.add_argument('-g', '--gpu_id', type=str, default='1', help='GPU ID')

    # args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # torch.backends.cudnn.benchmark = True if len(args.gpu_id) < 2 else False
    # if args.config:
    #     cfg.merge_from_file(args.config)
    # cfg.freeze()
    task = danet_args["task"]
    seed = danet_args["seed"]
    train_config = {'resume_dir': danet_args["resume_dir"], 'logname': danet_args["logname"]}
    print('Using config: ', danet_args)

    return train_config, danet_fit_args, danet_model_args, task, seed
def set_task_model(task, std=None, seed=1):
    if task == 'classification':
        clf = DANetClassifier(
            optimizer_fn=QHAdam,
            optimizer_params=dict(lr=fit_config['lr'], weight_decay=1e-5, nus=(0.8, 1.0)),
            scheduler_params=dict(gamma=0.95, step_size=20),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            layer=model_config['layer'],
            base_outdim=model_config['base_outdim'],
            k=model_config['k'],
            drop_rate=model_config['drop_rate'],
            seed=seed
        )
        eval_metric = ['accuracy']

    elif task == 'regression':
        clf = DANetRegressor(
            std=std,
            optimizer_fn=QHAdam,
            optimizer_params=dict(lr=fit_config['lr'], weight_decay=fit_config['weight_decay'], nus=(0.8, 1.0)),
            scheduler_params=dict(gamma=0.95, step_size=fit_config['schedule_step']),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            layer=model_config['layer'],
            base_outdim=model_config['base_outdim'],
            k=model_config['k'],
            seed=seed
        )
        eval_metric = ['mse']
    return clf, eval_metric
print('===> Setting configuration ...')
train_config, fit_config, model_config, task, seed = get_args()
logname = None if train_config['logname'] == '' else train_config['logname']
print('===> Getting data ...')
input_path = "."
tempPreprocessingFiles_path = "dmfinal-temppreprocessingfiles"
train_temp_path = os.path.join(input_path, tempPreprocessingFiles_path, "train_type1_temp.parquet")
test_temp_path = os.path.join(input_path, tempPreprocessingFiles_path, "test_type1_temp.parquet")
if os.path.exists(train_temp_path) and os.path.exists(test_temp_path):
    train = pd.read_parquet(train_temp_path)
    test = pd.read_parquet(test_temp_path)

y = train["sii"]
X = train.drop(columns=["sii"])

# 處理 missing value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(X)
# train = pd.DataFrame(imputer.transform(train), columns=train.columns)
X = imputer.transform(X)



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=seed)

mu, std = None, None
if task == 'regression':
    mu, std = y_train.mean(), y_train.std()
    print("mean = %.5f, std = %.5f" % (mu, std))
    y_train = normalize_reg_label(y_train, std, mu)
    y_valid = normalize_reg_label(y_valid, std, mu)
clf, eval_metric = set_task_model(task, std, seed)

clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_valid, y_valid)],
    eval_name=['valid'],
    eval_metric=eval_metric,
    max_epochs=fit_config['max_epochs'], patience=fit_config['patience'],
    batch_size=fit_config['batch_size'], virtual_batch_size=fit_config['virtual_batch_size'],
    logname=logname,
    resume_dir=train_config['resume_dir'],
    n_gpu=1
)

# preds_test = clf.predict(X_test)

# if task == 'classification':
#     test_acc = accuracy_score(y_pred=preds_test, y_true=y_test)
#     print(f"FINAL TEST ACCURACY FOR {train_config['dataset']} : {test_acc}")

# elif task == 'regression':
#     test_mse = mean_squared_error(y_pred=preds_test, y_true=y_test)
#     print(f"FINAL TEST MSE FOR {train_config['dataset']} : {test_mse}")
