from Utils import Config
import numpy as np 
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from Utils import Logger
import random as r
from subprocess import check_output

def get_pid(name):
    return list(map(int,check_output(["pidof",name]).split()))

class Trainer():
    def __init__(self , filename) -> None:
        config = Config(filename)
        self.config = config
        self.ID = filename.split('.')[0]
        self.logger = Logger(config)
        self.interval = config.interval
        self.dataset = Dataset(config)
        self.savedpath = config.savedpath
        config.device = torch.device("cuda")
        if os.environ["CUDA_VISIBLE_DEVICES"] != "":
            self.model = eval(f"{config.model}")(config).cuda()
        else:
            self.model = eval(f"{config.model}")(config)

        self.optimizer =  torch.optim.Adam(self.model.parameters(), lr=config.learning_rate , weight_decay=config.L2_penalty)
        self.loss_fn = nn.BCELoss()
        self.best_auc = 0.
        self.epoch = 0
        self.early_stop_cnt = config.early_stop
        self.config = config
        if hasattr(config , 'pretrain'):
            self.savedpath = config.pretrain
            self.resume()

    @property
    def current_state(self):
        return {
                'optimizer': self.optimizer.state_dict(), 
                'model': self.model.state_dict() , 
                'early_stop_cnt': self.early_stop_cnt , 
                'best_auc':self.best_auc,
                'epoch':self.step
                }
    
    def resume(self):
        save_info = torch.load(self.savedpath)
        self.optimizer.load_state_dict(save_info['optimizer'])
        self.model.load_state_dict(save_info['model'])
        self.epoch = save_info['epoch'] + 1
        self.best_auc = save_info['best_auc']
        self.early_stop_cnt = save_info['early_stop_cnt']
        print("model loaded !")
    
    def run(self):
        self.train_process()
        self.evaluation_process()

    def train_process(self):
        for i in range(self.epoch, 1000):
            self.step = i
            self.train_epoch()        
            if i % self.interval == 0:
                auc , logloss = self.test_epoch(self.dataset.val)
                self.logger.record(self.step, auc, logloss, 'val')
                if auc > self.best_auc:
                    self.has_live = True
                    print('find a better model !')
                    self.best_auc = auc
                    self.early_stop_cnt = self.config.early_stop
                    torch.save(self.current_state , self.savedpath + '_best')
                else:
                    self.early_stop_cnt -= 1
                if self.early_stop_cnt == 0:
                    return
    
    def evaluation_process(self):
        saved_info = torch.load(self.savedpath + '_best')
        self.model.load_state_dict(saved_info['model'])
        auc, logloss = self.test_epoch(self.dataset.test)
        self.logger.record(self.step, auc, logloss, 'test')

    def train_epoch(self):
        cnt = 0
        res = 0
        self.model.train()
        for fetch_data in tqdm(self.dataset.train) if self.config.verbose else self.dataset.train:
            cnt += 1

            # This evaluation mode is for streaming training, default **disabled**.
            if self.config.enable_in_batch_eval and cnt % self.config.in_batch_interval == 0:
                auc , logloss = self.test_epoch(self.dataset.val)
                self.logger.record(self.step , auc , logloss , 'val')
                if auc > self.best_auc:
                    print('find a better model !')
                    self.best_auc = auc
                    self.early_stop_cnt = self.config.early_stop
                    torch.save(self.current_state , self.savedpath + '_best')
                else:
                    self.early_stop_cnt -= 1
                    if self.early_stop_cnt == 0:
                        return True
            
            self.optimizer.zero_grad()
            prediction = self.model(fetch_data)
            loss = self.loss_fn(prediction.squeeze(-1) , fetch_data['label'].squeeze(-1).cuda())
            loss.backward()
            self.optimizer.step()
            res += loss.cpu().item()

    def test_epoch(self , datasource):
        with torch.no_grad():
            self.model.eval()
            val , truth = [] , []
            for fetch_data in tqdm(datasource) if self.config.verbose else datasource:
                prediction = self.model(fetch_data)
                val.append(prediction.cpu().numpy())
                truth.append(fetch_data['label'].numpy())
            y_hat = np.concatenate(val, axis=0).squeeze()
            y = np.concatenate(truth, axis=0).squeeze()
            auc = roc_auc_score(y, y_hat)
            logloss = - np.sum(y*np.log(y_hat + 1e-6) + (1-y)*np.log(1-y_hat+1e-6)) /len(y)
        return auc , logloss

if __name__ == '__main__':
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_files", help="path to load config", default="/home/data/tz/DAGFM_pytorch/RunTimeConf/default.yaml")
    parser.add_argument("--gpu", help="GPU ids", default=0)
    args = parser.parse_args()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from torch import nn
    from Data.dataset import Dataset
    from Model import *
    setup_seed(2022)
    trainer = Trainer(args.config_files)
    trainer.run()
