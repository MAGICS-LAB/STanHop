from data.data_loader import Dataset_MTS
from code_exp.exp_basic import Exp_Basic
from code_models.stanhop import STanHopNet
from torchmetrics.functional import r2_score

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import os
import time
import json
import pickle

import warnings
import wandb

warnings.filterwarnings('ignore')

class Exp_Stanhop(Exp_Basic):
    def __init__(self, args):
        super(Exp_Stanhop, self).__init__(args)

    def _build_model(self):        
        model = STanHopNet(
            self.args.data_dim, 
            self.args.in_len, 
            self.args.out_len,
            self.args.seg_len,
            self.args.win_size,
            self.args.factor,
            self.args.d_model, 
            self.args.d_ff,
            self.args.n_heads, 
            self.args.e_layers,
            self.args.dropout, 
            self.args.baseline,
            self.device
        ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def __get_alpha(self):
        
        # return [0], [0], [0]
        time_alpha = []
        sec_alpha_1 = []
        sec_alpha_2 = []

        for layers in self.model.encoder.encode_blocks:
            for l in layers.encode_layers:
                a = l.time_attention.inner_attention.entmax.alpha.item()
                time_alpha.append(1+ (1/(1 + np.exp(-a))))

        for layers in self.model.encoder.encode_blocks:
            for l in layers.encode_layers:
                a = l.dim_sender.inner_attention.entmax.alpha.item()
                sec_alpha_1.append( 1+ (1/(1 + np.exp(-a))))

        for layers in self.model.encoder.encode_blocks:
            for l in layers.encode_layers:
                a = l.dim_receiver.inner_attention.entmax.alpha.item()
                sec_alpha_2.append(1+ (1/(1 + np.exp(-a))))

        return time_alpha, sec_alpha_1, sec_alpha_2

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size;
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size;
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.in_len, args.out_len],  
            data_split = args.data_split,
        )
        g = torch.Generator()
        g.manual_seed(0)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            generator=g)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        scale_statistic = {'mean': train_data.scaler.mean, 'std': train_data.scaler.std}
        with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
            pickle.dump(scale_statistic, f)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        log_data = {
            'epoch':[],
            'time attn alpha':[],
            'sec attn alpha 1': [],
            'sec attn alpha 2':[],
            'train loss':[]
        }

        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                model_optim.step()
            
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # log_data = {
            #     'train loss': train_loss,
            #     'valid loss':vali_loss,
            #     'test loss':test_loss,
            #     'steps':train_steps,
            #     'epoch':epoch+1,
            # }
            # wandb.log(log_data)
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
        
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path+'/'+'checkpoint.pth')

        return self.model, early_stopping.val_loss_min

    def test(self, setting, save_pred = False, inverse = False):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, inverse)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        # log_data = {
        #     'test mse':mse,
        #     'test mae':mae
        # }
        # wandb.log(log_data)
        # wandb.finish()
        return float(mse), float(mae)

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse = False):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs = self.model(batch_x)

        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)

        return outputs, batch_y
    
    def eval(self, setting, save_pred = False, inverse = False):
        #evaluate a saved model
        args = self.args
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag='test',
            size=[args.in_len, args.out_len],  
            data_split = args.data_split,
            scale = True,
            scale_statistic = args.scale_statistic,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(data_loader):
                pred, true = self._process_one_batch(
                    data_set, batch_x, batch_y, inverse)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        a = torch.from_numpy(preds).permute(2,1,0).reshape(44, -1)
        b = torch.from_numpy(trues).permute(2,1,0).reshape(44, -1)
        r2 = r2_score(a, b).item()
        print(r2)

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        if (save_pred):
            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)

        return mae, mse, rmse, mape, mspe
