import pandas as pd
import os
import cvxpy as cp
import numpy as np
import copy
import math
from tqdm import tqdm,trange
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
#from cvxpylayers.torch import CvxpyLayer
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
from models import LTP_MMF
import yaml
#def setup_seed(seed):
seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
#torch.backends.cudnn.deterministic = True



class BatchedMFFeedback(object):
    def __init__(self,args):
        self.dim = args.dim
        self.device = args.device
        self.lambda_u = args.lambda_u
        self.lambda_i = args.lambda_i
        self.batch_size = args.batch_size
        self.topk = args.topk

        self.alpha_u = args.alpha_u
        self.beta_u = args.beta_u
        simulator_path = os.path.join('simulator', "steam_simulator.npy")
        print(simulator_path)
        #data_name =
        self.simulator = self.load_click_model(simulator_path)
        self.n_users, self.n_items = self.simulator.shape
        #self.fairness_type = args.fairness_type

        self.init_ranking_parameters(dim=args.dim)

        self.init_reranking_parameters(data_name=args.Dataset, user_field=args.user_field,
                                       item_field=args.item_field, provider_field=args.provider_field,
                                       topK=args.topk, alpha=args.alpha)


        self.fairness_model = LTP_MMF.LTP_MMF(rho=self.rho, M=self.M, TopK=args.topk, item_num=self.n_items)

    def load_click_model(self,click_model_path):
        simualtor = np.load(click_model_path)
        return simualtor

    def get_feedback(self,user,item):
        if random.random() < self.simulator[user,item]:
            return 1
        else:
            return 0


    def init_ranking_parameters(self, dim, trained_v_u = None, trained_v_i = None):

        self.v_u = torch.ones((self.n_users,dim)).to(self.device)
        self.v_i = torch.ones((self.n_items,dim)).to(self.device)

        self.v_u = F.normalize(self.v_u,p=2,dim=-1)
        self.v_i = F.normalize(self.v_i,p=2,dim=-1)

        #self.A = [self.lambda_u*torch.eye(dim) for u in self.n_users]
        self.A = torch.zeros((self.n_users,dim,dim)).to(self.device)
        for u in range(self.n_users):
            self.A[u] = self.lambda_u*torch.eye(dim).to(self.device)

        self.A_inverse = torch.zeros((self.n_users,dim,dim)).to(self.device)
        for u in range(self.n_users):
            self.A_inverse[u] = (1/self.lambda_u)*torch.eye(dim).to(self.device)

        self.b = torch.zeros(self.n_users,dim).to(self.device)

        self.C_inverse = torch.zeros((self.n_items,dim,dim)).to(self.device)
        for i in range(self.n_items):
            self.C_inverse[i] = (1/self.lambda_i)*torch.eye(dim).to(self.device)



        self.C = torch.zeros((self.n_items,dim,dim)).to(self.device)
        for i in range(self.n_items):
            self.C[i] = self.lambda_i*torch.eye(dim).to(self.device)

        self.d = torch.zeros(self.n_items,dim).to(self.device)

    def update_model(self,user,item,click):
        batch_size = click.shape
        #print(batch_size)
        for i in trange(batch_size[0]):
            v_i = self.v_i[item[i]].unsqueeze(-1)

            #self.A[user[i]] = self.A[user[i]] + torch.matmul(v_i,v_i.permute(0,2,1))
            self.A[user[i]] = self.A[user[i]] + torch.matmul(v_i,v_i.t())
            A_reverse = torch.inverse(self.A[user[i]])
            self.A_inverse[user[i]] = A_reverse

            self.b[user[i]] = self.b[user[i]] + self.v_i[item[i]] * click[i]
            self.v_u[user[i]] = torch.matmul(A_reverse,self.b[user[i]])

            #self.v_u[user[i]] = F.normalize(self.v_u[user[i]],p=2,dim=-1)
            self.v_u[user[i]] = F.normalize(self.v_u[user[i]],p=2,dim=-1)
            v_u = self.v_u[user[i]].unsqueeze(-1)


            #self.C[item[i]] = self.C[item[i]] + torch.matmul(v_u,v_u.permute(0,2,1))
            self.C[item[i]] = self.C[item[i]] + torch.matmul(v_u,v_u.t())
            self.d[item[i]] = self.d[item[i]] + self.v_u[user[i]] * click[i]

            C_reverse = torch.inverse(self.C[item[i]])
            self.C_inverse[item[i]] = C_reverse
            self.v_i[item[i]] = F.normalize(torch.matmul(C_reverse,self.d[item[i]]),p=2,dim=-1)


    def calulate_item_variance(self, x, A):
        #x [k,d]
        #A [d,d]
        xA = torch.matmul(x,A)
        xAx = torch.matmul(xA,x.t())
        return torch.sqrt(torch.diag(xAx))


    def calulate_user_variance(self, x, A):
        #x [d]
        #A [k,d,d]
        x = x.unsqueeze(0).unsqueeze(0)
        xA = torch.matmul(x,A).squeeze(1)
        #[k,d]
        x_t = x.squeeze(1)
        #[1,d]
        xAx = torch.matmul(xA,x_t.t())
        #[k,1]
        return torch.sqrt(xAx.squeeze(1))


    def train(self, datas):
        print("start to train...")
        length = len(datas)
        #users = []
        #items = []
        exposures_all = np.zeros(self.num_providers)
        ndcgs_list = []
        MMF_list = []
        ctr_all_list = []

        for b in trange(int(np.ceil(length/self.batch_size))):
            ndcg = []
            ctr_list = []
            exposures = np.zeros(self.num_providers)
            buffer_user = []
            buffer_item = []
            buffer_click = []

            min_index = b * self.batch_size

            max_index = min(length, (b+1)*self.batch_size)

            #users = datas[min_index:max_index]
            users = copy.copy(datas[min_index:max_index])
            random.shuffle(users)
            user_length = len(users)
            ## get ranking scores from the accuracy module
            predicted_UI = torch.clamp(torch.matmul(self.v_u[users], self.v_i.t()), 1e-2, 1.0)

            bandit_modified_UI = torch.zeros((user_length,self.n_items))
            for u_index in range(user_length):
                u = users[u_index]
                v_u = self.v_u[u]

                item_ucb = self.calulate_item_variance(self.v_i, self.A_inverse[u])
                user_ucb = self.calulate_user_variance(v_u, self.C_inverse)
                #note that in Eq.(8) $\sqrt(\ln t)$ will converge slower than (1-a)^t. Therefore, we only keep the (1-a)^t term.
                ucb = self.alpha_u*(0.0001**(b/100)) * item_ucb + self.beta_u*(0.0001**(b/100)) * user_ucb
                bandit_modified_UI[u_index,:] = predicted_UI[u_index,:] + ucb

            #make recommednation
            recommended_list = self.fairness_model.recommendation(bandit_modified_UI)


            #eval...
            for i in range(user_length):
                u = users[i]
                dcg = 0
                ctr = 0
                for k in range(self.topk):

                    item = recommended_list[i][k]
                    dcg = dcg + self.simulator[u,item]/np.log2(2+k)
                    ctr = ctr + self.simulator[u,item]
                    exposures[self.item2provider[item]] = exposures[self.item2provider[item]] + 1
                    exposures_all[self.item2provider[item]] = exposures_all[self.item2provider[item]] + 1
                    click = self.get_feedback(u,item)
                    buffer_user.append(u)
                    buffer_item.append(item)
                    buffer_click.append(click)
                ndcg.append(dcg/self.max_dcg[u])
                ctr_list.append(ctr/self.topk)
            #ndcgs_all.append(dcg/self.max_dcg[u])

            ndcg_epoch = np.mean(ndcg)
            ctr_epoch = np.mean(ctr_list)
            mmf_epoch = np.min(exposures/(self.rho*self.topk*user_length))
            print("epoch:", b)
            print("NDCG:%.3f MMF:%.3f CTR:%.3f"%(ndcg_epoch,mmf_epoch,ctr_epoch))
            ndcgs_list.append(ndcg_epoch)
            MMF_list.append(mmf_epoch)
            ctr_all_list.append(ctr_epoch)
            #exit(0)
            self.update_model(buffer_user, buffer_item, np.array(buffer_click))

        #self.update_v(user=list(set(datas[:,0])),item=list(set(datas[:,1])))
        print("final NDCG:%.3f MMF:%.3f CTR:%.3f"%(np.mean(ndcgs_list),np.mean(MMF_list),np.mean(ctr_all_list)))


    def init_reranking_parameters(self, data_name, user_field, item_field, provider_field, topK, alpha):
        data_path = os.path.join('dataset','steam.inter')
        frames = pd.read_csv(data_path,delimiter='\t',dtype={user_field:int,item_field:int, provider_field:int},usecols=[user_field,item_field,provider_field])
        self.num_providers = len(frames[provider_field].unique())
        providerLen = np.array(frames.groupby(provider_field).size().values)
        self.rho = (1-alpha)*providerLen/np.sum(providerLen) + alpha * np.array([2/self.num_providers for i in range(self.num_providers)])
        self.M = np.zeros((self.n_items,self.num_providers))
        iid2pid = []
        tmp = frames[[item_field,provider_field]].drop_duplicates()
        self.item2provider = {x:y for x,y in zip(tmp[item_field],tmp[provider_field])}
        for i in range(self.n_items):
            iid2pid.append(self.item2provider[i])
            self.M[i,self.item2provider[i]] = 1

        ####store max dcgs for evaluation
        self.max_dcg = []
        UI_matrix = copy.copy(self.simulator)
        UI_matrix_sort = np.sort(UI_matrix,axis=-1)
        # check_invalid_user = 0
        for u in range(self.n_users):
            self.max_dcg.append(0)

            for k in range(topK):
                self.max_dcg[u] = self.max_dcg[u] + UI_matrix_sort[u,self.n_items-k-1]/np.log2(k+2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="batched_mf")
    parser.add_argument('--base_model', default='mf')
    parser.add_argument('--Dataset', type=str, default='steam',
                        help='your data.')
    parser.add_argument('--device', type=str, default="cuda",
                        help='cuda.')

    parser.add_argument('--lambda_u', type=float, default=0.1)
    parser.add_argument('--lambda_i', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--sample_rate', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--topk', type=int, default=10)

    parser.add_argument('--alpha_u', type=float, default=1)
    parser.add_argument('--beta_u', type=float, default=1)

    parser.add_argument('--user_field', type=str, default='user_id:token')
    parser.add_argument('--item_field', type=str, default='product_id:token')
    parser.add_argument('--provider_field', type=str, default='publisher:token')
    parser.add_argument('--label_filed', type=str, default='label:float')
    parser.add_argument('--time_field', type=str, default='timestamp:float')

    args = parser.parse_args()
    print(args)
    model = BatchedMFFeedback(args)
    users = list(range(256))
    data = users * 100
    model.train(data)

