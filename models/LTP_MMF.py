import pandas as pd
import os
import cvxpy as cp
import numpy as np
import math
from tqdm import tqdm,trange
import yaml
import torch

def compute_projection_maxmin_fairness_with_order(ordered_tilde_dual, rho, lambd):

    m = len(rho)
    answer = cp.Variable(m)
    objective = cp.Minimize(cp.sum_squares(cp.multiply(rho,answer) - cp.multiply(rho, ordered_tilde_dual)))
    #objective = cp.Minimize(cp.sum(cp.multiply(rho,answer) - cp.multiply(rho, ordered_tilde_dual)))
    constraints = []
    for i in range(1, m+1):
        constraints += [cp.sum(cp.multiply(rho[:i],answer[:i])) >= -lambd]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    #print(type(result))
    #exit(0)
    #print(type(answer.value))
    return answer.value



def compute_next_dual(eta, rho, dual, gradient, lambd):
    #rho = self.problem.data.rho
    tilde_dual = dual - eta*gradient/rho/rho
    order = np.argsort(tilde_dual*rho)
    ordered_tilde_dual = tilde_dual[order]
    # print ordered_tilde_dual*rho
    ordered_next_dual = compute_projection_maxmin_fairness_with_order(ordered_tilde_dual, rho[order], lambd)
    # print(ordered_next_dual)
    # print("tilde_dual", rho*tilde_dual)
    # print("next_dual", rho*ordered_next_dual[order.argsort()])
    return ordered_next_dual[order.argsort()]

class LTP_MMF(object):
    def __init__(self, rho, M, TopK, item_num):
        self.rho = rho
        self.M = M
        self.TopK = TopK
        self.item_num = item_num
        f = open("properties/LTP-MMF.yaml",'r')
        self.hyper_parameters = yaml.load(f)
        f.close()
        print("hyper parameters:")
        print(self.hyper_parameters)
        self.lambd = self.hyper_parameters['lambda']
        self.learning_rate = self.hyper_parameters['learning_rate']
        self.gamma = self.hyper_parameters['gamma']
        #self.l = self.hyper_parameters['l']

    def recommendation(self, batch_UI):
        batch_UI = batch_UI.cpu().numpy()
        batch_size = len(batch_UI)
        B_t = batch_size*self.TopK*self.rho

        _ , num_providers = self.M.shape
        #B_l = np.zeros(num_providers)
        recommended_list = []
        mu_t = np.zeros(num_providers)
        eta = self.learning_rate/math.sqrt(self.item_num)
        gradient_cusum = np.zeros(num_providers)
        #gradient_list = []
       # if fairtype == 'OF':
        for t in range(batch_size):

            x_title = batch_UI[t,:] - np.matmul(self.M,mu_t)
            mask = np.matmul(self.M,(B_t>0).astype(np.float))
            mask = (1.0-mask) * -100.0
            x = np.argsort(x_title+mask,axis=-1)[::-1]
            x_allocation = x[:self.TopK]
            re_allocation = np.argsort(batch_UI[t,x_allocation])[::-1]
            x_allocation = x_allocation[re_allocation]
            recommended_list.append(x_allocation)
            B_t = B_t - np.sum(self.M[x_allocation],axis=0,keepdims=False)
            gradient = -np.mean(self.M[x_allocation],axis=0,keepdims=False) + self.rho

            #gradient_list.append(gradient)
            gradient = self.gamma * gradient + (1-self.gamma) * gradient_cusum
            gradient_cusum = gradient
            #gradient = -(B_0-B_t)/((t+1)*K) + rho
            for g in range(1):
                mu_t = compute_next_dual(eta, self.rho, mu_t, gradient, self.lambd)
        return recommended_list

