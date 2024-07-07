import random
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
import time
from transform import *
from model import *
from losses import LossFunction
import logging
logger = logging.getLogger(__name__)

def adjust_learning_rate(lr, epoch):
    """Sets the learning rate to the initial LR decayed 10 times every 10 rounds"""
    return lr * (0.5 ** (epoch // 5))

def count_parameters(model):
    params = sum(p.numel() for n,p in model.named_parameters() if p.requires_grad and 'blocks' in n)
    return params

def normalization(data, m):
    epsilon = 1e-8
    data = np.clip(data, 0 + epsilon, m - epsilon)
    return data.tolist()

import time
def train(args, dataloader, epoch, weight_bias):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    logger.info("===================================")
    logger.info(f"epoch:{epoch}")
    if((not args.continue_train) and epoch == 0):
        model = model_type(args, retrained = False).cuda()
        num_params = count_parameters(model)
        logger.info("Training parameters %s", args)
        logger.info(f"Total Parameter of original model: {num_params}")
    else:
        model = model_type(args, retrained = True).cuda()
    model.train()
    begin = time.time()
    cross_entropy = nn.CrossEntropyLoss()
    criterion = LossFunction(cross_entropy, model, alpha = args.alpha)
    optimizer = optim.SGD(model.parameters(), lr=args.lr , momentum = args.momentum)
    for round in range(args.rounds):
        for inputs, labels, sa, ma, paths in tqdm(dataloader,ncols=50):
            inputs, labels = inputs.to('cuda', non_blocking=True), labels.to('cuda', non_blocking=True) # torch.Size([32, 3, 224, 224])
            optimizer.zero_grad()
            args.v_grad = [0.] * len(args.varsigma)
            outputs = model(args, inputs, sa, ma) # outputs:torch.Size([32, 1000]) inputs: torch.Size([32, 3, 224, 224])
            loss = criterion(args, outputs, labels, weight_bias)
            loss.backward()
            optimizer.step()
            if(not args.manual):
                if(args.no_norm):
                    args.varsigma = [args.varsigma[i] - args.auto_lr  * args.v_grad[i] for i in range(len(args.varsigma))]
                else:
                    args.varsigma = normalization([args.varsigma[i] - args.auto_lr  * float(args.v_grad[i]) for i in range(len(args.varsigma))], 4)
                    
    logger.info(f'cost_time:{time.time() - begin:.4f}s')
    save_model(args, model)
    return

@torch.no_grad()
def allData_train_test(args, output_list, c_list, label_list, dataloader):
    model = model_type(args, training = False).cuda()
    model.eval()
    correct = 0
    total = 0
    for inputs, labels, _, _, paths in tqdm(dataloader,ncols=50):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.cuda.amp.autocast():
            outputs = model(args, inputs)
        
        output_list.append(outputs.detach().cpu().numpy())
        label_list.append(labels.detach().cpu().numpy())
        scores, indices = outputs.topk(1)

        for i in range(inputs.size(0)):
            if(paths[i].split('/')[-3] == 'test_A' or paths[i].split('/')[-4] == 'train_A'): # belongs to A domain
                if(indices[i,0] == labels[i]):
                    c_list.append('blue') # blue
                else:
                    c_list.append('midnightblue') # midnightblue
            else:
                if(indices[i,0] == labels[i]):
                    c_list.append('red') # red
                else:
                    c_list.append('darkred') # darkred
        if(args.debug):
            logger.info(f"A:{indices[0,0] == labels[0]}, label:{labels[0]}, top3:{indices[0][0]},{indices[0][1]},{indices[0][2]}:[{scores[0][0]:.2f},{scores[0][1]:.2f},{scores[0][2]:.2f}]")
        correct += (indices[:,0] == labels).sum()
        total += inputs.size(0)

    return correct/total

@torch.no_grad()
def allData_test(args, output_list, c_list, label_list, dataloader):
    model = model_type(args, training = False).cuda()
    model.eval()
    correct = 0
    total = 0
    for inputs, labels, _, paths in tqdm(dataloader,ncols=50):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.cuda.amp.autocast():
            outputs = model(args, inputs)
        
        output_list.append(outputs.detach().cpu().numpy())
        label_list.append(labels.detach().cpu().numpy())
        scores, indices = outputs.topk(1)

        for i in range(inputs.size(0)):
            if(paths[i].split('/')[-3] == 'test_A' or paths[i].split('/')[-4] == 'train_A'): # belongs to A domain
                if(indices[i,0] == labels[i]):
                    c_list.append('blue') # blue
                else:
                    c_list.append('midnightblue') # midnightblue
            else:
                if(indices[i,0] == labels[i]):
                    c_list.append('red') # red
                else:
                    c_list.append('darkred') # darkred
        if(args.debug):
            logger.info(f"A:{indices[0,0] == labels[0]}, label:{labels[0]}, top3:{indices[0][0]},{indices[0][1]},{indices[0][2]}:[{scores[0][0]:.2f},{scores[0][1]:.2f},{scores[0][2]:.2f}]")
        correct += (indices[:,0] == labels).sum()
        total += inputs.size(0)

    return correct/total

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from statistics import mean
@torch.no_grad()
def imbData_test(args, epoch, dataloader):
    model = model_type(args, training = False).cuda()
    model.eval()
    count_a, count_a_yhat1, count_a_y1, count_a_y0, count_a_y1_yhat1, count_a_y0_yhat0 = 0, 0, 0, 0, 0, 0 
    count_na, count_na_yhat1, count_na_y1, count_na_y0, count_na_y1_yhat1, count_na_y0_yhat0 = 0, 0, 0, 0, 0, 0  
    for idx, (inputs, labels, sa, paths) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.cuda.amp.autocast():
            outputs = model(args, inputs)
        # scores, indices = outputs.topk(1)
        scores, indices = outputs.max(1)
        for i in range(inputs.shape[0]):
            if sa[i] == 1:
                count_a += 1
                if indices[i] == 1:
                    count_a_yhat1 += 1
                    if labels[i] == 1:
                        count_a_y1_yhat1 += 1
                if indices[i] == 0:
                    if labels[i] == 0:
                        count_a_y0_yhat0 += 1
                if labels[i] == 1:
                    count_a_y1 += 1
                if labels[i] == 0:
                    count_a_y0 += 1
                
            else:
                count_na += 1
                if indices[i] == 1:
                    count_na_yhat1 += 1
                    if labels[i] == 1:
                        count_na_y1_yhat1 += 1
                if indices[i] == 0:
                    if labels[i] == 0:
                        count_na_y0_yhat0 += 1
                if labels[i] == 1:
                    count_na_y1 += 1
                if labels[i] == 0:
                    count_na_y0 += 1
    DP = abs(count_a_yhat1/count_a - count_na_yhat1/count_na)
    EO = abs(count_a_y1_yhat1/count_a_y1 - count_na_y1_yhat1/count_na_y1)
    BA = 1/4*(count_a_y1_yhat1/count_a_y1 + count_a_y0_yhat0/count_a_y0 + count_na_y1_yhat1/count_na_y1 + count_na_y0_yhat0/count_na_y0)
    logger.info(f"DP:{DP:.4f}, EO:{EO:.4f}, BA:{BA:.4f}")
    return DP, EO, BA

def show_2d(args, output_list, c_list, label_list, mode):
    output_list = np.vstack(output_list)
    c_list = np.hstack(c_list)
    label_list = np.hstack(label_list)
    X = [[],[]] # this calculates the whole dataset
    for i in range(output_list.shape[0]):
        X[0].append(output_list[i,label_list[i]])
        list_tmp = list(set(output_list[i].argsort()[-3:])- set([label_list[i]]))
        X[1].append(output_list[i,list_tmp].sum())

    plt.figure(figsize=(10,8),dpi=100)
    plt.scatter(X[0],X[1],c=c_list) # x: 200; y: 200

    # wrong number calculation
    AwX,BwX = [[],[]],[[],[]]
    for i in range(output_list.shape[0]):
        if(c_list[i] == 'midnightblue'):
            AwX[0].append(X[0][i])
            AwX[1].append(X[1][i])
        elif(c_list[i] == 'darkred'):
            BwX[0].append(X[0][i])
            BwX[1].append(X[1][i])
    AwX[0],AwX[1],BwX[0],BwX[1] = np.mean(AwX[0]),np.mean(AwX[1]),np.mean(BwX[0]),np.mean(BwX[1])

    plt.scatter(AwX[0],AwX[1],c ='cyan')
    plt.scatter(BwX[0],BwX[1],c ='orange')

    # Logistics line calculation
    Y = []
    for i in range(output_list.shape[0]):
        if(c_list[i] == 'midnightblue' or c_list[i] == 'darkred'):
            Y.append(0)
        elif(c_list[i] == 'blue' or c_list[i] == 'red'):
            Y.append(1)
        else:
            raise('Color error')
    X, Y = np.array(X).transpose(), np.array(Y)
    lr_cv = LogisticRegressionCV(Cs=10, cv=5, solver='lbfgs')
    lr_cv.fit(X, Y)
    theta_1, theta_2 = lr_cv.coef_[0]
    b = lr_cv.intercept_[0]

    # Drawing decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    xx = np.arange(x_min, x_max, 0.01)
    yy = (-theta_1 / theta_2) * xx - b / theta_2

    return (theta_1, theta_2, b)
    