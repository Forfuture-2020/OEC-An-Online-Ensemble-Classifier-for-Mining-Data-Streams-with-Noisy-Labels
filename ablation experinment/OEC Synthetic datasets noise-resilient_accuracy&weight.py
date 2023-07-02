import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import  Perceptron
from olsvm import kernel
from olsvm.nolca import NOLCA
from olsvm import loss

def model_init():
    clflg = SGDClassifier(loss='log')
    clfsgd = SGDClassifier()
    clfpe = Perceptron()
    model = [clflg,clfsgd,clfpe]
    model_name = ['Logistic','hinge-SVM','Perceptron']
    return model,model_name


def cal_weight(weight,yitazt,zt,j,yita):
    sum_he = np.dot(np.array(weight),np.exp(yitazt)) 
    weight[j] = (weight[j] * np.exp((-yita)*zt[j]))/(sum_he) 
    return weight

def parameter_init(base_model,x,yita):
    n = len(base_model)
    model_plot_data_small = [[0]*len(x) for _ in range(n)]
    weight = [1/n]*n
    zt = [0]*n
    yitazt = [x*(-yita) for x in zt]
    base_model_end = [0]*n
    vote_count = 0
    base_model_count = [0]*n    
    vote_count_small = []
    return n,model_plot_data_small,weight,zt,yitazt,base_model_end,vote_count,base_model_count,vote_count_small

def build_model(x,y,noise_y,yita_l):
    yita_set = []
    yy_noise = [noise_y for x in np.array(noise_y) for noise_y in x]
    xx = np.array(x)
    yy_noise = np.array(yy_noise)
    yy_noise = yy_noise.astype('int32')
    
    yy = [y for x in np.array(y) for y in x]
    xx = np.array(x)

    yy = np.array(yy)
    yy = yy.astype('int32')
    
    model_plot_data_big = []
    vote_list = []
    vote_count_big = []
    for yita_order,yita in enumerate(yita_l):
        yita_set.append(yita)
        
        base_model,base_model_name = model_init()
        
        n,model_plot_data_small,weight,zt,yitazt,base_model_end,vote_count,base_model_count,vote_count_small = parameter_init(base_model,x,yita)
        

        weight_i = weight.copy()
        weights = []
        weights.append(weight_i)

        
        for i in range(0,len(x)):
            import math as m
            count = m.floor(m.log2(i+1))
            if 2**count <= i <= 2**(count+1)-1 :
                a = 2**count
                yita = float(m.sqrt(2*m.log(5)/ a)) 
                yita_set.append(yita)
            
            if(i%(int(len(x)/10))==0):
                print('已完成{}%'.format(i/(int(len(x)/10))*10))
            for j in range(0,n):
                clf = base_model[j]
                clf.partial_fit(np.array(x.loc[i]).reshape(1,-1),noise_y['y'].loc[i].ravel(),classes=np.unique(noise_y))
                if i!=len(x)-1:
                    weight = cal_weight(weight,yitazt,zt,j,yita) 
                    single_pre = clf.predict(np.array(x.loc[i+1]).reshape(1,-1))
                    base_model_end[j] = single_pre[0]  
                    if single_pre==y['y'].loc[i+1]:
                        base_model_count[j] += 1  
                        zt[j] = 0
                    else:
                        zt[j] = 1
                    model_plot_data_small[j][i] = base_model_count[j]/(i+1)
            weight_i = weight.copy()
            weights.append(weight_i)

                
            if i!=len(x)-1:
                label_vote = np.dot(base_model_end,weight) 
                if label_vote>0:
                    label = 1
                else:
                    label = -1
                if label==y['y'].loc[i+1]:
                    vote_count += 1 
                vote_count_small.append(vote_count/(i+1))
        count_t = [round(m / len(x),4) for m in base_model_count]
        vote = round(vote_count/len(x),4)
        
        weights = np.array(weights)

        
        if((vote>np.array(count_t)).all()):
            print('当前yita:',yita,'基分类器准确率',count_t,'组合分类器准确率',vote,'组合分类器更优')
        else:
            print('当前yita:',yita,'基分类器准确率',count_t,'组合分类器准确率',vote,'组合分类器一般')
        print('基分类器的权重:',weight)
        model_plot_data_big.append(model_plot_data_small)
        vote_list.append(vote)
        vote_count_big.append(vote_count_small)
    max_index = vote_list.index(max(vote_list))

    for i in range(0,n):
        y_weight = []
        y_accuracy = []
        for index, value in enumerate(model_plot_data_big[max_index][i]):
            if index%100 == 0:
                y_accuracy.append(value)
        accuracy=np.array(y_accuracy)
        np.save(base_model_name[i]+data_name+'_concept&noise_noise-resilient.npy',accuracy)
        for index, value in enumerate(weights[:,i]):
            if index%100 == 0:
                y_weight.append(value)
        weight=np.array(weight)
        np.save(base_model_name[i]+data_name+'_concept&noise_noise-resilient_weight.npy',weight)

    vote_count_big[max_index].pop()
    y_total_accuracy = []
    for index, value in enumerate(vote_count_big[max_index]):
        if index%100 == 0:
            y_total_accuracy.append(value)
    y_total=np.array(y_total_accuracy)
    np.save('OEC'+data_name+'_concept&noise_noise-resilient.npy',y_total)


def add_noise_y(y,yita=0.1):
    m,n = y.shape
    noise = np.zeros([m,n])
    for i in range(m):
        if np.random.rand(1) < yita:
            if y['y'].loc[i]== -1:
                noise[i] = 1
            else:
                noise[i] = -1
        else:
            noise[i] = y['y'].loc[i]
    noise = pd.DataFrame(noise,columns = y.columns)
    return noise

data_list = ['hyperplane_100000.csv','SEA_21000.csv']
for i in data_list:
    data_name = i
    data = pd.read_csv(i)
    y = data.loc[:,data.columns=='y']
    x = data.loc[:,data.columns!='y']
    y.columns = ['y']
    noise_y = y
    noise_y = add_noise_y(y,0.4)
    yita_l = [0.001]
    build_model(x,y,noise_y,yita_l)




