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

# Construct classifier
def model_init():
    clflg = SGDClassifier(loss='log')
    clfsgd = SGDClassifier()
    clfpe = Perceptron()
    
    demo_kernel = kernel.Polynomial_Kernel(scale_factor = 2, intercept = 1,
                                       degree = 1)
    demo_kernel_2 = kernel.Linear_Kernel()
    demo_kernel_3 = kernel.RBF_Kernel(d = 0.5)
    demo_loss = loss.Ramp(parameter = -1.5, policy = "static") #1.5 is very resistant to noise and drift

    clfnolca_poly = NOLCA(demo_kernel, loss = demo_loss)
    clfnolca_rbf = NOLCA(demo_kernel_3,loss = demo_loss)
    model = [clflg,clfsgd,clfpe,clfnolca_poly,clfnolca_rbf]
    model_name = ['Logistic','hinge-SVM','Perceptron','ramp-SVM(Poly)','ramp-SVM(RBF)']
    return model,model_name

##Calculate the weights 
def cal_weight(weight,yitazt,zt,j,yita):
    sum_he = np.dot(np.array(weight),np.exp(yitazt)) 
    weight[j] = (weight[j] * np.exp((-yita)*zt[j]))/(sum_he) 
    return weight

##Initialization parameter
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
    yy = np.array(yy)
    yy = yy.astype('int32')
    # Perform different normalization operations according to different data characteristics
    if data_name == "connect-4_binary.csv" or data_name == "covertype_binary.csv":
        xx = (xx - np.min(xx)) / (np.max(xx) - np.min(xx))# 0-1normalization
    else:
        xx = (xx - np.mean(xx, axis=0)) / np.std(xx, axis=0)# Z-score
    model_plot_data_big = []
    vote_list = []##Store total score
    vote_count_big = []
    for yita_order,yita in enumerate(yita_l):
        yita_set.append(yita)
        ##The model is initialized with each yita corresponding to a combinatorial classifier
        base_model,base_model_name = model_init()
        
        n,model_plot_data_small,weight,zt,yitazt,base_model_end,vote_count,base_model_count,vote_count_small = parameter_init(base_model,x,yita)
        
        ##svm online learning
        clf1 = base_model[n-1]
        clf1.training(xx, yy_noise, learning_rate = 0.01, 
                     reg_coefficient = 0)
        clf2 = base_model[n-2]
        clf2.training(xx, yy_noise, learning_rate = 0.01, 
                     reg_coefficient = 0)
    
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
            for j in range(0,n-2):
                clf = base_model[j]
                clf.partial_fit(np.array(x.loc[i]).reshape(1,-1),noise_y['y'].loc[i].ravel(),classes=np.unique(noise_y)) ##Incremental learning
                if i!=len(x)-1:
                    weight = cal_weight(weight,yitazt,zt,j,yita) ##Computed weight function
                    
                    single_pre = clf.predict(np.array(x.loc[i+1]).reshape(1,-1)) ##Prediction for the next sample
                    base_model_end[j] = single_pre[0]  
                    if single_pre==y['y'].loc[i+1]:
                        base_model_count[j] += 1  ##Base classifier correct number
                        zt[j] = 0
                    else:
                        zt[j] = 1
                    model_plot_data_small[j][i] = base_model_count[j]/(i+1)
#             print('weight:', weight)
            weight_i = weight.copy()
            weights.append(weight_i)
#             print(weights)
            
            if i!=len(x)-1:
                weight = cal_weight(weight,yitazt,zt,n-1,yita) 
                single_pre = clf1.get_prediction()[i+1] 
                base_model_end[n-1] = single_pre  
                if single_pre==yy[i+1]:
                    base_model_count[n-1] += 1  
                    zt[n-1] = 0
                else:
                    zt[n-1] = 1
                model_plot_data_small[n-1][i] = base_model_count[n-1]/(i+1)
                

                weight = cal_weight(weight,yitazt,zt,n-2,yita) 
                single_pre = clf2.get_prediction()[i+1] 
                base_model_end[n-2] = single_pre  
                if single_pre==yy[i+1]:
                    base_model_count[n-2] += 1  
                    zt[n-2] = 0
                else:
                    zt[n-2] = 1
                model_plot_data_small[n-2][i] = base_model_count[n-2]/(i+1)

            

            
            if i!=len(x)-1:
                label_vote = np.dot(base_model_end,weight) ##The base classifier and the weight are used to calculate the value of the ensenmble classifier for the next sample
                if label_vote>0:
                    label = 1
                else:
                    label = -1
                if label==y['y'].loc[i+1]:
                    vote_count += 1 ##Calculate the correct number of ensenble classifiers
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
        y_accuracy = []
        for index, value in enumerate(model_plot_data_big[max_index][i]):
            if index%100 == 0:
                y_accuracy.append(value)
        accuracy=np.array(y_accuracy)
        np.save(base_model_name[i]+" "+data_name+'.npy',accuracy)

    vote_count_big[max_index].pop()
    y_total_accuracy = []
    for index, value in enumerate(vote_count_big[max_index]):
        if index%100 == 0:
            y_total_accuracy.append(value)
    y_total=np.array(y_total_accuracy)
    np.save('OEC'+" "+data_name+'.npy',y_total)




data_list = ['au2_10000.csv','connect-4_binary.csv','covertype_binary.csv','poker-hand_binary.csv']
for i in data_list:
    data_name = i
    data = pd.read_csv(i)
    y = data.loc[:,data.columns=='y']
    x = data.loc[:,data.columns!='y']
    y.columns = ['y']
    noise_y = y
    yita_l = [0.001]
    build_model(x,y,noise_y,yita_l)


