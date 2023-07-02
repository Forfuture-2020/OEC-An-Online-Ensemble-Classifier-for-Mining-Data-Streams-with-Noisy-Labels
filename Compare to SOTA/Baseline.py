'''
A Multi-Stage Automated Online Network Data Stream Analytics Framework for IIoT Systems
    
[1] L. Yang and A. Shami, “A Multi-Stage Automated Online Network Data Stream Analytics Framework for IIoT Systems,” 
IEEE Transactions on Industrial Informatics, vol. 19, no. 2, pp. 2107-2116, Feb. 2023, doi: 10.1109/TII.2022.3212003.

https://github.com/Western-OC2-Lab/MSANA-Online-Data-Stream-Analytics-And-Concept-Drift-Adaptation 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import time
from river import stream




"""online learning"""
from river import metrics
from river import stream
from river import tree,neighbors,naive_bayes,ensemble,linear_model
from river.drift import DDM, ADWIN
from river.drift import DDM, ADWIN,EDDM,HDDM_A,HDDM_W,KSWIN,PageHinkley
from river import feature_selection
from river import stats
from river import imblearn
from river import preprocessing
from river import evaluate
from river import metrics
# Define a generic adaptive learning function
# The argument "model" means an online adaptive learning algorithm
def adaptive_learning(model, X_train, X_test, y_train, y_test):
    metric = metrics.Accuracy() # Use accuracy as the metric
    i = 0 # count the number of evaluated data points
    t = [] # record the number of evaluated data points
    m = [] # record the real-time accuracy
    yt = [] # record all the true labels of the test set
    yp = [] # record all the predicted labels of the test set

    # Learn the training set
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        model.learn_one(xi1,yi1) 

    # Predict the test set
    for xi, yi in stream.iter_pandas(X_test, y_test):

        y_pred= model.predict_one(xi)  # Predict the test sample
        model.learn_one(xi,yi) # Learn the test sample
        metric = metric.update(yi, y_pred) # Update the real-time accuracy
        t.append(i)
        m.append(metric.get()*100)
        yt.append(yi)
        yp.append(y_pred)
        i = i+1
    print("Accuracy: "+str(round(accuracy_score(yt,yp),4)*100)+"%")
    print("Precision: "+str(round(precision_score(yt,yp),4)*100)+"%")
    print("Recall: "+str(round(recall_score(yt,yp),4)*100)+"%")
    print("F1-score: "+str(round(f1_score(yt,yp),4)*100)+"%")
    return t, m


"""Base model learning"""
    # Define a figure function that shows the real-time accuracy changes
def acc_fig(t, m, name):
    plt.rcParams.update({'font.size': 15})
    plt.figure(1,figsize=(10,6)) 
    sns.set_style("darkgrid")
    plt.clf() 
    plt.plot(t,m,'-b',label='Avg Accuracy: %.2f%%'%(m[-1]))

    plt.legend(loc='best')
    plt.title(name+' '+data_name, fontsize=15)   
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy (%)')
    plt.draw()
    plt.savefig(name+' '+data_name+'.PNG')



# Define the Window-based Performance Weighted Probability Averaging Ensemble (W-PWPAE) model

def MSANA(model1, model2, model3, model4, X_train, y_train, X_test, y_test):
    # Record the real-time accuracy of PWPAE and 4 base learners
    metric = metrics.Accuracy()
    metric1 = metrics.Accuracy()
    metric2 = metrics.Accuracy()
    metric3 = metrics.Accuracy()
    metric4 = metrics.Accuracy()
    
    metric_w1 = []
    metric_w2 = []
    metric_w3 = []
    metric_w4 = []


    i=0
    t = []
    m = []
    m1 = []
    m2 = []
    m3 = []
    m4 = []
    yt = []
    yp = []

    hat1 = model1
    hat2 = model2
    hat3 = model3
    hat4 = model4
    
    # Define the two feature selections methods: Variance Threshold and Select-K-Best
    selector1 = feature_selection.VarianceThreshold(threshold = 0.1)
    selector2 = feature_selection.SelectKBest(similarity=stats.PearsonCorr(),k=40)
    
    # Use EDDM to detect concept drift, it can be replaced with other drift detection methods like ADWIN, DDM, etc.
    eddm = EDDM()
    drift = 0

    # Initial feature selection on the training set
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        selector1.learn_one(xi1) 
    
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        xi1 = selector1.transform_one(xi1)
        selector2.learn_one(xi1,yi1) 
        
    # Train the online models on the training set
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        xi1 = selector1.transform_one(xi1)
        xi1 = selector2.transform_one(xi1)
        print(xi1)
        print(yi1)
        hat1.learn_one(xi1,yi1)
        hat2.learn_one(xi1,yi1)
        hat3.learn_one(xi1,yi1)
        hat4.learn_one(xi1,yi1)

    # Predict the test set
    for xi, yi in stream.iter_pandas(X_test, y_test):
        # The four base learners predict the labels
        xi = selector1.transform_one(xi)
        xi = selector2.transform_one(xi)
        y_pred1= hat1.predict_one(xi) 
        y_prob1= hat1.predict_proba_one(xi) 
        
        hat1.learn_one(xi,yi)

        y_pred2= hat2.predict_one(xi) 
        y_prob2= hat2.predict_proba_one(xi)
        
        hat2.learn_one(xi,yi)

        y_pred3= hat3.predict_one(xi) 
        y_prob3= hat3.predict_proba_one(xi)
        hat3.learn_one(xi,yi)

        y_pred4= hat4.predict_one(xi) 
        y_prob4= hat4.predict_proba_one(xi)
        hat4.learn_one(xi,yi)
        
        if y_pred1 == yi:
            metric_w1.append(0)
        else:
            metric_w1.append(1)
        if y_pred2 == yi:
            metric_w2.append(0)
        else:
            metric_w2.append(1)
        if y_pred3 == yi:
            metric_w3.append(0)
        else:
            metric_w3.append(1)
        if y_pred4 == yi:
            metric_w4.append(0)
        else:
            metric_w4.append(1)
        
        # Record their real-time accuracy
        metric1 = metric1.update(yi, y_pred1)
        metric2 = metric2.update(yi, y_pred2)
        metric3 = metric3.update(yi, y_pred3)
        metric4 = metric4.update(yi, y_pred4)    

        
        # Calculate the real-time window error rates of four base learners
        if i<1000:
            e1 = 0
            e2 = 0
            e3 = 0
            e4 = 0
        else:        
            e1 = sum(metric_w1[round(0.9*i):i])/len(metric_w1[round(0.9*i):i])
            e2 = sum(metric_w2[round(0.9*i):i])/len(metric_w1[round(0.9*i):i])
            e3 = sum(metric_w3[round(0.9*i):i])/len(metric_w1[round(0.9*i):i])
            e4 = sum(metric_w4[round(0.9*i):i])/len(metric_w1[round(0.9*i):i])

        
        ep = 0.001 # The epsilon used to avoid dividing by 0
        
        # Calculate the weight of each base learner by the reciprocal of its window real-time error rate
        ea = 1/(e1+ep)+1/(e2+ep)+1/(e3+ep)+1/(e4+ep)
        w1 = 1/(e1+ep)/ea
        w2 = 1/(e2+ep)/ea
        w3 = 1/(e3+ep)/ea
        w4 = 1/(e4+ep)/ea

        # Make ensemble predictions by the classification probabilities
        if  y_pred1 == 1:
            ypro10=1-y_prob1[1]
            ypro11=y_prob1[1]
        else:
            ypro10=y_prob1[0]
            ypro11=1-y_prob1[0]
        if  y_pred2 == 1:
            ypro20=1-y_prob2[1]
            ypro21=y_prob2[1]
        else:
            ypro20=y_prob2[0]
            ypro21=1-y_prob2[0]
        if  y_pred3 == 1:
            ypro30=1-y_prob3[1]
            ypro31=y_prob3[1]
        else:
            ypro30=y_prob3[0]
            ypro31=1-y_prob3[0]
        if  y_pred4 == 1:
            ypro40=1-y_prob4[1]
            ypro41=y_prob4[1]
        else:
            ypro40=y_prob4[0]
            ypro41=1-y_prob4[0]        

        # Calculate the final probabilities of classes 0 & 1 to make predictions
        y_prob_0 = w1*ypro10+w2*ypro20+w3*ypro30+w4*ypro40
        y_prob_1 = w1*ypro11+w2*ypro21+w3*ypro31+w4*ypro41
        
#         print(str(i)+" "+str(w1)+" "+str(w2)+" "+str(w3)+" "+str(w4)+" "+str(y_prob_0)+" "+str(y_prob_1))

        if (y_prob_0>y_prob_1):
            y_pred = 0
            y_prob = y_prob_0
        else:
            y_pred = 1
            y_prob = y_prob_1
        
        # Update the real-time accuracy of the ensemble model
        metric = metric.update(yi, y_pred)
        
        # Detect concept drift
        val = 0
        if yi != y_pred:
            val = 1 
        in_drift = eddm.update(float(val))
        # print(type(in_drift))
        if in_drift & (i>1000) :
            print(f"Change detected at index {i}")
            drift = 1 # indicating that a drift occurs
        
        # If a drift is detected
        if drift == 1: 
            x_new = X_test[round(0.9*i):i]
            y_new = y_test[round(0.9*i):i]
            
            # Relearn the online models on the most recent window data (representing new concept data)
            hat1 = ensemble.AdaptiveRandomForestClassifier(n_models=3) # ARF-ADWIN
            hat2 = neighbors.KNNClassifier(window_size=100)
            hat3 = ensemble.AdaptiveRandomForestClassifier(n_models=3,drift_detector=EDDM(),warning_detector=EDDM()) # ARF-EDDM
            hat4 = linear_model.PAClassifier() # SRP-DDM
            
            # Re-select features
            selector1 = feature_selection.VarianceThreshold(threshold = 0.1)
            selector2 = feature_selection.SelectKBest(similarity=stats.PearsonCorr(),k=40)
            
            for xj, yj in stream.iter_pandas(x_new, y_new):
                selector1 = selector1.learn_one(xj)
            for xj, yj in stream.iter_pandas(x_new, y_new):
                xj = selector1.transform_one(xj)
                selector2 = selector2.learn_one(xj, yj)      
            for xj, yj in stream.iter_pandas(x_new, y_new):
                xj = selector1.transform_one(xj)
                xa = selector2.transform_one(xj)
                hat1.learn_one(xa,yj)
                hat2.learn_one(xa,yj)
                hat3.learn_one(xa,yj)
                hat4.learn_one(xa,yj)
                
                if j ==1:
                    print(len(xa))
                    j=0
            drift = 0   
        
        j=1
        

        t.append(i)
        m.append(metric.get()*100)
        
        yt.append(yi)
        yp.append(y_pred)
        
        i=i+1
    
    # outputs
    print("Accuracy: "+str(round(accuracy_score(yt,yp),4)*100)+"%")
    print("Precision: "+str(round(precision_score(yt,yp),4)*100)+"%")
    print("Recall: "+str(round(recall_score(yt,yp),4)*100)+"%")
    print("F1-score: "+str(round(f1_score(yt,yp),4)*100)+"%")
    print(metric1.get()*100)
    print(metric2.get()*100)
    print(metric3.get()*100)
    print(metric4.get()*100)
    return t, m

data_list = ['hyperplane_100000.csv','SEA_21000.csv','au2_10000.csv','connect-4_binary.csv','covertype_binary.csv','poker-hand_binary.csv']
for i in data_list:
    data_name = i
    df = pd.read_csv(i)

    X = df.drop(['y'],axis=1)
    ##W-PWPEA requires negative data labels to be 0
    df['y'] = df['y'].replace(-1, 0)
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.001, test_size = 0.999, shuffle=False,random_state = 0)

        # Use the Adaptive Random Forest (ARF) model with ADWIN drift detector
    name1 = "ARF-ADWIN model"
    model1 = ensemble.AdaptiveRandomForestClassifier(n_models = 3, drift_detector = ADWIN()) # Define the model

    t, m1 = adaptive_learning(model1, X_train, X_test, y_train, y_test) # Learn the model on the dataset
    acc_fig(t, m1, name1) # Draw the figure of how the real-time accuracy changes with the number of samples

    a=np.array(m1)
    np.save('ARF-ADWIN'+" "+data_name+'.npy',a)


    # Use the Streaming Random Patches (SRP) model with ADWIN drift detector
    name3 = "SRP-ADWIN model"
    model3 = ensemble.SRPClassifier(n_models = 3, drift_detector = ADWIN()) # Define the model
    t, m3 = adaptive_learning(model3,  X_train, X_test, y_train, y_test) # Learn the model on the dataset
    acc_fig(t, m3, name3) # Draw the figure of how the real-time accuracy changes with the number of samples
    c=np.array(m3)
    np.save('SRP-ADWIN' +" "+data_name+'.npy',c)


    """comparison model learning"""
    # Use the Extremely Fast Decision Tree (EFDT) model 
    name5 = "EFDT model"
    model5 = tree.ExtremelyFastDecisionTreeClassifier() # Define the model
    t, m5 = adaptive_learning(model5,  X_train, X_test, y_train, y_test) # Learn the model on the dataset
    acc_fig(t, m5, name5) # Draw the figure of how the real-time accuracy changes with the number of samples
    e=np.array(m5)
    np.save('EFDT' +" "+data_name+'.npy',e)

    # Use the proposed Multi-Stage Automated Network Analytics (MSANA) model 

    # Select the four base online models, they can be changed based on the performance of the models
    bm1 = ensemble.AdaptiveRandomForestClassifier(n_models=3) # ARF-ADWIN
    bm2 = neighbors.KNNClassifier(window_size=100) # KNN-ADWIN
    bm3 = ensemble.AdaptiveRandomForestClassifier(n_models=3,drift_detector=EDDM(),warning_detector=EDDM()) # ARF-EDDM
    bm4 = linear_model.PAClassifier() # OPA

    name = "Proposed MSANA model"
    t, mm = MSANA(bm1, bm2, bm3, bm4, X_train, y_train, X_test, y_test) # Learn the model on the dataset
    acc_fig(t, mm, name) 
    a=np.array(mm)
    np.save('MSANA' +" "+data_name+'.npy',a)


"""
Active Weighted Aging Ensemble for drifted data stream classification. Information Sciences

[2] Woźniak, M., Zyblewski, P., & Ksieniewicz, P. (2023). Active Weighted Aging Ensemble for drifted data stream classification. Information Sciences, 630, 286-304.

https://github.com/w4k2/AWAE
"""
import numpy as np
import pandas as pd
import strlearn
from strlearn.evaluators import TestThenTrain
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from strlearn.ensembles import KUE,ROSE
from skmultiflow.trees import HoeffdingTreeClassifier
import random
import math

"""Online Active Learning Ensemble."""

class Active(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, treshold=0.5, budget=0.05, s=0.01, sigma=0.01,r=0.05,random_state=None):
        self.treshold = treshold
        self.budget = budget
        self.base_estimator = base_estimator
        self.random_state = random_state
        self.s = s
        self.sigma = sigma
        self.r = r
    def partial_fit(self, X, y, classes=None):
        np.random.seed(self.random_state)
        # First train
        counter = 0
        if not hasattr(self, "clf"):
            # Pierwszy chunk na pelnym
            try:
                self.clf = clone(self.base_estimator).partial_fit(X, y, classes=classes)
            except:
                self.clf = self.base_estimator.partial_fit(X, y, classes=classes)
            self.usage = []

        else:
            ##Each chunk does the initial training randomly
            idx = np.random.choice(X.shape[0], size=math.floor(self.r*X.shape[0]), replace=False)
            self.clf.partial_fit(X[idx], y[idx], classes=np.unique(y))
            X_delete = np.delete(X, idx, axis=0)
            y_delete = np.delete(y, idx, axis=0)
            ##active learning strategy
            for i in range(X_delete.shape[0]):

                supports = np.abs(self.clf.predict_proba(X_delete[i].reshape(1,X_delete[i].shape[0]))[:, 0] - self.clf.predict_proba(X_delete[i].reshape(1,X_delete[i].shape[0]))[:, 1])  
                if supports < self.treshold:
                    self.clf.partial_fit(X_delete[i].reshape(1,X_delete[i].shape[0]), np.array([y_delete[i]]), classes)  
                    counter+=1
                    self.treshold *= (1 - self.s)
                elif (random.random()<self.sigma):
                    self.clf.partial_fit(X_delete[i].reshape(1,X_delete[i].shape[0]), np.array([y_delete[i]]), classes)
                    counter+=1


    def predict(self, X):
        return self.clf.predict(X)



class OALE(ClassifierMixin, BaseEnsemble):
    def __init__(self, base_estimator=None, n_estimators=10, tiny=1,
                  norm_strategy='b'):
        """Initialization."""
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.tiny = tiny
        self.norm_strategy = norm_strategy

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []
            self.counter_ = 0
            self.weights = np.zeros(self.n_estimators+1)
            self.weights[0] = .5

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")

        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Append new estimators
        if self.X_.shape[0]>1:
            # print("self.X_.shape[0]",self.X_.shape[0])
            for i in range(2 if len(self.ensemble_) == 0 else 1):
                self.ensemble_.append(clone(self.base_estimator))

        # Train all this shit
        [clf.partial_fit(self.X_, self.y_, self.classes_)
         for clf in self.ensemble_]

        # Remove the oldest dynamic when ensemble becomes too large
        if len(self.ensemble_) > self.n_estimators+1:
                del self.ensemble_[1]

        if self.counter_ < self.n_estimators:
            self.weights[self.counter_+1] = 1/self.n_estimators
            self.weights[1:self.counter_+1] = self.weights[1:self.counter_+1] * (1-(1/self.n_estimators))

            # if self.counter_ > 1:
            if self.norm_strategy == 'a':
                self.norm = self.weights[1:] - np.min(self.weights[1:])
                self.norm = self.norm / (np.max(self.weights[1:])-np.min(self.weights[1:]))
                self.norm = np.append([[.5]], self.norm)
            elif self.norm_strategy == 'b':
                # Alternative
                self.norm = np.copy(self.weights)
                rrr = self.norm[1:]
                self.norm[1:] = rrr / (np.sum(rrr)*2)
            elif self.norm_strategy == 'c':
                # Alternative
                self.norm = np.linspace(.5,1,10)

        if self.norm_strategy == 'c':
            # Alternative
            self.norm = np.random.uniform(size=self.weights.shape)

        if self.X_.shape[0]>1:
            self.counter_ += 1
        return self

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) * self.norm[i]
                         for i, member_clf in enumerate(self.ensemble_)])

    def predict_proba(self, X):
        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        return average_support

    def predict(self, X):
        """
        Predict classes for X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : array-like, shape (n_samples, )
            The predicted classes.
        """

        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        # Return prediction
        return self.classes_[prediction]
    

## Use the synthesized data set after adding noise
stream_sea = strlearn.streams.CSVParser("SEA_21000.csv",chunk_size=500, n_chunks=40)
stream_hyper = strlearn.streams.CSVParser("hyperplane_100000.csv",chunk_size=500, n_chunks=200)
stream_au2 = strlearn.streams.CSVParser("au2_10000.csv",chunk_size=500, n_chunks=20)
stream_connect = strlearn.streams.CSVParser("connect-4_binary.csv",chunk_size=500, n_chunks=122)
stream_cov = strlearn.streams.CSVParser("covertype_binary.csv",chunk_size=500, n_chunks=990)
stream_poker = strlearn.streams.CSVParser("poker-hand_binary.csv",chunk_size=500, n_chunks=1893)
# construct classifier
for stream in [stream_sea,stream_hyper,stream_au2,stream_connect,stream_cov,stream_poker]:
    clf_oale = Active(OALE(base_estimator=HoeffdingTreeClassifier(split_criterion='hellinger')))
    ttt = TestThenTrain(metrics=[accuracy_score], verbose=True)
    ttt.process(stream, [clf_oale])
    accuracy_array = ttt.scores.reshape(-1)
    np.save('OALE'+" "+str(stream)+" accuracy",accuracy_array)
    ##Calculated average accuracy
    def average_accuracy(accuracy_array):
        ave_accuracy = np.array([])
        for i in range(accuracy_array .shape[0]):
            if i == 0:
                ave_accuracy = np.append(ave_accuracy,accuracy_array [i])
            else:
                ave_accuracy = np.append(ave_accuracy,np.sum(accuracy_array[:i+1])/(i+1))
        return ave_accuracy
    np.save('OALE'+" "+str(stream)+' average accuracy',average_accuracy(accuracy_array))


stream_sea = strlearn.streams.CSVParser("SEA_21000.csv",chunk_size=1000, n_chunks=20)
stream_hyper = strlearn.streams.CSVParser("hyperplane_100000.csv",chunk_size=1000, n_chunks=100)
stream_au2 = strlearn.streams.CSVParser("au2_10000.csv",chunk_size=1000, n_chunks=10)
stream_connect = strlearn.streams.CSVParser("connect-4_binary.csv",chunk_size=1000, n_chunks=61)
stream_cov = strlearn.streams.CSVParser("covertype_binary.csv",chunk_size=1000, n_chunks=495)
stream_poker = strlearn.streams.CSVParser("poker-hand_binary.csv",chunk_size=1000, n_chunks=946)

clf_rose = ROSE(base_estimator=HoeffdingTreeClassifier())
clf_kue = KUE(base_estimator=HoeffdingTreeClassifier())
clf_list = [clf_kue, clf_rose]
clf_name = ['KUE','ROSE']
for stream in [stream_sea,stream_hyper,stream_au2,stream_connect,stream_cov,stream_poker]:
    for clf in clf_list:
        i=0
        ttt = TestThenTrain(metrics=[accuracy_score], verbose=True)
        ttt.process(stream, [clf])
        accuracy_array = ttt.scores.reshape(-1)
        np.save(clf_name[i]+" "+str(stream)+" accuracy",accuracy_array)
        ##Calculated average accuracy
        def average_accuracy(accuracy_array):
            ave_accuracy = np.array([])
            for i in range(accuracy_array .shape[0]):
                if i == 0:
                    ave_accuracy = np.append(ave_accuracy,accuracy_array [i])
                else:
                    ave_accuracy = np.append(ave_accuracy,np.sum(accuracy_array[:i+1])/(i+1))
            return ave_accuracy
        np.save(clf_name[i]+" "+str(stream)+' average accuracy',average_accuracy(accuracy_array))
        i+=1