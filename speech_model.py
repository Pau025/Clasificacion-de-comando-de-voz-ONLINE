import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm

df1 = pd.read_csv("data_mc.csv")


df2 = pd.read_csv("data_gaby.csv")
df3 = pd.read_csv("data_itzelt.csv")

df = pd.concat([df1, df2])
df = pd.concat([df, df3])
df = df.drop(columns=['Unnamed: 0'])
print(df)
x = df.drop(columns = ['class'])
y = df['class']
print(y)
x = x.to_numpy()
y = y.to_numpy()


#-----------------------------------------------
#Validacion cruzada en 100 k-pliegues
#-----------------------------------------------

def invoke(func):
    return func()

def get_report(y_test, y_pred):
    print(classification_report(np.concatenate(y_test), np.concatenate(y_pred)))

def SVC_linear():
    return SVC(kernel = 'linear')

def SVC_rbf():
    return SVC(kernel = 'rbf')

def LDA():
    return LinearDiscriminantAnalysis()

def KNN5():
    return KNeighborsClassifier(n_neighbors=5)

def MLP(): #red neuronal de dos capas de 10 neuronas cada una
    return MLPClassifier(hidden_layer_sizes= (10, 10), max_iter= 10000)

def RFC():
    return RandomForestClassifier(max_depth=3, random_state=0)

def StratifiedKFold_Validator(clasificador, x, y):
    n_folds = 10
    kf = StratifiedKFold(n_splits = n_folds, shuffle = True) 

    cv_y_test = []
    cv_y_pred = []

    for train_index, test_index in kf.split(x, y):
        x_train = x[train_index, :]
        y_train = y[train_index]
        
        clf = invoke(clasificador)
        clf.fit(x_train, y_train)
    
        x_test = x[test_index, :]
        y_test = y[test_index]
        y_pred = clf.predict(x_test)
    
        cv_y_test.append(y_test)
        cv_y_pred.append(y_pred)
    
    return cv_y_test, cv_y_pred

#-------------------------------------------------------------------------
# EVALUACION DE MODELOS DE CLASIFICACION
#-------------------------------------------------------------------------

#SVM linear
print("Evaluación de Modelo de Clasificación SVM lineal")
y_test, y_pred = StratifiedKFold_Validator(SVC_linear, x, y)
get_report(y_test, y_pred)

#SVM radial
print("Evaluación de Modelo de Clasificación SVM radial")
y_test, y_pred = StratifiedKFold_Validator(SVC_rbf, x, y)
get_report(y_test, y_pred)

#Linear Discrminant Analysis
#print("Evaluación de Modelo de Clasificación Linear Discriminant Analysis (LDA)") #SVD sometimes doesnt converge
#y_test, y_pred = StratifiedKFold_Validator(LDA, x, y)
#get_report(y_test, y_pred)

#KNN with 10 nearest neighbors
print("Evaluación de Modelo de Clasificación K Nearest Neighbors (5)") 
y_test, y_pred = StratifiedKFold_Validator(KNN5, x, y)
get_report(y_test, y_pred)

#MLPClassifier with 2 layers
print("Evaluación de Modelo Red Neuronal de dos capas") 
y_test, y_pred = StratifiedKFold_Validator(MLP, x, y)
get_report(y_test, y_pred)


#-----------------------------------------------------------------------
# Modelos no vistos en clase
#-----------------------------------------------------------------------
print("Evaluación de Modelo Random Forest Classifier con tres niveles de profundidad") 
y_test, y_pred = StratifiedKFold_Validator(RFC, x, y)
get_report(y_test, y_pred)