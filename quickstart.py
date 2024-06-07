import streamlit as st
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
from st_audiorec import st_audiorec
import pickle


df = pd.read_csv("data_mc.csv")
df = df.drop(columns=['Unnamed: 0'])

x = df.drop(columns = ['class'])
y = df['class']

x = x.to_numpy()
y = y.to_numpy()

def get_report(y_test, y_pred):
    report = classification_report(np.concatenate(y_test), np.concatenate(y_pred), output_dict= True)
    print(report)
    return report

def invoke(func):
    return func()

def SVC_linear():
    return SVC(kernel = 'linear')

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
report = get_report(y_test, y_pred)

words = {'1': 'temperatura', '2': 'tarea', '3': 'avisos', '4': 'alemania', '5': 'fotografiar', '6': 'onda', '7': 'mes', '8': 'rascar'}
report = pd.DataFrame(report).T
disp_report = report.rename(index = words)
disp_report = disp_report.drop(columns = ['support', 'f1-score'])
disp_report = disp_report.drop(['weighted avg', 'macro avg', 'accuracy'])


# Set the title of the application
st.title("Clasificación de audios de voz")

# Display the trained words
words = ['temperatura', 'tarea', 'avisos', 'alemania', 'fotografiar', 'onda', 'mes', 'rascar']
st.write("Palabras entrenadas: ", ', '.join(words))
st.dataframe(disp_report)

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')

    # Save the audio data as a .obj file
    with open('audio_data.obj', 'wb') as audio_data_file:
        pickle.dump(wav_audio_data, audio_data_file)

