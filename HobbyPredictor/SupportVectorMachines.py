from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df_test = pd.read_csv("Data/processed_test_data.csv")
df_train = pd.read_csv("Data/processed_training_data.csv")

keyValuesForHobbies = {"Academics": [1],
                       "Arts":[0],
                       "Sports":[-1]}

def ConvertWordToVector(x):
    return [keyValuesForHobbies[t] for t in x]

def accuracy(y_pred,y_label):
    TotalNum = y_label.shape[0]
    correct = (np.equal(y_pred,y_label)).sum().item()
    return correct/TotalNum *100.0


X_data,Y_data = [],[]

for column in df_train:    
    if(column == "Unnamed: 0" or column == "Predicted Hobby"):
        continue
    else:
        X_data.append((df_train[column]).values)

Y_data = np.array(ConvertWordToVector(df_train["Predicted Hobby"]))
X_data = np.array(X_data).T

X_train,X_test,Y_train,Y_test = train_test_split(X_data,Y_data,test_size=0.1)

X_train = np.squeeze(X_train)
Y_train = np.squeeze(Y_train)
X_test = np.squeeze(X_test)
Y_test = np.squeeze(Y_test)

model_SVC = svm.SVC()
model_NuSVC = svm.NuSVC()
model_LinearSVC = svm.LinearSVC()



model_SVC.fit(X_train,Y_train)
model_NuSVC.fit(X_train,Y_train)
model_LinearSVC.fit(X_train,Y_train)

y_pred_SVC = model_SVC.predict(X_test)
y_pred_NuSVC = model_NuSVC.predict(X_test)
y_pred_LinearSVC = model_LinearSVC.predict(X_test)

print(f"      SVC: Accuracy:  {accuracy(y_pred=y_pred_SVC,y_label=Y_test):.4f}%")
print(f"    NuSVC: Accuracy:  {accuracy(y_pred=y_pred_NuSVC,y_label=Y_test):.4f}%")
print(f"LinearSVC: Accuracy:  {accuracy(y_pred=y_pred_LinearSVC,y_label=Y_test):.4f}%")

