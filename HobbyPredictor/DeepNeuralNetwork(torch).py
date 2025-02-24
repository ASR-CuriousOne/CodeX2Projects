import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split

from utils import accuracy,ConvertWordToVector
from NeuralNetwork import HobbyPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 7000
LEARNING_RATE = 0.01
DECAY = 0.01
SHOW = 100
keyValuesForHobbies = {"Academics": [[1],[0],[0]],
                       "Arts":[[0],[1],[0]],
                       "Sports":[[0],[0],[1]]}

#DataLoading

df_test = pd.read_csv("Data/processed_test_data.csv")
df_train = pd.read_csv("Data/processed_training_data.csv")

X_data,Y_data = [],[]

for column in df_train:    
    if(column == "Unnamed: 0" or column == "Predicted Hobby"):
        continue
    else:
        X_data.append((df_train[column]).values)

Y_data = ConvertWordToVector(df_train["Predicted Hobby"])
X_data = np.array(X_data)

X_data = torch.tensor(X_data,dtype=torch.float32).to(DEVICE).T
Y_data = torch.tensor(Y_data,dtype=torch.float32).to(DEVICE).squeeze(2)

X_train,X_test,Y_train,Y_test = train_test_split(X_data,Y_data,test_size=0.3)


HobbiesPredictor = HobbyPredictor().to(DEVICE)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(HobbiesPredictor.parameters(),lr=LEARNING_RATE,weight_decay=DECAY)

training_losses,testing_losses,training_accuracies,testing_accuracies = [],[],[],[]

for epoch in range(EPOCHS):
    HobbiesPredictor.train()

    y_logits = HobbiesPredictor(X_train)
    y_pred  = (torch.softmax(y_logits,dim=1))

    loss = loss_fn(y_logits,Y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    
    if(epoch % SHOW == 0):
        HobbiesPredictor.eval()
        test_logits = HobbiesPredictor(X_test)
        test_pred = (torch.softmax(test_logits,dim=1))

        test_loss = loss_fn(test_pred,Y_test)

        training_losses.append(loss.to("cpu"))
        training_accuracies.append(accuracy(y_pred=y_pred,y_label=Y_train))
        testing_losses.append(test_loss.to("cpu"))
        testing_accuracies.append(accuracy(y_pred=test_pred,y_label=Y_test))

        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Correct: {accuracy(y_pred,Y_train):.5f} | Test_Loss: {test_loss:.5f} | Test_Accuracy: {accuracy(test_pred,Y_test):.5f}")


HobbiesPredictor.eval()

with torch.inference_mode():
    plt.plot(testing_accuracies)
    plt.plot(training_accuracies)
    plt.legend(["testing accuracy", "training accuracy"])
    plt.show()
    