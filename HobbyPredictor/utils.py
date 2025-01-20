import torch

def ConvertWordToVector(x,keyValues):
    return [keyValues[t] for t in x]

def accuracy(y_pred,y_label):
    TotalNum = y_label.shape[0]
    correct = (torch.eq(torch.argmax(y_pred,dim=1),torch.argmax(y_label,dim=1))).sum().item()
    return correct/TotalNum *100.0