from libraries import *

#Metriche
def metrics(test_y,y_pred,label):
    conf_matrix= confusion_matrix(test_y, y_pred, labels=label)
    TP=conf_matrix[0][0]
    TN=conf_matrix[1][1]
    FP=conf_matrix[1][0]
    FN=conf_matrix[0][1]
    Sensitivity_TPR=TP/(TP+FN)
    Specificity_TNR=TN/(TN+FP)
    FPR=FP/(TN+FP)
    FNR=FN/(TP+FN)
    Precision=TP/(TP+FP)
    Recall=TP/(TP+FN)
    F=(2*Recall*Precision)/(Recall+Precision)
    print(f"Accuracy : {compute_accuracy(y_pred, test_y)}\nSensitivity : {round(Sensitivity_TPR,2)}\nSpecificity : {round(Specificity_TNR,2)}\nFPR : {round(FPR,2)}\nFNR : {round(FNR,2)}\nPrecision: {round(Precision,2)}\nRecall : {round(Recall,2)}\nF-Score : {round(F,2)}")

def compute_accuracy(pred_y, test_y):
    return (pred_y == test_y).sum() / len(pred_y)