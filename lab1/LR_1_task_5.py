import pandas as pd

df = pd.read_csv('data_metrics.csv')
print(df.head())

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
print(df.head())

from sklearn.metrics import confusion_matrix
print("\nconfusion_matrix function:")
print(confusion_matrix(df.actual_label.values, df.predicted_RF.values))

#----------------------

def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

print('TP:', find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_TN(df.actual_label.values, df.predicted_RF.values))

#------------------------

import numpy as np
def find_conf_matrix_values(y_true,y_pred):
# calculate TP, FN, FP, TN
    TP = find_TP(y_true,y_pred)
    FN = find_FN(y_true,y_pred)
    FP = find_FP(y_true,y_pred)
    TN = find_TN(y_true,y_pred)
    return TP,FN,FP,TN

def lukashevych_confusion_matrix(y_true, y_pred):
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return np.array([[TN,FP],[FN,TP]])

print("\nlukashevych_confusion_matrix function:")
print(lukashevych_confusion_matrix(df.actual_label.values, df.predicted_RF.values))
print(lukashevych_confusion_matrix(df.actual_label.values, df.predicted_RF.values))


assert np.array_equal(lukashevych_confusion_matrix(df.actual_label.values,df.predicted_RF.values),
                      confusion_matrix(df.actual_label.values, df.predicted_RF.values)), \
    'lukashevych_confusion_matrix() is not correct for RF'
assert np.array_equal(lukashevych_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                      confusion_matrix(df.actual_label.values, df.predicted_LR.values)), \
    'lukashevych_confusion_matrix() is not correct for LR'

#--------------------------

from sklearn.metrics import accuracy_score
print("\naccuracy_score:")
print(accuracy_score(df.actual_label.values, df.predicted_RF.values))


def lukashevych_accuracy_score(y_true, y_pred):
# calculates the fraction of samples
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return (TP + TN)/(TP + TN + FP + FN)


assert lukashevych_accuracy_score(df.actual_label.values, df.predicted_RF.values) == \
       accuracy_score(df.actual_label.values, df.predicted_RF.values), \
    'lukashevych_accuracy_score failed on RF'
assert lukashevych_accuracy_score(df.actual_label.values, df.predicted_LR.values) == \
       accuracy_score(df.actual_label.values, df.predicted_LR.values), \
    'lukashevych_accuracy_score failed on LR'

print("\nlukashevych_accuracy_score:")
print('Accuracy RF: %.3f'%(lukashevych_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f'%(lukashevych_accuracy_score(df.actual_label.values, df.predicted_LR.values)))

#-------------------------

from sklearn.metrics import recall_score
print("\nrecall_score:")
print(recall_score(df.actual_label.values, df.predicted_RF.values))



def lukashevych_recall_score(y_true, y_pred):
# calculates the fraction of positive samples predicted correctly
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return TP / (TP + FN)

assert lukashevych_recall_score(df.actual_label.values, df.predicted_RF.values) == \
       recall_score(df.actual_label.values, df.predicted_RF.values),\
    'lukashevych_recall_score failed on RF'

assert lukashevych_recall_score(df.actual_label.values, df.predicted_LR.values) == \
       recall_score(df.actual_label.values, df.predicted_LR.values), \
    'lukashevych_recall_score failed on LR'

print("\nlukashevych_recall_score:")
print('Recall RF: %.3f'%(lukashevych_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f'%(lukashevych_recall_score(df.actual_label.values, df.predicted_LR.values)))

#----------------------------

from sklearn.metrics import precision_score
print("\nprecision_score:")
print(precision_score(df.actual_label.values, df.predicted_RF.values))


def lukashevych_precision_score(y_true, y_pred):
# calculates the fraction of predicted positives samples that are actually positive
    TP,FN,FP,TN = find_conf_matrix_values(y_true,y_pred)
    return TP / (TP + FP)

assert lukashevych_precision_score(df.actual_label.values, df.predicted_RF.values) == \
       precision_score(df.actual_label.values, df.predicted_RF.values), \
    'lukashevych_precision_score failed on RF'
assert lukashevych_precision_score(df.actual_label.values, df.predicted_LR.values) == \
       precision_score(df.actual_label.values, df.predicted_LR.values), \
    'lukashevych_precision_score failed on LR'

print("\nlukashevych_precision_score:")
print('Precision RF: %.3f'%(lukashevych_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision LR: %.3f'%(lukashevych_precision_score(df.actual_label.values, df.predicted_LR.values)))

from sklearn.metrics import f1_score
print("\nf1_score:")
print(f1_score(df.actual_label.values, df.predicted_RF.values))

def lukashevych_f1_score(y_true, y_pred):
# calculates the F1 score
    recall = lukashevych_recall_score(y_true,y_pred)
    precision = lukashevych_precision_score(y_true,y_pred)
    return 2 / (1 / precision + 1 / recall)
    # return (2 * precision * recall) / (precision + recall)

# print(lukashevych_f1_score(df.actual_label.values, df.predicted_RF.values))
# print(f1_score(df.actual_label.values, df.predicted_RF.values))
# assert lukashevych_f1_score(df.actual_label.values, df.predicted_RF.values) == \
#        f1_score(df.actual_label.values, df.predicted_RF.values), \
#     'lukashevych_f1_score failed on RF'

# print(lukashevych_f1_score(df.actual_label.values, df.predicted_LR.values))
# print(f1_score(df.actual_label.values, df.predicted_LR.values))
assert lukashevych_f1_score(df.actual_label.values, df.predicted_LR.values) == \
       f1_score(df.actual_label.values, df.predicted_LR.values), \
    'lukashevych_f1_score failed on LR'

print("\nlukashevych_f1_score:")
print('F1 RF: %.3f' % (lukashevych_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f' % (lukashevych_f1_score(df.actual_label.values, df.predicted_LR.values)))

#----------------------------

print('\nscores with threshold = 0.5')
print('Accuracy RF: %.3f'%(lukashevych_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f'%(lukashevych_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f'%(lukashevych_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f'%(lukashevych_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')
print('scores with threshold = 0.25')
print('Accuracy RF: %.3f'%(lukashevych_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f'%(lukashevych_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f'%(lukashevych_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f'%(lukashevych_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))

#-----------------------------

from sklearn.metrics import roc_curve
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

import matplotlib.pyplot as plt
plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF')
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR')
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#-----------------------------

from sklearn.metrics import roc_auc_score
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('\nAUC RF: %.3f'% auc_RF)
print('AUC LR: %.3f'% auc_LR)

plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF AUC: %.3f'%auc_RF)
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR AUC: %.3f'%auc_LR)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()