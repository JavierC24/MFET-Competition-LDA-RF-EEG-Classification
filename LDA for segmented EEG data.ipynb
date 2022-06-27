import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import itertools


#Import of Data
allPeoples = scipy.io.loadmat('x1TrainingData.mat')

#Training and Testing Sets
X_train = np.transpose(allPeoples['P10_TData'])
y_train = np.transpose(allPeoples['P10_TLabel'])
X_test = np.transpose(allPeoples['P10_TestData'])
y_test = np.transpose(allPeoples['P10_TestLabel'])

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#LDA - Feature Map Extraction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(solver='eigen',shrinkage=0.05,n_components=3)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

#Classification
classifier = RandomForestClassifier(max_depth = 30)

classifier.fit(X_train, y_train)
y_predic = classifier.predict(X_test)
y_prob_predic = classifier.predict_proba(X_test)


#Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true= y_test, y_pred=y_predic)

num_classes = 5;
def plot_confusion_matrix(cm, num_classes,
                           normalize=False,
                           title='Confusion Matrix',
                           cmap=plt.cm.Blues):
     plt.imshow(cm, interpolation='nearest', cmap=cmap,vmax=100,vmin=0)
     plt.title(title)
     plt.colorbar()
     tick_marks = np.arange(len(num_classes))
     plt.xticks(tick_marks, num_classes, rotation=45)
     plt.yticks(tick_marks, num_classes)
   
     if normalize:
         cm = 100*cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
         print("Normalized Confusion matrix")
     else:
         print("Confusion matrix, without normalization")
     print(cm)
     thresh = cm.max()/2.
     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       
         plt.text(j, i, cm[i,j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
     plt.tight_layout()
     plt.ylabel('True Label')
     plt.xlabel('Predicted Label')
    
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})    
cm = np.round(100*cm.astype('float')/cm.sum(axis=1)[:, np.newaxis],2)

 
cm_plot_labels = ['supination', 'pronation','hand open','palmar grasp','lateral grasp']
plot_confusion_matrix(cm=cm, num_classes=cm_plot_labels, title='Confusion Matrix')
import sklearn.metrics as met
qq = met.accuracy_score(y_test,y_predic)
print(f"The accuracy of the model is {round(qq,3)*100} %")
#ROC
y_test[y_test ==  8] = 0
y_test[y_test ==  9] = 1
y_test[y_test ==  11] = 2
y_test[y_test == 925] = 3
y_test[y_test ==  926] = 4

plt.figure()

from sklearn.metrics import roc_auc_score
rocaucscore = roc_auc_score(y_test,y_prob_predic, multi_class='ovr', average='weighted')
print('ROC-AUC-score', rocaucscore)

fpr = dict()
tpr = dict()
thresh = dict()
roc_auc = dict()
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score
num_classes = 5
for i in range(num_classes):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_prob_predic[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "violet"])
for i, color in zip(range(num_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=2,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')