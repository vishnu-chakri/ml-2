import pandas as pd
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import time
import scipy
from scipy.stats import friedmanchisquare

#loading the data set
dataset = pd.read_csv("spambase.data")

# taking all the columns and rows except the last column which is spam or not spam
attribute_list = dataset.iloc[:,:-1].values

#  taking all the columns and rows only the column which is spam or not spam(0 or 1)
class_list = dataset.iloc[:,-1].values

skf = StratifiedKFold(n_splits=10)
skf.split(attribute_list, class_list)

time_list = []
acc_list = []
f1_list = []
c = 0

for train_index, test_index in skf.split(attribute_list, class_list):
    attribute_train, attribute_test = attribute_list[train_index], attribute_list[test_index]
    class_train, class_test = class_list[train_index], class_list[test_index]

    time_list.append([])
    acc_list.append([])
    f1_list.append([])


    LR = LogisticRegression()
    t = time.time()
    LR.fit(attribute_train, class_train)
    t = time.time() - t
    time_list[c].append(t)
    acc_list[c].append(accuracy_score( class_test, LR.predict(attribute_test)))
    f1_list[c].append(f1_score(class_test, LR.predict(attribute_test)))

    KNN = KNeighborsClassifier()
    t = time.time()
    KNN.fit(attribute_train, class_train)
    t = time.time() - t
    time_list[c].append(t)
    acc_list[c].append(accuracy_score(class_test, KNN.predict(attribute_test)))
    f1_list[c].append(f1_score(class_test, KNN.predict(attribute_test)))

    SVM = svm.SVC()
    t = time.time()
    SVM.fit(attribute_train, class_train)
    t = time.time() - t
    time_list[c].append(t)
    acc_list[c].append(accuracy_score(class_test, SVM.predict(attribute_test)))
    f1_list[c].append(f1_score(class_test, SVM.predict(attribute_test)))

    c = c+1

print(" ")
print("Accuracy values in all folds")
for i in range(0,10):
    print (' {:1d}        {:0.4f}           {:0.4f}          {:0.4f}'.format(i+1, acc_list[i][0], acc_list[i][1], acc_list[i][2]))

print(" ")
print("F-measure vlaues in all folds")
for i in range(0,10):
    print (' {:1d}        {:0.4f}           {:0.4f}                {:0.4f}'.format(i+1, f1_list[i][0], f1_list[i][1], f1_list[i][2]))


print(" ")
print("training time values in all folds ")
for i in range(0,10):
    print (' {:1d}        {:0.4f}           {:0.4f}                {:0.4f}'.format(i+1, time_list[i][0], time_list[i][1],time_list[i][2]))

x,y,z = [],[],[]
for i in range(0,10):
    x.append(acc_list[i][0])
    y.append(acc_list[i][1])
    z.append(acc_list[i][2])

f_acc, p = friedmanchisquare(x, y, z)
print(("average accuracy for LR: "+str(sum(x)/10)))
print(("average accuracy for KNN: "+str(sum(y)/10)))
print(("average accuracy for SVM: "+str(sum(z)/10)))
print(("Fredmans accuracy for accuracy is "+str(f_acc)))
print(" ")

x,y,z = [],[],[]
s = 0

for i in range(0,10):
    x.append(f1_list[i][0])
    y.append(f1_list[i][1])
    z.append(f1_list[i][2])

f_f1, p = friedmanchisquare(x, y, z)
print(("average f1-score for LR: "+str(sum(x)/10)))
print(("average f1-score for KNN: "+str(sum(y)/10)))
print(("average f1-score for SVM: "+str(sum(z)/10)))
print(("Fredmans statistic for f1-score is "+str(f_f1)))
print(" ")

a,b,c = [],[],[]
for i in range(0,10):
    x.append(time_list[i][0])
    y.append(time_list[i][1])
    z.append(time_list[i][2])
f_t, p = friedmanchisquare(x, y, z)
print(("average training time for LR: "+str(sum(x)/10)))
print(("average training time for KNN: "+str(sum(y)/10)))
print(("average training time for SVM: "+str(sum(z)/10)))
print("Fredmans statistic for time is "+str(f_t))



