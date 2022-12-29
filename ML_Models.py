from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


dataset = pd.read_csv('adult.csv')
print(dataset.shape)
dataset['workclass'].replace('?', np.nan, inplace=True)
dataset.dropna(subset=['workclass'], inplace=True)
dataset['education'].replace('?', np.nan, inplace=True)
dataset.dropna(subset=['education'], inplace=True)
dataset['marital.status'].replace('?', np.nan, inplace=True)
dataset.dropna(subset=['marital.status'], inplace=True)
dataset['occupation'].replace('?', np.nan, inplace=True)
dataset.dropna(subset=['occupation'], inplace=True)
dataset['relationship'].replace('?', np.nan, inplace=True)
dataset.dropna(subset=['relationship'], inplace=True)
dataset['race'].replace('?', np.nan, inplace=True)
dataset.dropna(subset=['race'], inplace=True)
dataset['sex'].replace('?', np.nan, inplace=True)
dataset.dropna(subset=['sex'], inplace=True)
dataset['native.country'].replace('?', np.nan, inplace=True)
dataset.dropna(subset=['native.country'], inplace=True)
print(dataset.shape)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(y)
print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

x_train[:, [0, 2, 4, 10, 11, 12]] = imputer.fit_transform(
    x_train[:, [0, 2, 4, 10, 11, 12]])

x_test[:, [0, 2, 4, 10, 11, 12]] = imputer.transform(
    x_test[:, [0, 2, 4, 10, 11, 12]])

columns_list = [1, 3, 5, 6, 7, 8, 9, 13]
ct_obj = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(sparse=False), columns_list)], remainder="passthrough")

x_train = np.array(ct_obj.fit_transform(x_train))
x_test = np.array(ct_obj.transform(x_test))
print(x_train.shape)
print(x_test.shape)


y_train = pd.DataFrame(y_train, columns=['0'])
y_train['0'].replace('>50K', 'Yes', inplace=True)
y_train['0'].replace('<=50K', 'No', inplace=True)

y_test = pd.DataFrame(y_test, columns=['0'])
y_test['0'].replace('>50K', 'Yes', inplace=True)
y_test['0'].replace('<=50K', 'No', inplace=True)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

lb_obj = LabelEncoder()
y_train = lb_obj.fit_transform(y_train)
y_test = lb_obj.transform(y_test)

sc_obj = StandardScaler(with_mean=False)
x_train[:, :] = sc_obj.fit_transform(x_train[:, :])
x_test[:, :] = sc_obj.transform(x_test[:, :])

print(x_train.shape)
print(y_train.shape)
classifier = SVC(C=1, kernel='linear', random_state=0, probability=True)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap=plt.cm.OrRd)
plt.xlabel("Predicted labels")
plt.ylabel("Actual labels")
plt.show()

print("Accuracy SVM: ", accuracy_score(y_test, y_pred))
print("Recall SVM: ", recall_score(y_test, y_pred))
print("Precision SVM: ", precision_score(y_test, y_pred))
print("F1-score SVM: ", f1_score(y_test, y_pred))

# Plotting the ROC curve and calculating the AUC
# get false and true positive rates
classifier.probability = True
y_pred_prob = classifier.predict_proba(x_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
# get area under the curve
roc_auc = auc(fpr, tpr)
print("AUC SVM: ", roc_auc)
# plot ROC curve
plt.plot(fpr, tpr, color='green')
plt.title('ROC Curve ')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.show()

accuracy = (5293+1064)/(5293+1064+356+828)
recall = (1064)/(1064+828)
precision = (1064)/(1064+356)
print("Accuracy SVM: ", accuracy)
print("Recall SVM: ", recall)
print("Precision SVM: ", precision)
print("F1-score: SVM", 2*(precision*recall/(precision+recall)))


classifier = DecisionTreeClassifier(
    criterion='entropy', max_depth=20, random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap=plt.cm.OrRd)
plt.xlabel("Predicted labels")
plt.ylabel("Actual labels")
plt.show()

# Caluclate the performance metrics
print("Accuracy DTC: ", accuracy_score(y_test, y_pred))
print("Recall DTC: ", recall_score(y_test, y_pred))
print("Precision DTC: ", precision_score(y_test, y_pred))
print("F1-score DTC: ", f1_score(y_test, y_pred))
# Plotting the ROC curve and calculating the AUC
# get false and true positive rates
y_pred_prob = classifier.predict_proba(x_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
# get area under the curve
roc_auc = auc(fpr, tpr)
print("AUC DTC: ", roc_auc)
# plot ROC curve
plt.plot(fpr, tpr, color='green')
plt.title('ROC Curve ')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.show()

accuracy = (5087+1151)/(5087+1151+741+562)
recall = (1151)/(1151+741)
precision = (1151)/(1151+562)
print("Accuracy DTC: ", accuracy)
print("Recall DTC: ", recall)
print("Precision DTC: ", precision)
print("F1-score DTC: ", 2*(precision*recall/(precision+recall)))
