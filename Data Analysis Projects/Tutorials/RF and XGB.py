import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/Developer/.spyder-py3/archive/HR_comma_sep.csv")
df.shape
print(df.columns)
df.head(10)
column = "Department"

categorical_variables = ["Department","salary"]
target_variable = ["left"]
numeric_variables = list(set(df.columns.values) - set(categorical_variables) - set(target_variable))

def label_encoder(df,column):
    le = preprocessing.LabelEncoder()
    df[column] = le.fit_transform(df[column])
    ohe = preprocessing.OneHotEncoder()
    temp_array = ohe.fit_transform(df[[column]]).toarray()
    column_names = [column+"_"+str(m) for m in le.classes_]
    return(pd.DataFrame(data = temp_array, columns = column_names))

new_df = df[numeric_variables]
for column in categorical_variables:
    new_df = pd.concat([new_df,label_encoder(df,column)],axis=1)
new_df.shape
new_df.columns

#split into test and train
X, X_test, y, y_test = train_test_split(new_df,df[target_variable],test_size = 0.3)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X,y)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, f1_score, roc_auc_score
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

df["left"].value_counts()

precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)
roc_curve(y_test, y_pred)
fpr,tpr,thresholds = roc_curve(y_test, y_pred)

import matplotlib.pyplot as plt
plt.plot(fpr,tpr,"b")
plt.plot([0,1],[0,1],"r-")

roc_auc_score(y_test, y_pred)

#Cross Validation
from sklearn.model_selection import KFold, cross_val_score
model = LogisticRegression()

cv = KFold(n_splits=10)

for train_index, test_index in cv.split(X):
    model.fit(X.iloc[train_index],y.iloc[train_index,-1])
    print(model.score(X.iloc[train_index],y.iloc[train_index,-1]))
    print(model.score(X.iloc[test_index],y.iloc[test_index,-1]))


cross_val_score(model,X,y["left"],cv=10)

#Hyperparameter tunning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
C = uniform(loc=0, scale=4)

parameters = {"C":[0.001,0.01,0.1,0.5,1,5,10,100]}
parameters_2 = {"C":C}

cv_model = GridSearchCV(model, parameters)
cv_model.fit(X,y["left"])
cv_model.best_estimator_
cv_model.best_score_
cv_model.best_params_

cv_random = RandomizedSearchCV(model,param_distributions = parameters_2)
cv_random.fit(X,y["left"])
cv_random.best_estimator_
cv_random.best_score_

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 1000,max_depth = 6,min_samples_split = 15)
model.fit(X,y["left"])
model.score(X,y["left"])
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)