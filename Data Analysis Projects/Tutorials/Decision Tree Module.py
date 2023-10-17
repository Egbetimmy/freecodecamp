import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
df = pd.read_csv("C:/Users/Developer/.spyder-py3/archive/HR_comma_sep.csv")
export_graphviz(dt, feature_names=["salary2","Work_accident"], out_file=("C:/Users/Developer/.spyder-py3/archive/test.dot"))
print(df.shape)
print(df.columns)

df["Work_accident"].unique()
df["Work_accident"] = df["Work_accident"].astype("category")

df.dtypes

df["salary"].unique()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["salary2"] = le.fit_transform(df["salary"])

df["salary2"] = df["salary2"].astype("category")

dt = DecisionTreeClassifier()

dt.fit(df[["salary2","Work_accident"]], df["left"])

dt.predict(df[["salary2","Work_accident"]])

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
temp = ohe.fit_transform(df[["salary2"]]).toarray()
print(temp)

column_names = ["salary_"+x for x in le.classes_]
print(column_names)

temp = pd.DataFrame(temp, columns = column_names)
print(temp)
temp.head(10)

df2 = pd.concat([df,temp], axis = 1)

print(df2.head(10))
df2.shape
