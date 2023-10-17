import pandas as pd
df = pd.read_csv("archive/HR_comma_sep.csv")

print(df.shape, df.columns)
print(df['left'].unique())

# y : left
# x : satisfaction_level

from sklearn.linear_model import LogisticRegression

train = df.iloc[:10000,:]
test = df.iloc[10001:,:]

lor = LogisticRegression()
lor.fit(train["satisfaction_level"].values.reshape(-1,1), train["left"])

print(lor.predict(test["satisfaction_level"].values.reshape(-1,1)))

print(lor.predict_proba(test["satisfaction_level"].values.reshape(-1,1)))

print(lor.intercept_)
print(lor.coef_)

print(df.columns)

lor.fit(train[["satisfaction_level","number_project"]], train["left"])

print(lor.predict(test[["satisfaction_level","number_project"]]))

print(lor.intercept_)
print(lor.coef_)
