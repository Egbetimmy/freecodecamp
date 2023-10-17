import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Developer/.spyder-py3/Bike-Sharing-Dataset/day.csv")

df.shape
df.head(10)
df.columns

X = df["temp"].values.reshape(-1, 1)
Y = df["cnt"].values.reshape(-1, 1)
#Total count(sales) as a function of temp
#independent = Temp
#Dep = saleas (cnt)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

# fit and predict

# y = mx + c
lm.fit(X, Y)

print(lm.intercept_)
print(lm.coef_)
print(lm.predict([[0.15]]))

plt.scatter(df["temp"], df["cnt"])

# y = m0 + m1x1 +...+ mnxn
lm.fit(df[["temp","hum"]], df["cnt"].values.reshape(-1, 1))

print(lm.intercept_)
print(lm.coef_)

plt.scatter(df["hum"], df["cnt"])

print(df[["temp","hum"]].head(10))

print(lm.predict([[0.34, 0.55]]))
