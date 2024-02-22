import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import pylab
import scipy.stats as stats



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


df = pd.read_csv('ecommerce.csv')
print(df.head())
print(df.info())
print(df.describe())

# EDA
# sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df, alpha=0.5)
# plt.show()

sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df, alpha=0.5)
# plt.show()

sns.pairplot(df, kind = 'scatter', plot_kws={'alpha': 0.4})
# plt.show()

sns.lmplot(x= "Length of Membership",
           y = 'Yearly Amount Spent',
           data = df,
           scatter_kws={'alpha':0.3})
# plt.show()


X = df[['Avg. Session Length', 'Time on App', 'Time on Website','Length of Membership',]]
y = df['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print(X_train)

# Training the model
lm = LinearRegression()
lm.fit(X_train, y_train)
# print(lm.coef_)
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coef'])
print(cdf)


# Predictions
predictions = lm.predict(X_test)
print(predictions)

print('Mean Absolute Error: ', mean_absolute_error(y_test, predictions))
print('Mean Squared Error: ', mean_squared_error(y_test, predictions))
print('RMSE: ', math.sqrt(mean_squared_error(y_test, predictions)))


# Residuals
residuals = y_test - predictions

sns.displot(residuals, bins = 30, kde =True)
plt.show()


stats.probplot(residuals, dist='norm', plot=pylab)
pylab.show()