import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


df = pd.read_csv('ecommerce.csv')
# print(df.head())
print(df.info())
print(df.describe())

# EDA
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df, alpha=0.5)
plt.show()

sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df, alpha=0.5)
plt.show()

sns.pairplot(df, kind = 'scatter', plot_kws={'alpha': 0.4})
plt.show()

sns.lmplot(x= "Length of Membership",
           y = 'Yearly Amount Spent',
           data = df,
           scatter_kws={'alpha':0.3})
plt.show()