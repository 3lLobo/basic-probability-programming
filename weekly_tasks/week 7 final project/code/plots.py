"""
Simple plots.
"""

from sklearn import datasets ## imports datasets from scikit-learn
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

data = datasets.load_boston() ## loads Boston dataset from datasets library 

print(data.DESCR)

# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])

def linear_model(predictors, target):
    """
    :param predictors: an n k array of predictors (n - nobs; k - the number of predictors)
    :param target: an n array representing the dependent variable (what is predicted)
    """
    model = sm.OLS(target, predictors).fit()

    return model.summary()

#checking a linear model
#linear_model(X, y)

#create the dataset for all plotting
plotted_data = df.assign(MEDV=target)

#Example1 - scatterplot of two variables

#create the plot
sns_plot = sns.jointplot(x="RM", y="MEDV", data=plotted_data)

#name the axis in the way you want
sns_plot.set_axis_labels(xlabel="No. of rooms", ylabel="House price")

#save the plot as a png figure
sns_plot.savefig("example1.png")

#Example2 - adding regression line

#create the plot
sns_plot = sns.lmplot(x="RM", y="MEDV", data=plotted_data)

#save the plot as a png figure
sns_plot.savefig("example2.png")

#Example3 - pairplots

#subset to interesting pairs and plot scatterplot for all pairs
sns_plot = sns.pairplot(plotted_data[["RM", "AGE", "MEDV"]])

sns_plot.savefig("example3.png")

#Example4 - noticing the non-linear trend
sns_plot = sns.jointplot(x="DIS", y="MEDV", data=plotted_data)

sns_plot.set_axis_labels(xlabel="Distance", ylabel="House price")

sns_plot.savefig("example4.png")


#The non-linear trend is confirmed in models below: R-squared improves if we log-transform the DIS variable; cf model_basic and model_logtr

model_basic = linear_model(df["DIS"], target)

print("Model using the raw distance")
print(model_basic)

print("Model using the log-transformed distance")
model_logtr = linear_model(np.log(df["DIS"]), target)
print(model_logtr)

#restart figure
sns.FacetGrid(plotted_data)

#Example 5 - categorical variable
ax = sns.stripplot(x="CHAS", y="MEDV", data=plotted_data)

sns_plot = ax.figure

sns_plot.savefig("example5.png")

#Example 6 - categorical variable - better
ax = sns.stripplot(x="CHAS", y="MEDV", data=plotted_data, jitter=True)


sns_plot = ax.figure


sns_plot.savefig("example6.png")

