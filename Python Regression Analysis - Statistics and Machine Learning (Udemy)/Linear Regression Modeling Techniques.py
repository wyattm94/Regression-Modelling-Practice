"""
Regression analysis in Python
> Exploring the relationship between 2 quantitative variables
> Correlations quantifies magnitude/direction of a relationship
    >> NOTE: Correlation DOES NOT mean causation
> Regression modeling is building a formalized relationship between the response Y and the feature/predictor X
"""
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Data
iris = sns.load_dataset("iris")

# Check correlation of data within the entire DF
iris_corr = iris.corr()

# Create a scatterplot of sepal length and width
plt.scatter(
    iris['sepal_length'], iris['petal_width'],
    marker='.',
    color='black',
    alpha=0.7,
    s=100
)
plt.title("Sepal length vs. Sepal width")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

# Create a colorized correlation matrix plot using statsmodels
from statsmodels import api as sm
sm.graphics.plot_corr(iris_corr, xnames=list(iris_corr.columns))
plt.show()

"""
The theory of linear regression:
> Used for modeling quantitative dependency
    >> Helps answers the question: How does a change in X influence Y?
> The formula is Y = a + bX + e where e is some irreducible error
> The objective is to create a line of BEST FIT between all data points
    >> This is done using the Ordinary Least Squares (OLS)
    >> The line also must minimize the sum of squared errors
> The null hypothesis is that Y is independent of X (Variations in X do NO influence Y)
    >> This implies the slope (b coefficient) is 0 
> Multiple Linear Regression involves this over multiple X features
> Performance is assess using R^2 (for multiple regression we use adjusted R^2)
    >> The closer R^2 is to 1, the better the fit (0 is a very bad fit)
> We also must evaluate to ensure the pvalue of the slope is < 0.05 (statistically significant to reject the null hypothesis)
"""

# Import additional resources to implement linear regression
import pandas as pd
import numpy as np
import scipy as stats
from sklearn import linear_model

# Select predictor (X) and response (Y) to determine relationship of petal length and width
X = iris['petal_length']
Y = iris['petal_width']

# Create and fit the model
model = sm.OLS(Y, X)
result = model.fit()
print(result.summary())

"""
> The R^2 tells us what % of Y's behavior is explained by the feature X
> The pvalue for each coefficient tells us how statistically significant each feature is
> The AIC and BIC criterion can be used to compare across multiple models to find the best one
    >> This is coupled with R^2 as well as std errors for coefficients as well as the model as a whole
> The Jacque Bera test is used to assess normality of data (see if skewness and kurtosis fit normal distribution params)
    >> In this case, it refers to the residuals of the model
    >> We could also use Q-Q plots to visually check for normality
    >> The null hypothesis is that the data IS normally distributed (rejecting the null hypothesis means it may not be)
    >> Values close to 0 support normality while values away from 0 do not
    >> Also, using the Prob(JB) tells us if we can accept or reject the null (< 0.05 means reject)
    >> Check these sites out for some more info on JB tests in python
        + https://learn.co/lessons/dsc-01-10-14-regression-diagnostics
        + https://www.statology.org/jarque-bera-test-python/ 
> Kurtosis for a normal distribution is about 3
> Skew for a normal distribution should be about 0
> Also, the Prob (F-statistic) is the pvalue for the model as a whole
> Documentation walk-through:
    >> https://www.statsmodels.org/devel/examples/notebooks/generated/regression_diagnostics.html

"""

# Next we add a constant to create an intercept coefficient in the model
X = iris['petal_length']
X = np.vander(X, 2)
Y = iris['petal_width']

# Create and fit the model
model = sm.OLS(Y, X)
result = model.fit()
print(result.summary())

"""
> Now we see an additional coefficent added to the model and there is 1 less residual 
    >> No. Observations = 150, Df Residuals = 148
> This new model also shows that we cannot reject the null hypothesis which states the residuals are normally distributed
"""

# Now we try multiple linear regression (multiple features)
X = iris[['petal_length', 'sepal_length']]
X = sm.add_constant(X)
Y = iris['petal_width']

# Create and fit the model
model = sm.OLS(Y, X)
result = model.fit()
print(result.summary())

"""
> Here we see that the intercept coefficient is NOT statistically significant (can be ignored in final functional form)
> We also see that while both features are statistically significant, petal_length is vastly more important than sepal_length
> We can also reject the null hypothesis of normally distributed residuals
"""

# Finally, we use many features to model the respone (Y) - first we create dummy variables for species categories
dummies = pd.get_dummies(iris["species"])
iris = pd.concat([iris, dummies], axis=1)

# Then split up the new DF into features and response variables
X = iris[['petal_length', 'sepal_length', 'setosa', 'versicolor', 'virginica']]
X = sm.add_constant(X)
Y = iris['petal_width']

# Create and fit the model
model = sm.OLS(Y, X)
result = model.fit()
print(result.summary())

"""
> In this model, we can see that some coefficients are NOT significant and we can reject normal residuals
> In general, models with the lowest AIC/BIC scores are generally better
    >> When assessing a model, we want to capture as much info as possible without adding too much variance to the residuals 
    >> This means not too much error/uncertainty --> This makes a model less reliable in the real world
> Eigen Values:
    >> In regression analysis, these measure the spread (variance) in the direction defined by the new axis
    >> If variance is too small, trends cannot be computed accurately 
    >> This is basically how much effect the variable had on shifting the function
    >> Very low values could indicate multicollinearity
        >> This is basically having 2 or more features that are closely related
        >> Therefore they contribute little new information while adding variance in residuals (more uncertainty than information)

---

Some assumptions for building LRM:
> X and Y have some kind of linear relationship (linear scatter plot, etc...)
> Errors (residuals) are normally distributed
> Errors (residuals) are independent (no autocorrelation between errors)
    >> This is something we check for heavily in time series analysis 
    >> Autocorrelations are effects that data has on other data points in the same series/set
    >> Basically, it states that the value of an error in some way DEPENDS on a previous error value
        + This means there is some behavior that HAS NOT been modeled away (is still present but not captured/understood by the model)
        + A model has not MODELED data if some pattern/behavior still exists that is not captured --> problems down the road 
> Error variance should be constant (the variance does not change - we have homoscedasticity)
    >> In time series analysis we test for heteroscedasticity, where variance is changing
> Avoid multicollinearity between predictors
    >> You want to capture as much info as possible without creating too much residual variance (model uncertainty potential)
    >> 'Sometimes less is more' has never been so true
"""

# Now, we will use some tools for checking LRM conditions in python
sns.pairplot(iris[['petal_width', 'petal_length', 'sepal_length']].dropna(how='any', axis=0)) # Show feature relationships
plt.show()

# These plots show there is clearly some sort of linear relationship present between the data

"""
Now, what questions do we need to ask and how can we tell what the answer is? (Extension from earlier comment blocks)
---
> [1] Are the residuals normally distributed?
    >> Using the JB test - NOTE: low numbers of observations can cause this test to state non-normal incorrectly

"""

# Check residuals visually with a Q-Q plot
residuals = result.resid
sm.qqplot(residuals) # Shows a majority of residuals are normally distributed (by linear line) - this can help clarify a JB test
plt.show()

"""
> [2] Is there multicollinearity?
    >> This states that coefficients (predictors/features) are correlated 
    >> We can use a Breusch-Pagen test for homoscedasticity (variance does not depend on auxiliary regressors)
"""
import statsmodels.stats.diagnostic as ssd
from statsmodels.compat import lzip
name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
test = ssd.het_breushpagan(result.resid, result.model.exog) # Issues with implementing this method
lzip(name, test)

# If p-value is < 0.05 we reject the null hypothesis which is homoscedasticity (no variance changes in residuals)

"""
> [3] Is there influence being contributed by outliers?
    >> Influence tests help us determine is outliers are generating influence on the slope of the regression line
"""
from statsmodels.graphics.regressionplots import *
plot_leverage_resid2(result)
plt.show()
influence_plot(result)
plt.show()