
"""
> Polynomial Regression
    >> Used when there is not a linear relationship but something curved (non-linear)
    >> In these cases, one may need to use polynomial parameters (coefficients that are squared or raised to a higher power)
"""

# Import resources
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

from sklearn import linear_model
from sklearn.metrics import r2_score

# Load the data
iris = sns.load_dataset('iris')

# Extract info
x = iris.sepal_length
y = iris.petal_length

# Create a linear model object
lr = linear_model.LinearRegression()

# Now, fit multiple models where the exponent (power) varies from 1 to 5 for the current feature coefficient (Sepal Length)
for exp in [1,2,3,4,5]:

    # Fit the model and predict 'y' using the new model
    lr.fit(np.vander(x, exp+1), y)
    y_predict = lr.predict(np.vander(x, exp+1))

    # Plot the results
    plt.plot(x, y_predict, label=f'power of {exp}')
    plt.legend(loc=2)

    print(r2_score(y, y_predict)) # Report the fit (r^2)

plt.plot(x, y, 'ok') # Show the final combined plot (all fits against data)
plt.show()

"""
> We can see that an exponent (order) of 4 seems to have the best fit by the r^2 metric 
> We can determine this by not only the largest value, but the amount of change between each metric per new order
"""

"""
> Another situation arises where the residuals may not be normally distributed - which means some behaviour has NOT been captured by our model

> We then turn to GLM (General Linear Models) for things like:
    >> Proportions (rates) - the variance will be an almost n shaped function of the mean
    >> Count data - the variance will often increase with the mean
    >> Binary responses - (only 2 choices such as living/dead, 1/0, etc...)
    
> As an example, we use flexible generalization of ordinary regression to explain NON-NORMAL RESPONSE DATA
    >> [General Formula] y_ijk = a + b1x1 + b2x2 + b3x1x2 + e_ijk
    >> Checking for external influences on the change in the response variable(s)

> GLM generalizes linear functions using a 'link' function
    >> This allows a wide array of distributions (normal, Poisson, binomial, gamma, etc...)
    >> The link function transforms Y to linearize its values --> ni = g(yi)

> The 2 most common models are:
    >> Logistic Regression - used for cases where the response is binary (0 or 1) - error structure is assumed to be binomial
    >> Poisson Regression - used for count data - error structure is assumed to be poisson

"""

# Logistic Regression

"""
> Logit 
    >> 1) Linear relationship
    >> 2) Y must NOT be categorical
    >> 3) Logarithmic transformation is used to express the non-linear relationship 
    >> 4) Errors are independent (and DO NOT need to be normally distributed)

> We use deviance to compare models
    >> The probability of success = p/(1-p)
"""
# Read the data into python
data = pd.read_csv('data/trainT.csv') # Titantic Data Set

# Check shape and for amount of null (missing) data
data.shape
data.isnull().sum()

# Create a subset of the data set and drop NA data (drops all rows with any NA data)
new_data = data[['Survived', 'Pclass', 'Age', 'Fare']]
new_data = new_data.dropna()

# Plot survival data (yes or no)
plt.figure(figsize=(5,5))
fig, ax = plt.subplots()
data.Survived.value_counts().plot(kind='barh', color='blue', alpha=0.65) # Use internal plotting tools (pandas) to add to figure (fig)
ax.set_ylim(-1, len(data.Survived.value_counts()))
plt.title("Survival Breakdown (1 = Survived, 0 = Died")

plt.show() # More people died

# Use boxplots to visualize the data more granularly
sns.factorplot(x="Pclass", y="Fare", hue="Survived", data=data, kind="box")
plt.show()

"""
> We can see from this plot that people in first class tended to have a much higher survival rate than other passengers
"""

# Now we will model this data using logistic regression
y = new_data[['Survived']]
x = new_data[['Pclass', 'Age', 'Fare']]

logit = sm.Logit(y, x.astype(float))
result = logit.fit()
print(result.summary())

"""
> Looking at the pseudo-R^2 metric, we see this model only explains about 10% of the total variance in the data (very low)
> The formula derived from this would be:
    >> Log [p/(1-p)] = -0.28*Pclass + 0.0146*Fare - 0.01*Age
"""

# To get the ODDS of survival, we must take the exponential value of this formula
print(np.exp(result.params))

"""
> The "Odds" state that the odds of death increase by a factor of 'p' for each unit change in that feature
> The probability is then: prob = odds / (1 + odds)
"""

from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
import statsmodels.discrete.discrete_model as smd

# ETL new data
data2 = pd.read_csv('data/trainT.csv')
new_data2 = data2[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
new_data2 = new_data2.dropna()

# Create the variables (c indicated categorical variable)
y, X = dmatrices('Survived ~ C(Pclass) + C(Sex) + Age + Fare', new_data2, return_type = 'dataframe')

# Create and fit the model
model = LogisticRegression(fit_intercept=False, C=1e9)
model_fit = model.fit(X, y)
print(model_fit.coef_)

logit = sm.Logit(y, X)
print(logit.fit().params)

# Fit the model and check the results
result = logit.fit()
print(result.summary())

"""
> This model captures about 33% of the variance - not great but much better than before
> We can also see that Fare is NOT significant in this new model
"""

