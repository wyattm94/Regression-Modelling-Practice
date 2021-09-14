
"""
Supervised Learning
    > Inferring a function/model from supervised (labelled) - the training data
    > The goal is to predict the values of unlabelled data - the testing data
    > For example, determine the type of fruit from characteristics such as size, color and shape

    > 2 Major types:
        >> Regression: Used for continuous numerical values (quantitative information)
        >> Classification: Used for categorical values (factors) which can be numerical or character based be binary or multiclass (more than 2 values)

    > Common Algorithms:
        >> Lazy Learning: KNN (K-Nearest Neighbor)
            + Data points are given a class/value of majority based on its nearest neighbors (other data points close to it)
            + We can specify a distance calculation (the most common is euclidean) as well as how many neighboring values it is close too
            + The output is class (group) membership

        >> Support Vector Machines: Classifications/Regressions
            + This is a discriminative classifier which uses separating hyperplanes (basically dividing lines) instead of groups
            + The goal is to create optimal hyperplanes to categorize new examples - the line should provide a maximum separation between data points in different classes
            + Multi-class output is achieved using kernels for high dimension input space
                ++ Kernel functions transform data to make them linearly separable (using straight lines)
                ++ Some common function are: Radial Bias Functions (RBF), Gaussian and Polynomial
            + For SVM regression the same principles apply but outputs are continuous numerical values

        >> Tree Based/Ensemble Classifiers: Decision Trees, Random Forests and Gradient Boosting
            + Single decision trees use features to find logical paths to responses - but single trees are not that reliable
            + Ensemble Tree methods follow the rationale that many 'weak' learners (single trees) can be combined to create 1 'strong' learner
            + Take the results of multiple trees and choose the best results democratically - using a voting classifier (votes, weighted-confidence and/or highest confidence)


        + Logistic Regression: Used for binary repsonses

"""

# Import resources
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Read training and testing data for models (NOTICE: Testing data lacks the 'Survived' column because we don't yet know the response for testing)
train = pd.read_csv('data/trainT.csv')
test = pd.read_csv('data/testT.csv')

"""
Generally, we would use 80% of data fpr training and 20% of data for testing (splitting the data accordingly before model building)
"""

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# Let's test this on the diabetes data set (Load and format - extract the response y)
columns = "age sex bmi map tc ldl hdl tch ltg glu".split()
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=columns)
y = diabetes.target

# Create the training and testing data set
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

"""
Depending on the amount of data available, we can have unstable results on smaller data sets - some argue that this is the equivalent of testing on the same data set (not accurate when splitting)

We can fix this by using cross-validation to subdivide the data into multiple 'folds' (even distribution of cases over the entire data set)
    > One fold will be used for testing while the others will be used for training
    > Use different combinations of folds for testing and training, creating a much more diverse training/testing protocol 
    > Calculate the average and standard deviation (variance ^ 2) of all folds' test results - use the average accuracy (usually)

"""

from sklearn.model_selection._validation import cross_val_score
from sklearn.model_selection._split import KFold

# Load the Iris data set
iris = datasets.load_iris()
data_input = iris.data
data_output = iris.target

# Create K number of folds for Cross Validation (CV) - create the splitting object - one can build a custom engine for this
kf = KFold(5, shuffle=True) #5 fold KV

for train_set, test_set in kf.split(data_input):
    print(train_set, test_set)

"""
Notes on accuracy assessment

    > Using training data we 'train' models and test their prediction accuracy on test data
    > The goal is to maximize the generality of the model (be an accurate model across multiple training/testing sets)
    
    > We want to prevent overfitting - when a model tries to describe too much of the underlying noise and variability instead of just the core behavior
        >> This occurs, generally, as models become more complex or try to solve vastly more complex problems
    
    > Hold-out test data and cross validation can help detect overfitting 

Best practices for determining accuracy of models

    > R-Squared Score:
        >> The coefficient of determination - similar to the r^2 used in linear regression models
        >> Computed between the actual and predicted responses (0-1) - the higher the value, the better the model is as predicting the response
    
    > Mean Squared Error (MSE):
        >> (1/n) * SUM( (y_i - yhat_i)^2 ) where i goes from 1 to n 
        >> This measures the quality of the ML model - the closer to 0 the better
    
    > Accuracy Score:
        >> In multi-label classification this function computes the subset accuracy - the set of labels predicted for a sample must match EXACTLY the set of actual Y labels
        >> This is a confusion matrix - we evaluate the accuracy for each type of category individually in a matrix (table)
    
    > Precision:
        >> The measure of a classifiers exactness
        >> True positives divided by the number of all positives (true and false)
    
    > Recall:
        >> The measure of a classifiers completeness
        >> True positives divided by the number of true positives + false negatives
    
    > F1 Score:
        >> A balance between precision and recall
        >> 2 * ( (Precision*Recall) / (Precision+Recall) )

"""

# Import additional resources for model evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

# Load and extract data - we are trying to predict the median value for homes using the boston housing data set (Used for ALL following models)
data = datasets.load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
Y = pd.DataFrame(data.target, columns=['MEDV'])

# Create training and testing data sets (Used for ALL following models)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=25) # 20% Data hold-put for testing

"""
#1 Random Forest Regression
"""

# Load RF resources
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Build the tree model
rf_regr = RandomForestRegressor(max_depth=2, random_state=0)
rf_regr.fit(x_train, y_train)

# Extract and print the importance of each feature (determined by the RF)
rf_importances = rf_regr.feature_importances_
print(rf_importances)

# Now, identify WHICH features are most important (by name, ranked by level of importance)
rf_indices = np.argsort(rf_importances)[::-1]

for feat in range(X.shape[1]):
    print(f"{feat+1}. Feature {rf_indices[feat]} ({rf_importances[rf_indices[feat]]})")

# Now, predict using the model
rfy_pred = rf_regr.predict(x_test)

# Evaluate model performance
mean_squared_error(y_test, rfy_pred)    # MSE was 33.85 (not too useful alone, but useful when COMPARING multiple models)
r2_score(y_test, rfy_pred)              # R^2 of 0.5036 (About 50% of variability captured - not the best prediction)

"""
> Trying different combinations of features (generally only the most important) can assist in creating a better model
    >> To do this, use subsets of feature columns from the previous dataset
"""

"""
#2 Support Vector Machine Regression (SVM)
"""

# Import some additional resources
from sklearn.svm import SVR

# Create and fit the model
svm_model = SVR()
svm_model.fit(x_train, y_train)

# Predict using the model
svm_y_pred = svm_model.predict(x_test)

# Evaluate the performance of the prediction model
mean_squared_error(y_test, svm_y_pred)  # MSE was 60.31 (Almost double the error from the RF model)
r2_score(y_test, svm_y_pred)            # R^2 of 0.1560 (About 16% of variability captured - a poor prediction)

"""
#3 KNN Regression
"""

# Import some additional resources
from sklearn.neighbors import KNeighborsRegressor

# Create and fit the model
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(x_train, y_train)

# Predict using the model
knn_y_pred = knn_model.predict(x_test)

# Evaluate the performance of the prediction model
mean_squared_error(y_test, knn_y_pred)  # MSE was 45.16 (Better than SVM but not better than RF)
r2_score(y_test, knn_y_pred)            # R^2 of 0.3379 (About 34% of variability captured - better than SVM but still worse than RF)

"""
> Changing the feature sets and number of k neighbors can assist in better model production
"""

"""
#4 Gradient Boosting Regression
"""

# Import some additional resources
from sklearn.ensemble import GradientBoostingRegressor

# Create and fit the model
gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=3) # Estimators is the number of trees to be modeled, at a max depth of 3
gb_model.fit(x_train, y_train)

# Predict using the model
gb_y_pred = gb_model.predict(x_test)

# Evaluate the performance of the prediction model
mean_squared_error(y_test, gb_y_pred)  # MSE was 7.45 (Clearly better than all previous models)
r2_score(y_test, gb_y_pred)            # R^2 of 0.8906 (About 89% of variability captured - best model so far)

# We can also analyze the feature importances (boosting is related to tree models - we are boosting the forest)
gb_importances = gb_model.feature_importances_
print(gb_importances)

"""
#5 Multi-Layer Perceptron (MLP) Regression - Extension of ANN/DNN Neural Networks
"""

