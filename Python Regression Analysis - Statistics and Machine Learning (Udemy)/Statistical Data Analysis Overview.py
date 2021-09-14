"""
What is statistical analysis?
> Collection of data about a population (samples) to draw inferences from to gain insight about relationships, significance and predictability of outcomes in the population
> Statistical modeling is how we approx. reality about a population
> Collecting enough quality data is ver important - you need enough to represent the population (~10%) and you must make it diverse without to much bias (use randomness)
> A parameter is a characteristic of a population
> A statistic is a characteristic of a sample of a population
"""

### PART I: Descriptive Statistics
import pandas as pd
from sklearn import datasets

# Load and format data into a DataFrame
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Generate a table of descriptions across each column - you can access this individually as well with named methods
iris_description = iris_df.describe()


### PART II: Group and summarize data by categories

# Load data into DF
uni_df = pd.read_csv("data/cwurData.csv")

# Group by country and get basic stats info (creates a new column for each type of stat. for each column - many columns)
by_country = uni_df.groupby("country")
by_country_info = by_country.describe()

# Counts by 2 group paramters
g1 = uni_df.groupby(['country', 'influence']).count()

# Create a new DF with the counts of influencial schools by country - created a new name for the column using .reset_index()
influence_counts = pd.DataFrame(by_country.size().reset_index(name="InfluenceAggregate"))

# Load another data set into python
rain_df = pd.read_csv('data/rainfall1901_2015.csv')

# Get the mean rainfall by each subdivision
mean_rainfall = rain_df.groupby('SUBDIVISION').mean()


### PART III: Visualize Descriptice Statistics-Boxplots (using iris)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Create a boxplot
iris_df.boxplot()

# Create another boxplot using seaborn
iris_sns = sns.load_dataset('iris')
sns.boxplot(x="species", y='sepal_length', data=iris_sns)

"""
Some notes on distributions:
> Standard Deviation refers to how far data are dispersed from the mean
> The mean is the average while the median is the center
> A normal distribution is shaped like a bell
    >> Distributions are important because they help determine significance of outcomes
    >> 68% of data lie within 1 STD of the mean
    >> 95% of data lie within 2 STD of the mean
    >> 99.7% of data lie within 3 STD of the mean
"""

### PART IV: Checking for normally distributed data
iris_sepal_length = iris_sns["sepal_length"]

# Show a histogram
iris_sepal_length.hist()

# Use a Shapiro-Wilks Test to test for normality - the null hypothesis is that the data IS normally distributed (accept or reject the null hypothesis)
import scipy
import scipy.stats
shapiro_results = scipy.stats.shapiro(iris_sepal_length)

"""
The p-value must be > 0.05 to accept the null hypothesis, else we reject it --> indication the data may NOT be normally distributed (we CANNOT conclude normality)
> In this test, the pvalue = 0.0108 so we reject the null hypothesis

---

Standard-Normal Distribution:
> Mean 0 and STD of 1
> A distribution made up of Z-scores, a standard calculation to compare values' distance from the mean
> The formula is z = (x - mu)/std
> Basically, a z-score can tell you how many standard deviations from the mean a value is --> this is how teachers curve??

---

Confidence intervals:
> Indicate how much uncertainty there is in a sample statistic
> The goal is to compute how confident we can be that results reflect the population parameters we are trying to estimate
> 95% CI is used to show how much variability will appear in repeated measures
    >> This means that after repeating experiments across multiple samples, 95% of the intervals will contain the population mean
> We calculate the sample statistic as well as the Margin of Error (MOE)
> We can see this on a distribution in the TAIL ends of the distribution (the z value of 0.95 is ~ 1.96)
> 95% of values fall within 2 STDs of the mean
> Because stats are estimates of parameters, CIs help to capture the true value of the statistic in the range of possible error
    >> The % used (95, 97, 99) states the chances of covering the true value 
    >> CI = Mean +- MOE
    >> MOE = Z * (STD/sqrt(n)) where n is the number of data points (observations)

---

Student T Distributions:
> Similar to a normal distribution but is used when there is not much data to use
> Shorter with fatter tails

"""

### PART V: Confidence Intervals (CI)
import random, math

# Set a random seed to keep random number consistent (non-changing for each run)
np.random.seed(10)

# Create a poisson distribution of "people weights" (numbers)
population_weight = scipy.stats.poisson.rvs(loc=18, mu=35, size=150000)

# Extract a sample of the values and calculate the mean of that sample
sample_size = 1000
sample = np.random.choice(a=population_weight, size=sample_size)
sample_mean = sample.mean()

# Calculate the z-critical value
z_critical = scipy.stats.norm.ppf(q=0.975)
print(f"> The z-critical value = {z_critical}") # The value should be ~1.96

# Calculate the STD and MOE
pop_std = population_weight.std()
moe = z_critical * (pop_std/math.sqrt(sample_size))

# Finally, calculate the CI
ci = (sample_mean - moe, sample_mean + moe)
print(f"> The confidence interval = {ci}")
