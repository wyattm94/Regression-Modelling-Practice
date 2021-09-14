"""
Key Packages (Modules)
> Numpy         : Numerical Python, used for mathematical operations on multi-dimensional arrays
> Pandas        : Allows for reading and manipulating of tabular data - used DataFrames
> Matplotlib    : Default (basic) data visualization tools
> Seaborn       : More advanced Data Visualization (Many packages for this type of work exist including plotly)
> Scipy         : Builds on Numpy for scientific computation (linear algebra, interpolation, etc...) + stats
> Statsmodels   : Complement to Scipy for statistical computations (descriptive, estimations, regression, hypothesis, etc...)
> Scikit-Learn  : Machine Learning (Conventional ML techniques)
> H20           : Framework for basic deep learning
"""

# First off, we dig into Pandas (Series and DataFrames)
import pandas as pd

# Creating series
myDict = {'a':1, 'b':2, 'c':3}
my_series = pd.Series(myDict)

oneD = pd.Series([1,2,3],['a','b','c'])
oneD2 = pd.Series([1,2,3], index=['a','b','c'])

# Using .loc[] to locate value by index
locFilter = oneD.loc[['a','c']]

# Get data by index positions (row numbers)
posFilter = oneD[[0,2]]

# Get data at a specific index position
getData = oneD.iloc[1]

# Check for value in series
findAValue = 'a' in oneD

# Create a DataFrame (DF) using series (indices match) in a dictionary
myDict2 = {
    'A' : pd.Series([1,2,3,4,5], index=['a','b','c','d','e']),
    'B' : pd.Series([6,7,8,9,10], index=['a','b','c','d','e'])
}
myDF = pd.DataFrame(myDict2)

# Get row names (index) and column names (columns)
myDF.index
myDF.columns

# Subset a DF
newDF = pd.DataFrame(myDF, index=['a','c','e'], columns=['A'])

# Reading in CSV files
myFile = pd.read_csv('data/Resp2.csv')
myFile.head() # Show top 5 values

# You can specify the value to seperate columns on in the original file (\t is for tabs)
anotherFile = pd.read_csv('data/winequality-red.csv', sep=";")

"""
For EXCEL files: 
Use pd.ExcelFile(file) to load files into python
Use .sheet_names to get sheet names
Use .parse() to create a DF from a loaded excel file, specify the sheet as a parameter

For HTML files:
Use html5lib package
Use .read_html() to pull data into python (specify the web address in the parameters - gets ALL tables at location into a list)

"""

"""
Notes on data cleaning:
Use .isnull() to get a boolean repr. of the DF showing where NA values are
Use .dropna() to delete rows that contain NA values (NAN, NaN, etc...)
    > To select columns instead of rows, use param. 'axis'=1
    > To specify a 'threshold' or number of NAs to warrant removal, use the param 'thresh'
Use .fillna() to specify a value to replace NA values with
    > Use param 'method' to specify a filling method instead of a static value - ffill and bfill are forward/backward fills
    
"""

# Conditional Data Selection
DF = pd.read_csv('data/endangeredLang.csv')

# Subset by columns
DF_sub1 = DF[['Countries', 'Name in English']]

# Subset by rows
DF_sub2 = DF[3:10]

# Subset by condition
DF_sub3 = DF[DF['Number of speakers'] < 5000]

"""
Notes on data grouping:
Use .sort_values() to sort a DF - use param 'by' to select a column, use 'ascending' to state direction
Use .groupby() to group data by some qualitative variable
Use .count() to count data that has been grouped together (some similarities to SQL here...)
    > You can also use multiple values to group by and coupling that with count gives more information about groups of info
You can also use lambda function (example --> lambda x: x <= value) in a .apply() to filter data
    > DF[column].apply(lambda function)
    > You could also ADD a column by simply saying DF[new_col] = DF[column].apply(lambda function) 
Use .size() to get details of a .groupby() operations
use .value_counts() to get quant. counts of labels by their values from within a column

DATA SUBSETTING follows similar rules as normal indexing but used .loc and .iloc

DATA SORTING AND RANKING 

You can also CONCATENATE/MERGE/JOIN DFs using .concat() and .merge(). with join types matching SQL table merging operations

NOTE: You can use .apply() to apply a pre-defined function to all cells in a given DF
"""



