"""
After the model training and evaluation, we need to check the performance in terms
of errors. We considered the Absolute Error and Relative Error as our metrics. 
Further, we separated the erros into bins for analysis. Also, we checked the
difference between predicted and true emissions in terms of min, max, mean, median and 
range.
"""

# import necessary packages
import pandas as pd

df = pd.read_csv('filepath.csv') # the filepath will be your own name that you decided while storing the results
df['error'] = (df['True Emissions'] - df['Average Emissions']).round(3) # Error between true and predicted emissions rounded off to 3 decimal places
df['absolute_error'] = (abs(df['error'])).round(3) # absolute error 
df['relative_error'] = ((df['error'] / (df['True Emissions'] + 1e-15)) * 100).round(3) # relative error

"""
absolute error
"""
abs = df['absolute_error']

# creating different bins : (0-2), (2-5), (5-10) and (10 above) [all in Mt/yr]
in0_2 = (abs >= 0) & (abs <= 2)
in2_5 = (abs > 2) & (abs <= 5)
in5_10 = (abs > 5) & (abs <= 10)
ab10 = abs > 10
mean = round(abs.mean(), 3) # mean
median = round(abs.median(), 3) # median
std = round(abs.std(), 3) # standard deviation

print("in between 0 and 2 : ", in0_2.sum())
print("in between 2 and 5 : ", in2_5.sum())
print("in between 5 and 10 : ", in5_10.sum())
print("above 10 : ", ab10.sum())
print("mean : ", mean) 
print("median : ", median)
print("std : ", std)

"""
absolute relative error
"""
df['abs_rel_error'] = abs(df['relative_error'])
rel = df['abs_rel_error']

# creating different bins : (0-20), (20-50), (50-100), (100-150) and (150 above)
above150 = (rel > 150)
in150_100 = (rel > 100) & (rel <= 150)
in100_50 = (rel > 50) & (rel <= 100)
in50_20 = (rel > 20) & (rel <= 50)
in20_0 = (rel > 0) & (rel <= 20)

mean = round(rel.mean(), 3) # mean
median = round(rel.median(), 3) # median
std = round(rel.std(), 3) # standard deviation

print("above 150% : ", above150.sum()) 
print("in between 100% and 150% : ", in150_100.sum()) 
print("in between 50% and 100% : ", in100_50.sum())
print("in between 20% and 50% : ", in50_20.sum())
print("in between 0% and 20% : ", in20_0.sum())
print("mean : ", mean) 
print("median : ", median)
print("std : ", std) 

"""
statistics between true and predicted emissions
"""
true = df['True Emissions']
min_true = round(true.min(), 3)
max_true = round(true.max(), 3)
mean_true = round(true.mean(), 3)
std_true = round(true.std(), 3)
median_true = round(true.median(), 3)
range_true = round(max_true-min_true, 3)

pred = df['Average_Emissions']
min_pred = round(pred.min(), 3)
max_pred = round(pred.max(), 3)
mean_pred = round(pred.mean(), 3)
std_pred = round(pred.std(), 3)
median_pred = round(pred.median(), 3)
range_pred = round(max_pred-min_pred, 3)

print("True statistics : ")
print()
print("min : ", min_true)
print("max : ", max_true)
print("mean : ", mean_true)
print("std : ", std_true)
print("median : ", median_true)
print("range : ", range_true) 
print()
print("Predicted statistics : ")
print()
print("min : ", min_pred)
print("max : ", max_pred) 
print("mean : ", mean_pred)
print("std : ", std_pred)
print("median : ", median_pred)
print("range : ", range_pred)

df.to_csv('filepath.csv', index = False) # save



