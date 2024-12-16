"""
In this section, we will focus mainly on the
distribution of the datasets
"""

# Necessary packages
import numpy as np
import matplotlib.pyplot as plt
import os

# In this part, case refers to a particular dataset for which
# you want to evaluate the distribution. There are 3 different cases:
# 1. sim (simulated data) 2. sat (satellite data) 3. comb (combined data)

train = np.load('case_train_outputs.npy')
valid = np.load('case_valid_outputs.npy')
test = np.load('case_test_outputs.npy')

# Bins to distribute the data
bins = np.array([0, 3.5, 7, 17, np.inf])

# Function to plot the histogram
def plot_hist(data, filename, title, bins):
    plt.figure(figsize = (12, 6))
    plt.hist(data, bins = bins, edgecolor = 'black', alpha = 0.7)
    plt.title(title)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Replace case with whichever dataset you are working with
plot_hist(train.flatten(), 'case_train_dist.png', 'Histogram of Train Outputs', bins)
plot_hist(valid.flatten(), 'case_valid_dist.png', 'Histogram of Valid Outputs', bins)
plot_hist(test.flatten(), 'case_test_dist.png', 'Histogram of Test Outputs', bins)

# Function to calculate certain statistics about the emission data
def comp_stat(data):
    total_min = np.min(data)
    total_max = np.max(data) 
    total_range = total_max - total_min
    total_mean = np.mean(data)
    total_median = np.median(data)
    return {
        'total_min': total_min,
        'total_max': total_max,
        'total_range': total_range,
        'total_mean': total_mean,
        'total_median': total_median
    }

trainstat = comp_stat(train)
validstat = comp_stat(valid)
teststat = comp_stat(test)

print("Train Outputs Statistics:")
for key, value in trainstat.items():
    print(f"{key}: {value}")

print("\nValidation Outputs Statistics:")
for key, value in validstat.items():
    print(f"{key}: {value}")

print("\nTest Outputs Statistics:")
for key, value in teststat.items():
    print(f"{key}: {value}")

# Function to calculate the percentage of data in a certain bin
def calc_per(data, bins):
    counts, _ = np.histogram(data, bins = bins)
    total = np.sum(counts)
    percentages = (counts / total) * 100
    return counts, percentages

train_counts, train_per = calc_per(train.flatten(), bins)
valid_counts, valid_per = calc_per(valid.flatten(), bins)
test_counts, test_per = calc_per(test.flatten(), bins)

# Function to print the counts along with its respective percentage in a particular bin
def stats(name, counts, per, bins):
    print(f'\n{name} Data Histogram:')
    for i in range(len(counts)):
        print(f'Bin {bins[i]} to {bins[i+1]}:')
        print(f'Count: {counts[i]}')
        print(f'Percentage: {per[i]:.2f}%')

stats('Train', train_counts, train_per, bins)
stats('Valid', valid_counts, valid_per, bins)
stats('Test', test_counts, test_per, bins)
