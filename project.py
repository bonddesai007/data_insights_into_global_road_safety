import pandas as pd #Pandas is a python library used when working with dataset
import numpy as np #NumPy is a python library used to perform some operations with numbers.
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# Read the csv file
df = pd.read_csv("road_accident_dataset.csv")

#Task-2 (Data Collection & Preprocessing)
# Checking if there is any missing values in dataset
print("Before handling missing values")

# Counts missing values per column
print(df.isnull().sum())  

print(df.info())  # Provides an overview of non-null counts

# Applying imputation methods
df["Number of Vehicles Involved"].fillna(df["Number of Vehicles Involved"].mean(), inplace=True)  # Mean
df["Driver Alcohol Level"].fillna(df["Driver Alcohol Level"].median(), inplace=True)  # Median
df["Driver Fatigue"].fillna(df["Driver Fatigue"].mode()[0], inplace=True)  # Mode
df["Pedestrians Involved"].fillna(df["Pedestrians Involved"].median(), inplace=True)

# Rounding the values to the desired decimal places
df["Number of Vehicles Involved"] = df["Number of Vehicles Involved"].round(0)  # Rounding to 0 decimals
df["Driver Alcohol Level"] = df["Driver Alcohol Level"]  
df["Pedestrians Involved"] = df["Pedestrians Involved"].round(0)  # Rounding to 0 decimals

# Check again if there is any missing value left after imputation
print("After handling missing values")

# Ensure no missing values remain
print(df.isnull().sum())  

# Save cleaned dataset
# Saves without index
df.to_csv("cleaned_road_accident_dataset.csv", index=False)  
#df.head()

#Task-3 (Data Summarization & Descriptive Analysis)
# Compute central tendency (Mean, Median, Mode)

# Select numerical columns
num_columns = df.select_dtypes(include=[np.number])

# Compute Mean
mean = num_columns.mean()

# Compute Median
median = num_columns.median()

# Compute Mode (returns multiple values, so we take the first)
mode = num_columns.mode().iloc[0]

# Display results
print("Mean Values:\n", mean)
print("\nMedian Values:\n", median)
print("\nMode Values:\n", mode)

# Measures of Variation ( Range, Variance, Standard Deviation)

# Compute Range (Max - Min)
# Range is difference between maximum and minimum values in dataset
range = num_columns.max() - num_columns.min()

# Compute Variance
variance = num_columns.var()

# Compute Standard Deviation
standard_deviation = num_columns.std()

# Display results
print("\nRange:\n", range)
print("\nVariance:\n", variance)
print("\nStandard Deviation:\n", standard_deviation)

# Cross-tabulation
# Cross tabulation is a method used to analyze the relationship between two or more categorical variables

# Perform cross-tabulation: Example (Accident Severity vs. Weather Conditions)
cross_tab = pd.crosstab(df["Accident Severity"], df["Weather Conditions"])

# Display the cross-tabulation table
print("Cross-tabulation results")
print(cross_tab)

# Save the cross-tabulation to a CSV file
cross_tab.to_csv("road_accident_crosstab.csv")

# Frequency distribution is a representation that displays how often any specific values occur in dataset
# Select categorical columns ( Can also be calculated based on numerical values)
categorical_cols = df.select_dtypes(include=['object'])

# Compute and display frequency distribution for each categorical column
for col in categorical_cols.columns:
    print(f"\nFrequency Distribution for {col}:\n")
    print(df[col].value_counts())

#Data Smoothing 
#Data Smoothing is a technique used to reduce the noise in the data and make it more understandable
#Using technique - Moving Average Smoothing
#Moving Average Smoothing is a technique used to reduce the noise with the average of its nearby values



# List of columns to analyze for outliers
columns_to_analyze = [
    'Number of Vehicles Involved',
    'Driver Alcohol Level',
    'Driver Fatigue',
    'Pedestrians Involved',
    'Number of Injuries',
    'Number of Fatalities',
    'Emergency Response Time',
    'Traffic Volume',
    'Medical Cost',
    'Economic Loss'
]

# Function to detect outliers using IQR method
def detect_outliers(column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    return outliers

# Detect and print outliers for each column
for column in columns_to_analyze:
    outliers = detect_outliers(column)
    print(f"Outliers in column: {column}")
    print(outliers)
    print("\n")

# Detect and print outliers for each column
for column in columns_to_analyze:
    outliers = detect_outliers(column)
    print(f"Outliers in column: {column}")
    if outliers.empty:
        print("No outliers found.\n")
    else:
        print(outliers)
        print("\n")

# Remove unnecessary calculations for Q1, Q3, and IQR

# Filter outliers
outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]

print("Outliers in column:", column_name)
print(outliers)

# # Calculate Q1 and Q3
# Q1 = df[column_name].quantile(0.25)
# Q3 = df[column_name].quantile(0.75)
# IQR = Q3 - Q1

# # Calculate bounds
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# # Filter outliers
# outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]

# print("Outliers in column:", column_name)
# print(outliers)

# Task-6 (One-Hot Encoding)
# Identify categorical columns for one-hot encoding
categorical_columns = [
    'Country',
    'Month',
    'Day of Week',
    'Time of Day',
    'Urban/Rural',
    'Road Type',
    'Weather Conditions',
    'Driver Age Group',
    'Driver Gender',
    'Vehicle Condition',
    'Accident Severity',
    'Road Condition',
    'Accident Cause',
    'Region'
]

# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Save the encoded DataFrame to a new CSV file
df_encoded.to_csv("encoded_road_accident_dataset.csv", index=False)
print("One-hot encoding completed and saved to 'encoded_road_accident_dataset.csv'.")

# Select the first numeric column for smoothing
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

if numeric_columns:
    target_column = numeric_columns[0]
    print(f"Applying smoothing on column: {target_column}")

    # Moving Average Smoothing
    df['Moving_Avg_Smooth'] = df[target_column].rolling(window=3, min_periods=1).mean()

    # Gaussian Filtering
    df['Gaussian_Smooth'] = gaussian_filter1d(df[target_column].fillna(method='ffill'), sigma=1)

    # Plot original vs smoothed data
    plt.figure(figsize=(12, 6))
    plt.plot(df[target_column], label='Original', alpha=0.5)
    plt.plot(df['Moving_Avg_Smooth'], label='Moving Average (window=3)', linestyle='--')
    plt.plot(df['Gaussian_Smooth'], label='Gaussian Filter (sigma=1)', linestyle='-.')
    plt.title(f"Smoothing on '{target_column}'")
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel(target_column)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No numeric columns found in the dataset.")