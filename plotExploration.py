import matplotlib as mt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
# import sklearn as skl

# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# from scipy import interpolate

########################################
### Test plots from the ASSASSN Pandas 
########################################

# set directory
MAIN_DIR = '/Users/isaacperez/Downloads/CS70 Homework/P-Stars_Project'
DATA_DIR = os.path.join(MAIN_DIR, 'team_Sparse')
DATA_DIR2 = os.path.join(MAIN_DIR, 'var_output/v0.1.0')
LC_DIR = os.path.join(MAIN_DIR, 'g_band_lcs') # for folded .dats
LC_OUT = os.path.join(MAIN_DIR, 'sample_lcs')

# While I figure out how to properly work a directory :(
DATA_DIR3 = '/Users/isaacperez/Downloads/CS70 Homework/P-Stars_Project/team_Sparse/HCV.csv'

X_AXIS = 'chi2'
Y_AXIS = 'mad'
ERROR_BARS = 'Mag_err'
RANGE = 37

def plotCurves():
    """
    Plots every lightcurve in the data set AllVar.phot.csv
    """

    # Load the CSV file into a DataFrame
    file_path = DATA_DIR3 
    data = pd.read_csv(file_path)

    # Get unique values in the "ID" column
    unique_ids = data['ID'].unique()

    # Create a scatter plot for each unique ID
    for unique_id in unique_ids:
        # Extract data for the current ID
        subset = data[data['ID'] == unique_id]
        
        # Extract the "Dec" and "Mag" columns for the current ID
        dec_values = subset[X_AXIS]
        mag_values = subset[Y_AXIS]
        frequency_values = data[ERROR_BARS]

        # Create a scatter plot for the current ID
        plt.figure(figsize=(10, 6)) 
        plt.scatter(dec_values, mag_values, c='blue', marker='o', s=10)  # Customize the scatter plot appearance

        # Create data bars using plt.bar
        # plt.bar(dec_values, frequency_values, width=0.1, align='center', alpha=0.5, color='red', label='Frequency')

        # Set plot labels and title
        plt.xlabel(X_AXIS)
        plt.ylabel(Y_AXIS)
        plt.title(f'Scatter Plot for ID {unique_id}')

        # Show the plot
        plt.grid(True)
        plt.show()



def example_Specific_Curve(ID, X_Axis, Y_Axis):
    """
    Outputs an curve with a given ID, X-Axis, and Y-Axis
    """

    file_path = DATA_DIR3
    data = pd.read_csv(file_path)

    subset  = data[data['groupid'] == ID]

    if subset.empty:
        print(f"No data found for ID {ID}")
        return 

    dec_values = subset[X_Axis]
    mag_values = subset[Y_Axis]

    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.scatter(dec_values, mag_values, c='blue', marker='o', s=10)  # Customize the scatter plot appearance

    # Set plot labels and title
    plt.xlabel(X_Axis)
    plt.ylabel(Y_Axis)
    plt.title(f'Scatter Plot for ID {ID}')

    # Show the plot
    plt.grid(True)
    plt.show()




def example_Problem_Curve():
    """
    Outputs an example of data with very few data points that we will be working with in the project. 
    """

    min_data_points = 10

    # Load the CSV file into a DataFrame
    file_path = DATA_DIR3  # Replace with the path to your CSV file
    data = pd.read_csv(file_path)
    ID = 'groupid'

    # Group data by the "ID" column and count the number of data points for each ID
    id_counts = data[ID].value_counts()

    # Filter IDs with less than the specified minimum data points
    ids_to_plot = id_counts[id_counts < min_data_points].index

    # Iterate through the filtered IDs and create scatter plots
    i = 0
    for unique_id in ids_to_plot:
        if i == 5:
            break
    # Extract data for the current ID
        subset = data[data[ID] == unique_id]

        dec_values = subset[X_AXIS]
        mag_values = subset[Y_AXIS]

        # Create a scatter plot for the current ID
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        plt.scatter(dec_values, mag_values, c='blue', marker='o', s=10)  # Customize the scatter plot appearance

        # Set plot labels and title
        plt.xlabel(X_AXIS)
        plt.ylabel(Y_AXIS)
        plt.title(f'Scatter Plot for ID {unique_id}')

        # Show the plot
        plt.grid(True)
        plt.show()
        i = i + 1

### Main: 

# example_Problem_Curve()
example_Specific_Curve(ID=423081, X_Axis="lightcurve_d", Y_Axis="lightcurve_m")
# example_Specific_Curve(ID=82373, X_Axis="lightcurve_d", Y_Axis="lightcurve_m")
# example_Specific_Curve(ID=81871, X_Axis="lightcurve_d", Y_Axis="lightcurve_m")
# example_Specific_Curve(ID=82136, X_Axis="lightcurve_d", Y_Axis="lightcurve_m")
# example_Specific_Curve(ID=68312, X_Axis="lightcurve_d", Y_Axis="lightcurve_m")
# example_Specific_Curve(ID=64702, X_Axis="lightcurve_d", Y_Axis="lightcurve_m")



#######################
# Methods of Approach #
#######################

# 1.) Feature Extraction: Extract relevant features from the limited data points to represent the light curve. 
# Features can include basic statistics like mean, standard deviation, skewness, and kurtosis of the magnitude values.

# Kurtosis: Kurtosis is a statistical measure used to describe the tailedness of a probability distribution of a real-valued random variable. 
# It describes how often outliers occur, and excess kurtosis is the tailedness of a distribution relative to a normal distribution.
# When normally distributed data is plotted on a graph, it generally takes the form of a bell, and the plotted data that are furthest from the mean 
# usually form the tails on each side of the curve.

# 2.) Use Simple Models: With limited data, it's often best to use simpler classification models that don't require large amounts of data. 
# Decision trees or random forests can work well in this situation

# Dropout will be used to prevent overfitting. We can recycle Astro101-Final-Project/ nn_v0.0.1/nn.py
