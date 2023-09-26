# Data Preprocessing Workflow
# Step 1: Subtract Zero Load
# Step 2: Remove Outlier Samples
# Step 3: Remove Outliers
# Step 4: Data Aggregation

import numpy as np

def Mov_load(zerofile, datafile, outputfile):
    file = open(outputfile, 'w')
    y = np.loadtxt(zerofile, delimiter=',')  # Load data from zerofile
    y1 = np.loadtxt(datafile, delimiter=',')  # Load data from datafile
    len = y.shape[0]
    len1 = y1.shape[0]
    pin = np.zeros(96)
    
    # Calculate the mean for each column
    for i in range(96):
        for g in range(len):
            pin[i] = pin[i] + y[g, i]
        pin[i] = pin[i] / (len)
    print(pin)
    
    # Subtract the mean from data and write to the output file
    for i in range(len1):
        for g in range(96):
            a = y1[i, g] - pin[g]
            file.write(str(a))
            if (g < 95):
                file.write(',')
        file.write('\n')

def clean_data(cc_file, ss_file, cc_clean_file, ss_clean_file):
    """
    Remove data points with decimal values, cleaning cc.txt and ss.txt files.
    
    Args:
        cc_file (str): Path to cc.txt file.
        ss_file (str): Path to ss.txt file.
        cc_clean_file (str): Output file path for cleaned cc data.
        ss_clean_file (str): Output file path for cleaned ss data.
        
    Returns:
        None
    """
    # Read data from cc.txt and ss.txt files
    with open(cc_file, 'r') as f:
        cc_data = f.readlines()
    with open(ss_file, 'r') as f:
        ss_data = f.readlines()

    # Create new cc_clean.txt and ss_clean.txt files
    cc_clean = open(cc_clean_file, 'w')
    ss_clean = open(ss_clean_file, 'w')

    # Iterate over each line in cc.txt
    for i, line in enumerate(cc_data):
        # Check for decimal points, if present, skip the line
        if '.' in line:
            continue
        else:
            # Write the line to cc_clean.txt
            cc_clean.write(line)
            # Write the corresponding ss.txt data to ss_clean.txt
            ss_clean.write(ss_data[i])

    # Close all open files
    cc_clean.close()
    ss_clean.close()

def remove_outliers_mad(data, threshold=50):
    """
    Apply the Median Absolute Deviation (MAD) method for outlier removal.
    
    Args:
        data (np.ndarray): NumPy array containing the data.
        threshold (float): Threshold value for identifying outliers (default is 50).
        
    Returns:
        np.ndarray: Data with outliers replaced by the median of nearby values.
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad  # 0.6745 represents the difference between the 75th percentile and the median in a standard normal distribution
    mask = np.abs(modified_z_scores) > threshold
    # Replace outliers with the median of nearby data points
    for i in range(len(data)):
        if mask[i]:
            data[i] = np.median(data[max(0, i - 5):min(len(data), i + 5)])
    return data

def filter_outliers(infile, outfile):
    """
    Read the specified file, remove outliers for each column, and save the processed data to a new file.
    
    Args:
        infile (str): Path to the input file that requires MAD-based outlier removal.
        outfile (str): Path to the output file for the processed data.
        
    Returns:
        None
    """
    data = np.loadtxt(infile, delimiter=',')
    data_no_outliers = np.zeros_like(data)
    for i in range(data.shape[1]):
        col_data = data[:, i]
        col_data_no_outliers = remove_outliers_mad(col_data)
        data_no_outliers[:, i] = col_data_no_outliers
    np.savetxt(outfile, data_no_outliers, delimiter=',')