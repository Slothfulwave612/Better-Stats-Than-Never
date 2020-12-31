import numpy as np
import pandas as pd

def make_frequency_distribution(data, user_input=None, extra=True):
    """
    Function to make frequency distribution.
    
    Args:
        data (numpy.array): data containing records.
        user_input (tuple, optional): 
                    user_input for start_value, end_value, total_classes.
                    Defaults to None.
        extra (bool, optional): to make extra columns like cumulative, relative frequency.
    
    Returns:
        pandas.DataFrame: required frequency distribution.
    """
    ## total number of observations
    length = len(data)

    ## lowest and highest number in the data
    lowest = min(data)
    highest = max(data)
    
    ## total number of class 
    if user_input == None:    
        total_classes = int(np.sqrt(length))
    else:
        lowest, highest, total_classes = user_input

    ## range of the data
    range_ = highest - lowest
    
    print(f"Start value: {lowest}")
    print(f"End value: {highest}")
    print(f"Range: {range_}")
    print(f"Total Number of Classes: {total_classes}")
    
    ## calculate width
    width = range_ / total_classes
    
    ## list of all class intervals
    class_intervals = [
        np.round(start,3) for start in np.linspace(lowest, highest, total_classes+1)
    ]
    
    print(f"Class Width = {np.round(width, 3)}", end="\n\n")
    
    ## calculate frequency for each class
    hist, _ = np.histogram(data, bins=class_intervals)
    
    ## frequency table
    df = pd.DataFrame(
        {
            "Class Intervals": [
                f"{first} - under {second}" if second != highest \
                else f"{first} -  {second}" \
                for first, second in zip(class_intervals, class_intervals[1:])
            ],
            "Frequency": hist
        }
    )
    
    if extra:
        ## class midpoint
        df["Class Midpoint"] = df["Class Intervals"].apply(
            lambda x: (
                ( float(x.split(' ')[0]) + float(x.split(' ')[-1]) ) / 2
            )
        )

        ## relative frequency
        df["Relative Frequency"] = df["Frequency"] / df["Frequency"].sum()

        ## cumulative frequency
        df["Cumulative Frequency"] = df["Frequency"].cumsum()

    return df