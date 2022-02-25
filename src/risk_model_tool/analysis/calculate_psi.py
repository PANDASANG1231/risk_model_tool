import sys
import numpy as np

def scale_range (input_, min_, max_):
    input_ += -(np.min(input_))
    scaler = np.max(input_) / np.array(max_ - min_)
    input_ = input_ / scaler
    input_ += min_
    return input_

def sub_psi(e_perc, a_perc):
    """Calculate the actual PSI value from comparing the values.
       Update the actual value to a very small number if equal to zero
    """
    if a_perc == 0:
        a_perc = 0.0001
    if e_perc == 0:
        e_perc = 0.0001

    value = (e_perc - a_perc) * np.log(e_perc / a_perc)
    return value
    
def calculate_psi(expected, actual, buckettype="bins", breakpoints=None, buckets=10, axis=0):
    """Calculate the PSI (population stability index) across all variables
    
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values
       buckettype: type of strategy for creating buckets, bins splits into even splits, 
               quantiles splits into quantile buckets, customize split into customized buckets 
       breakpoints: if buckettype is customizer, pass a numpy array as breakpoints 
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    
    Returns:
       psi_values: ndarray of psi values for each variable
    
    """
    def psi(expected_array, actual_array, buckets, breaks=None):
        """Calculate the PSI for a single variable

        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values
           buckets: number of percentile ranges to bucket the values into
           breaks: default None, customize breakpoints

        Returns:
           psi_value: calculated PSI value

        """

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])
        elif buckettype == 'customize':
            assert breaks is not None, "buckettype is customize, breakpoints should not be None"
            breakpoints = breaks


        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)


        psi_value = sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return psi_value
    
    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])  

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets, breakpoints)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets, breakpoints)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets, breakpoints)

    return psi_values
    

