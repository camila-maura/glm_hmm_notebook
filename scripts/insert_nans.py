import numpy as np

def insert_nan_every_n(array, n=90):
    """Insert NaN every n items in the array"""
    array = np.array(array)  # Convert to numpy array if not already
    
    # Calculate how many NaNs we need to insert
    num_nans = len(array) // n
    
    # Create result array with space for NaNs
    result = []
    
    for i in range(len(array)):
        result.append(array[i])
        # After every n items, add a NaN (but not at the very end)
        if (i + 1) % n == 0 and (i + 1) < len(array):
            result.append(np.nan)
    
    return np.array(result)

# Example usage:
def insert_nan_rows_every_n(array, n=90):
    """Insert a slice of NaNs every n rows along axis 0 in an n-dimensional array"""
    array = np.array(array)
    
    # Create a NaN slice with the same shape as a single row (all dims except axis 0)
    nan_slice = np.full((1, *array.shape[1:]), np.nan)
    
    result = []
    for i in range(len(array)):
        result.append(array[i])
        # After every n items, add a NaN slice (but not at the very end)
        if (i + 1) % n == 0 and (i + 1) < len(array):
            result.append(nan_slice[0])
    
    return np.stack(result)

# Example with your dimensions:
my_array = np.random.rand(5040, 3)  # Example: 5040 rows, 3 columns
result = insert_nan_rows_every_n(my_array, 90)

print(f"Original shape: {my_array.shape}")
print(f"New shape: {result.shape}")
print(f"Number of NaN rows inserted: {(result.shape[0] - my_array.shape[0])}")

