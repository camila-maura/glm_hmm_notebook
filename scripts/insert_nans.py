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
    """Insert a row of NaNs every n rows in a 2D array"""
    array = np.array(array)
    
    # Get the number of columns
    num_cols = array.shape[1]
    # Create a NaN row with the same number of columns
    nan_row = np.full((1, num_cols), np.nan)
    result = []
    for i in range(len(array)):
        result.append(array[i])
        # After every n items, add a NaN row (but not at the very end)
        if (i + 1) % n == 0 and (i + 1) < len(array):
            result.append(nan_row[0])
    return np.array(result)

# Example with your dimensions:
my_array = np.random.rand(5040, 3)  # Example: 5040 rows, 3 columns
result = insert_nan_rows_every_n(my_array, 90)

print(f"Original shape: {my_array.shape}")
print(f"New shape: {result.shape}")
print(f"Number of NaN rows inserted: {(result.shape[0] - my_array.shape[0])}")

result = insert_nan_every_n(choices_mouse, 90)
result_x = insert_nan_rows_every_n(X_mouse, 90)