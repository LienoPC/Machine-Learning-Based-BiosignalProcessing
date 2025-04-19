from scipy.io import loadmat

# Replace 'filename.mat' with the path to your .mat file.
data = loadmat('subject4.mat')

# Examine the keys in the resulting dictionary.
print(data.keys())

# Assuming your file contains a variable named 'myVar'
my_var = data.get('start_sample')

print(my_var)