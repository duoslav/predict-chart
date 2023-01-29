import csv

# Input file name
input_file = 'sinusoidal_values_3600_1.csv'
# Output file name
output_file = 'windows_3600_1.csv'

# Read data from input file
with open(input_file, 'r') as f_in:
    reader = csv.reader(f_in)
    header = next(reader)
    input_data = [list(map(float, row)) for row in reader]

# Get number of rows and columns from input data
num_rows = len(input_data)
num_cols = len(input_data[0])

# Create a list to store output data
output_data = []

window = 16

# Iterate through each row
for i in range(num_rows + 1 - window):
    new_row = []

    for col in range(num_cols):
        for j in range(window):
            new_row += input_data[i + j]

    output_data.append(new_row)

# Write data to output file
with open(output_file, 'w') as f_out:
    writer = csv.writer(f_out)
    writer.writerows(output_data)
