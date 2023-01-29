import csv
import matplotlib.pyplot as plt

sin_filename = "sinusoidal_values.csv"
rsi_filename = "rsi_values.csv"

# Read the sinusoidal values from the file
sin_values = []
with open(sin_filename, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # skip the header row
    for row in reader:
        sin_values.append(float(row[0]))

# Read the RSI values from the file
rsi_values = []
with open(rsi_filename, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # skip the header row
    for row in reader:
        rsi_values.append(float(row[0]))

# Ensure that the arrays have the same length
if len(sin_values) != len(rsi_values) + 14:
    raise ValueError("The arrays have different lengths")

# Plot the sinusoidal and RSI values
x = range(0, len(sin_values))
plt.ylim(-1.1, 1.1)
plt.plot(x, sin_values, label='sinusoidal values')
plt.plot(x[14:], rsi_values, label='RSI values')
plt.legend()
plt.xlabel('Sample index')
plt.ylabel('Value')
plt.title('Sinusoidal and RSI values')
plt.show()
