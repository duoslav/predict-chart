import csv
import math

sin_filename = "sinusoidal_values.csv"
rsi_filename = "rsi_values.csv"

# Read the sinusoidal values from the file
sin_values = []
with open(sin_filename, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # skip the header row
    for row in reader:
        sin_values.append(float(row[0]))

# Calculate the RSI values
rsi_values = []
for i in range(14, len(sin_values)):
    up_sum = 0
    down_sum = 0
    for j in range(i-14, i):
        if sin_values[j] < sin_values[j+1]:
            up_sum += sin_values[j+1] - sin_values[j]
        else:
            down_sum += sin_values[j] - sin_values[j+1]
    rs = up_sum / 14 if up_sum > 0 else 0
    rd = down_sum / 14 if down_sum > 0 else 1  # changed this line
    rsi = 100 - 100 / (1 + rs/rd)
    rsi_values.append(rsi)

# Write the RSI values to a file
with open(rsi_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["rsi_value"])
    for rsi in rsi_values:
        writer.writerow([rsi])

print("RSI values saved in {}".format(rsi_filename))
