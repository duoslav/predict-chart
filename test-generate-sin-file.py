import csv
import math

filename = "sinusoidal_values.csv"

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sin_value"])
    
    for i in range(0, 3600, 1):
        sin_value = math.sin(math.radians(i/10))
        writer.writerow([sin_value])

print("Sinusoidal values saved in {}".format(filename))
