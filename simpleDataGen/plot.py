import numpy as np
import matplotlib.pyplot as plt
from cont_linCorr import *

# Define the range of x values
x_values = np.linspace(0, 1, 100)  # Generates 100 x values evenly spaced between -5 and 5
A   = 0.5
B   = 10
C   = -5
y_values = cont_linCorr_L(x_values,A,B,C)

with open('lowData.txt', 'r') as file:
    low = file.read()
# Split the content into a list of elements separated by spaces
lowFid = low.split(',')

# Convert elements to integers (or floats if needed)
lowFid = [float(lowFid_i) for lowFid_i in lowFid]

with open('highData.txt', 'r') as file:
    high = file.read()
# Split the content into a list of elements separated by comma
highFid = high.split(',')

# Convert elements to integers (or floats if needed)
highFid = [float(highFid_i) for highFid_i in highFid]

# Plot the equation
plt.plot(x_values, y_values, label='y = x^2 - 2x + 1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of the Equation y = x^2 - 2x + 1')
plt.grid(True)
plt.legend()
plt.show()

y_values = cont_linCorr_L(x_values,A,B,C)

with open('lowData.txt', 'r') as file:
    low = file.read()
# Split the content into a list of elements separated by spaces
lowFid = low.split(',')

# Convert elements to integers (or floats if needed)
lowFid = [float(lowFid_i) for lowFid_i in lowFid]

with open('highData.txt', 'r') as file:
    high = file.read()
# Split the content into a list of elements separated by spaces
highFid = high.split(',')

# Convert elements to integers (or floats if needed)
highFid = [float(highFid_i) for highFid_i in highFid]

# Plot the equation
plt.plot(x_values, y_values, label='y = x^2 - 2x + 1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of the Equation y = x^2 - 2x + 1')
plt.grid(True)
plt.legend()
plt.show()
