import math

#----------------------------------------------------------------------------------------#
# Generates simple low fidelity data using a continuous function with linear correlation
# Inputs:
#       x: input vectore where values are between [0,1]
#       A: equation parameter
#       B: equation parameter
#       C: equation parameter
# Outputs:
#       Low fidelity data
def cont_linCorr_L(x, A, B, C):
    y_L = [A * (6*x_i - 2)**2 * math.sin(12*x_i - 4) + B*(x_i - 0.5) + C for x_i in x]
    # y_L = A * (6*x - 2)**2 * math.sin(12*x - 4) + B*(x - 0.5) + C ## for individual checking purpose
    return y_L
    # with open("lowData.txt", "w") as file:
    #     for x_i in x:
    #         y_i = A * (6*x_i - 2)**2 * math.sin(12*x_i - 4) + B*(x_i - 0.5) + C 
    #         print(y_i, file=file)
    #         print('\n', file=file)

#----------------------------------------------------------------------------------------#
# Generates simple high fidelity data using a continuous function with linear correlation
# Inputs:
#       x: input vectore where values are between [0,1]
# Outputs:
#       High fidelity data
def cont_linCorr_H(x):
    y_H = [(6*x_i - 2)**2 * math.sin(12*x_i - 4) for x_i in x]
    return y_H


