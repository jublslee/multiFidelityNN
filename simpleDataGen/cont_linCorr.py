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

#---------specify input parameters--------#
x_L = [i / 10 for i in range(11)]
x_H = [0,0.4,0.6,1]
A   = 0.5
B   = 10
C   = -5
#--------------------------------------------------------#

#---------generate data and store in txt file--------#
# Output content
print(cont_linCorr_L(x_L,A,B,C))
# print(cont_linCorr_L(0,A,B,C))
# print(cont_linCorr_L(0.1,A,B,C))

# Create .txt file for low fidelity data
with open("lowData.txt", "w") as file:
    # print(cont_linCorr_L(x_L,A,B,C), file=file)
    file.write(','.join(map(str, cont_linCorr_L(x_L,A,B,C))))

# Create .txt file for high fidelity data
with open("highData.txt", "w") as file:
    # print(cont_linCorr_H(x_H), file=file)
    file.write(','.join(map(str, cont_linCorr_H(x_H))))

print("Output has been written to separate .txt files.")

#--------------------------------------------------------#


