## COMP1730/6730 Homework 2

# Your ANU ID: u7782042
# I declare that this submission is my own work
# [ref: https://www.anu.edu.au/students/academic-skills/academic-integrity ]
# Your NAME: Daoyan Zhu

import math

## You should implement the function `neural_network` below.
## You can use define new function(s) if it helps you decompose the problem
## into smaller problems.

# while 1:
#     x1=float(input ("tell me the value of x1 please"))
#     x2=float(input ("tell me the value of x2 please"))
#     x3=float(input ("tell me the value of x3 please"))
#     print ("you inputed",x1 , x2 , x3,"right?")
#     float (x1)
#     float (x2)
#     float (x3)
#     confirm1=input('press "y" for "yes" and press"n" for "no"')
#     if confirm1 == 'n':
#         continue
#     if confirm1 == 'y':
#         break

# while 1:
#     w1=float(input ("tell me the value of w1 please"))
#     w2=float(input ("tell me the value of w2 please"))
#     w3=float(input ("tell me the value of w3 please"))
#     w4=float(input ("tell me the value of w4 please"))
#     b1=float(input ("tell me the value of b1 please"))
#     b2=float(input ("tell me the value of b2 please"))
#     print ("you inputed",w1,w2,w3,w4,b1,b2,"right?")
#     confirm2=input('press "y" for "yes" and press"n" for "no"')
#     if confirm2 == 'n':
#         continue
#     if confirm2 == 'y':
#         break

# while 1:
#     w5=float(input ("tell me the value of w5 please"))
#     w6=float(input ("tell me the value of w6 please"))
#     b3=float(input ("tell me the value of b3 please"))
#     print ("you inputed",w5,w6,b3,"right?")
#     confirm3=input('press "y" for "yes" and press"n" for "no"')
#     if confirm3 == 'n':
#         continue
#     if confirm3 == 'y':
#         break
x1=float()
x2=float()
x3=float()
b1=float()
b2=float()
b3=float()
w1=float()
w2=float()
w3=float()
w4=float()
w5=float()
w6=float()
def ReLU(x):
    if x<=0:
        return 0
    else:
        return x


def sigmoid(x):
    return 1/(1+math.exp(-x))


def neural_network(x1, x2, x3, w1, w2, w3, w4, w5, w6, b1, b2, b3):
    a=x1*w1+x2*w2+b1
    b=x2*w3+x3*w4+b2
    c=ReLU(a)*w5+ReLU(b)*w6+b3
    d=sigmoid(c)
    return d
    # print ('your output is ',d)
# neural_network(x1, x2, x3, w1, w2, w3, w4, w5, w6, b1, b2, b3)

    
    

	

################################################################################
#               DO NOT MODIFY ANYTHING BELOW THIS POINT
################################################################################    

def test_neural_network():
    '''
    This function runs a number of tests of the neural_network function.
    If it works ok, you will just see the output ("all tests passed") at
    the end when you call this function; if some test fails, there will
    be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''

    assert math.fabs(neural_network(0, 0, 3.2, -0.1, 0.9, 0.7, -3.6, -1.0, -1.0, -1, -2.2, 0)  - 0.5) < 1e-6
    assert math.fabs(neural_network(0, 0, 3.2, -0.5, 1.9, 0.8, 1.0, 0.1, 1.3, -1, -2.2, 5.0)  - 0.9981670611) < 1e-6
    assert math.fabs(neural_network(0, 0, 3.2, -0.7, -0.4, -1.4, 4.1, -0.9, 2.6, 1, -2.2, 0)  - 1.0) < 1e-6
    assert math.fabs(neural_network(0, 0, 3.2, -0.9, 1.1, -0.7, -1.0, 0.7, -3.9, 1, -2.2, 5.0)  - 0.9966651927) < 1e-6
    assert math.fabs(neural_network(0, 2.5, 3.2, -0.9, 2.5, 0.0, -4.1, 1.0, 3.6, -1, -2.2, 0)  - 0.9947798743) < 1e-6
    assert math.fabs(neural_network(0, 2.5, 3.2, -0.8, -0.5, -1.5, -1.9, 0.2, -2.7, -1, -2.2, 5.0)  - 0.9933071491) < 1e-6
    assert math.fabs(neural_network(0, 2.5, 3.2, 0.4, -2.7, -1.3, 3.2, -0.2, -0.6, 1, -2.2, 0)  - 0.0534539038) < 1e-6
    assert math.fabs(neural_network(0, 2.5, 3.2, 0.2, -0.1, -0.5, -4.7, 0.5, 3.7, 1, -2.2, 5.0)  - 0.9953904278) < 1e-6
    assert math.fabs(neural_network(1, 0, 3.2, 1.0, 1.0, -0.6, -1.4, 0.4, 1.4, -1, -2.2, 0)  - 0.5) < 1e-6
    assert math.fabs(neural_network(1, 0, 3.2, -0.8, -1.6, -0.5, 0.1, 0.6, 0.0, -1, -2.2, 5.0)  - 0.9933071491) < 1e-6
    assert math.fabs(neural_network(1, 0, 3.2, -0.9, 1.9, -0.3, 0.2, -0.6, -0.6, 1, -2.2, 0)  - 0.4850044984) < 1e-6
    assert math.fabs(neural_network(1, 0, 3.2, -0.2, 1.6, -1.6, 0.5, -0.6, 1.6, 1, -2.2, 5.0)  - 0.98922827) < 1e-6
    assert math.fabs(neural_network(1, 2.5, 3.2, -0.9, -1.3, -0.6, 1.4, -0.9, -0.3, -1, -2.2, 0)  - 0.4417654819) < 1e-6
    assert math.fabs(neural_network(1, 2.5, 3.2, -0.6, -1.3, -0.1, 4.6, -0.0, 3.6, -1, -2.2, 5.0)  - 1.0) < 1e-6
    assert math.fabs(neural_network(1, 2.5, 3.2, -0.7, 2.5, -0.5, -3.1, 0.5, 3.3, 1, -2.2, 0)  - 0.9635611346) < 1e-6
    assert math.fabs(neural_network(1, 2.5, 3.2, -1.0, -1.4, 1.1, -1.6, -0.5, -3.9, 1, -2.2, 5.0)  - 0.9933071491) < 1e-6    

    print("all tests passed")
