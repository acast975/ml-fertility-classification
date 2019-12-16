import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt


def create_model(first_param_name: str, second_param_name: str, first_data: list, second_data: list):
    N = 10
    ind = np.arange(N)
    width = 0.35

    p1 = plt.bar(ind, first_data, width)
    p2 = plt.bar(ind, second_data, width,
                 bottom=first_data)

    minimum = 0
    maximum = 100

    step = 10

    plt.title(f'{first_param_name} and {second_param_name}')
    plt.xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    plt.legend((p1[0], p2[0]), (first_param_name, second_param_name))

    plt.show()

#
# data_file_name = "./fertilityDiagnosis.txt"
# with open(data_file_name, 'r') as f:
#     dataset = [[float(num) for num in line.split(',')] for line in f]
#
# matrix = np.array(dataset)
#
# first_data = matrix[0]
#
# for row in matrix:
#     create_model(first_param_name='first', second_param_name='second', first_data=first_data, second_data=row)


testdata1 = (10,20,30,40,50,60,70,80,1,5)
testdata2 = (15, 25, 35, 45, 55, 65, 75,10,12,17)

create_model(first_param_name='first', second_param_name='second', first_data=testdata1, second_data=testdata2)
