import itertools
import numpy as np
import random

PROBE_RESULTS_1 = "../data/..."
PROBE_RESULTS_2 = "../data/..."

QUAL_RESULTS_2 = "../data/..."
QUAL_RESULTS_2 = "../data/..."

NUM_MODELS = 2

def getData(file_list, probe_file):
    matrix_A = []
    for file in file_list:
        data1 = np.loadtxt(file, delimiter='\n')
        matrix_A.append(data1)
    matrix_A = np.array(matrix_A)
    matrix_A = matrix_A.T

    s = np.loadtxt(probe_file, delimiter = '\n')
    return (matrix_A, s)

def blend(A, s):
    A_T = A.T
    weights_1 = np.linalg.inv(np.matmul(A_T, A))
    weights_2 = np.matmul(A_T, s)
    return np.matmul(weights_1, weights_2)

def applyWeights(qual_results, weights):
    qual_Matrix = []
    for file in qual_results:
        data1 = np.loadtxt(file, delimiter='\n')
        qual_Matrix.append(data1)
    qual_Matrix = np.array(qual_Matrix)
    qual_Matrix = qual_Matrix.T

    blended_results = []
    for row in range(len(qual_Matrix)):
        blended_rating = 0.0
        for i in range(NUM_MODELS):
            blended_rating += weights[i] * qual_Matrix[row][i]
        blended_results.append(blended_rating)

    return blended_results


if __name__ == '__main__':
    probe_result_files = [file1, file2, ...]
    qual_result_files = [file1, file2, ...]
    # probe_result_files = ['data/test.dta', 'data/test2.dta']
    # qual_result_files = ['data/test.dta', 'data/test2.dta']
    # probe_file = 'data/test.dta'
    A,s = getData(probe_result_files, probe_file)
    weights = blend(A, s)
    blended_output = applyWeights(qual_result_files, weights)
    print(len(blended_output))
