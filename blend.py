import itertools
import numpy as np
import random

# the probe output of a system
PROBE_RESULTS_1 = "../data/probe_results_SVD_0.dta"
PROBE_RESULTS_2 = "../data/probe_results_SVD_1.dta"
PROBE_ACTUAL_RESULTS = "../data/probe_actual.dta"

#
QUAL_RESULTS_1 = "../data/results_qual_0.dta"
QUAL_RESULTS_2 = "../data/results_qual_1.dta"
BLENDED_QUAL = "../data/results_blended.dta"

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

def applyWeights(qual_results, weights, write_file):
    qual_Matrix = []
    for file in qual_results:
        data1 = np.loadtxt(file, delimiter='\n')
        qual_Matrix.append(data1)
    qual_Matrix = np.array(qual_Matrix)
    qual_Matrix = qual_Matrix.T

    w_file = open(write_file, 'w')
    for row in range(len(qual_Matrix)):
        blended_rating = 0.0
        for i in range(NUM_MODELS):
            blended_rating += weights[i] * qual_Matrix[row][i]

        if blended_rating > 5 :
            blended_rating = 5
        if blended_rating < 1 :
            blended_rating = 1
        
        w_file.write("%s\n" % blended_rating)
    return


if __name__ == '__main__':
    probe_result_files = [PROBE_RESULTS_1, PROBE_RESULTS_2]
    qual_result_files = [QUAL_RESULTS_1, QUAL_RESULTS_2]
    # probe_result_files = ['data/test.dta', 'data/test2.dta']
    # qual_result_files = ['data/test.dta', 'data/test2.dta']
    # probe_file = 'data/test.dta'
    A,s = getData(probe_result_files, PROBE_ACTUAL_RESULTS)
    weights = blend(A, s)
    print(weights)
    blended_output = applyWeights(qual_result_files, weights, BLENDED_QUAL)
    #print(len(blended_output))