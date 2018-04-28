import os
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise import SVD, SVDpp, Dataset, Reader, accuracy, dump
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate, PredefinedKFold


# As we're loading a custom dataset, we need to define a reader. 
reader = Reader(line_format='user item timestamp rating', sep=' ')

# load in the training data
data_train = Dataset.load_from_file("../data/output_base.dta", reader=reader)
# split the data in train and test sets
train_set, test_set = train_test_split(data_train, test_size=.02)

#data = Dataset.load_from_folds([("../data/small.dta", "../data/output_valid.dta")], reader=reader)
# test_data = Dataset.load_from_file("../data/output_valid.dta", reader=reader)
# test = Dataset.load_from_file("../data/output_valid.dta", reader=reader)
#train, test = train_test_split(train_data, test_size=0)
#test,train2 = train_test_split(test_data, test_size=0)
#pfk = PredefinedKFold()

# choose SVD
algo = SVD(verbose = True)
algo.fit(train_set)
predictions = algo.test(test_set)

accuracy.rmse(predictions, verbose = True)

# write results to file
input_file = open('../data/qual.dta', 'r') 
output_file = open('../data/surprise_qual.dta', 'w') 

for line in input_file:
	line_split = line.split(" ")
	pred = algo.predict(line[0], line[1])
	output_file.write('%s\n' %(pred.est))
