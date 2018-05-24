#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iomanip>
#include <unistd.h>
using namespace std;

// #include "SVD.hpp"

#define NUM_USERS 458293
#define NUM_MOVIES 17770
#define FILE_PATH_ALL     "../data/all.dta"
#define FILE_PATH_SMALLER "../data/smaller.dta"
#define FILE_PATH_SMALLE "../data/smalle.dta"
#define FILE_PATH_SMALL "../data/small.dta"

#define FILE_PATH_QUAL    "../data/qual.dta"
#define FILE_PATH_SQUAL    "../data/small_qual.dta"
#define OUTPUT_FILE_PATH_1 "../data/output_base.dta"
#define OUTPUT_FILE_PATH_2 "../data/output_valid.dta"
#define OUTPUT_FILE_PATH_3 "../data/output_hidden.dta"
#define OUTPUT_FILE_PATH_4 "../data/output_probe.dta"

#define INPUT_1
#define INPUT_2
#define INPUT_3
#define INPUT_4
#define INPUT_5

#define NUM_INPUTS 6

#define NUM_IN_QUAL 2749898

#define RESULTS_FILE_PATH_QUAL "../data/results_qual.dta"

#define GLOBAL_MEAN 3.5126


void LoadData(int data[2][16499388], string inputFiles[]){
  // reading file line by line

  for (int i = 0; i < 6; i++)
  {
    string train_file = inputFiles[i];
    ifstream infile(train_file);
    string line;
    int count = 0;
    // while not the end of the file
    while (getline(infile, line)){
      // read in current line, separate line into 4 data points
      // user number, movie number, date number, rating
      istringstream iss(line);
      // u : user number
      // i : movie number
      // d : date number
      // y : rating
      int user_id, rating;
      //cout << u << " " << i << " "  << d << " " << y << "\n";
      if (!(iss >> user_id >> rating)) { break; }
      // Change from 1-indexed to 0-indexed
      data[0][count] = user_id;
      data[1][count] = rating;
      count++;
    }
    infile.close();
  }
}

int main(int argc, char* argv[])
{
  int data[2][NUM_IN_QUAL*6];
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < NUM_IN_QUAL*6; j++)
    {
      data[i][j] = 0;
    }

  }

  ofstream qual_results;
  qual_results.open("../data/testblending.dta");
  string inputFiles[6] = {"data/results_qual_1.dta", "data/results_qual.dta"};
  LoadData(data, inputFiles);
  double b0 = 0;
  double b1 = 0;
  double alpha = 0.01;
  for (int i = 0; i < NUM_IN_QUAL*6; i++) {
    double p = b0 + b1 * data[0][i];
    double err = p - data[1][i];
    b0 = b0 - alpha * err;
    b1 = b1 - alpha * err * data[0][i];
  }

  for (int i = 0; i < NUM_IN_QUAL*6; i++){
    double rating = b0 + b1*data[0][i];
    qual_results << rating << "\n";

  }
}
