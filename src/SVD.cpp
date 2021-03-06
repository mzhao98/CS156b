#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
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
#define OUTPUT_FILE_PATH_1 "../data/output_base.dta"
#define OUTPUT_FILE_PATH_2 "../data/output_valid.dta"
#define OUTPUT_FILE_PATH_3 "../data/output_hidden.dta"
#define OUTPUT_FILE_PATH_4 "../data/output_probe.dta"

#define RESULTS_FILE_PATH_QUAL "../data/results_qual.dta"
#define SHUFFLED_DATA "../data/shuf.dta"

#define GLOBAL_MEAN 3.5126

/*
 * A class to perform SVD on the Netflix data to get user prediction ratings
 */
class SVD{
  public:
      /*
       * Public variables
       */
      double *bu;  // bias for users
      double *bi;  // bias for movies
      double **pu; // matrix of users
      double **qi; // matrix of movies
      int k;       // number of latent factors
      double reg1; // first regularization term
      double reg2; // second regularization term
      double eta;  // the learning

      //double **y;  // matrix of factor vectors for y_i
      // int *Ru;

      // double **data; /* input data matrix */
      // std::string train_filename;

      SVD(int k_factors, double regularize1, double regularize2, double learning_rate); // Constructor
      ~SVD(); // destructor
      double predict(double *curr_pu, double *curr_qi, double curr_bu, double curr_bi);
      double train(string train_file, int iters);
      double get_error();
      void write_results(string write_file, string in_file);
};


/*
 * Constructor Method
 *
 * @param k_factors : the number of latent factors to use
 * @param regularize1 : the first regularization factor
 * @param regularize 2 : the second regularization factor
 * @param learning_rate : the learning rate for every iteration
 */

SVD::SVD(int k_factors, double regularize1, double regularize2, double learning_rate)
{
    reg1 = regularize1;
    reg2 = regularize2;
    eta = learning_rate;

    k = k_factors;
    bu = new double[NUM_USERS];   // user biases
    bi = new double[NUM_MOVIES];  // movie biases
    // Ru = new int[NUM_USERS];   // number of movies a user has rated
    pu = new double*[NUM_USERS];  // user matrix
    qi = new double*[NUM_MOVIES]; // movie matrix


    // init the bu and bi bias arrays
    for (int i = 0; i < NUM_USERS; i++){
       bu[i] = 0.0;
    }

    for (int i = 0; i < NUM_MOVIES; i++){
      bi[i] = 0.0;
    }
    // create 2D array pu
    for(int i = 0; i < NUM_USERS; i++){
      pu[i] = new double[k];
    }

    // Create random number generator for generating from -0.5 to 0.5
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> dis(-0.5, 0.5); // Uniform distribution

    // initialize pu to small random variables
    int n, m;
    for (n = 0; n < NUM_USERS; n++){
	    for (m = 0; m < k; m++){
	      pu[n][m] = dis(gen);
	    }
	   }

    // create 2D array qi
    for(int i = 0; i < NUM_MOVIES; i++){
      qi[i] = new double[k];
    }

    // init qi to all 0.1
    for (n = 0; n < NUM_MOVIES; n++){
	    for (m = 0; m < k; m++){
	      qi[n][m] = dis(gen);
	    }
	   }
}

// Destructor
SVD::~SVD() {
  delete[] bu;
  delete[] bi;
  for (int i = 0; i < NUM_USERS; i++){
    delete[] pu[i];
  }
  for(int i = 0; i < NUM_MOVIES; i++){
    delete[] qi[i];
  }
  delete[] pu;
  delete[] qi;
}

/*
 * This function trains on the input file
 *
 *
 * @param train_file : the name of the file containing the training data
 */
double SVD::train(string train_file, int iters){
    double * grad_pu = new double[k];
    double * grad_qi = new double[k];
    // Shuffle data to file SHUFFLED_DATA
    for (int m = 0; m < iters; m++)
    {
      cout << "Iter: " << m << endl;

      string command = "gshuf " + train_file + " > " + SHUFFLED_DATA;
      system(command.c_str());
      cout << "Shuffled data" << endl;

      // reading file line by line
      ifstream infile(SHUFFLED_DATA);
      string line;
      // while not the end of the file
      int counter = 0;
      while (getline(infile, line)){
        // read in current line, separate line into 4 data points
        // user number, movie number, date number, rating
        istringstream iss(line);
        // u : user number
        // i : movie number
        // d : date number
        // y : rating
        int u, i, d, y;
        //cout << u << " " << i << " "  << d << " " << y << "\n";
        if (!(iss >> u >> i >> d >> y)) { break; }
        // Change from 1-indexed to 0-indexed
        u = u - 1;
        i = i - 1;

        // START OF TRAINING
        double error = y - predict(pu[u], qi[i], bu[u], bi[i]);
        //printf("old error = %f \n", error);
        // update the biases
        bu[u] += eta * (error - reg1 * bu[u]);
        bi[i] += eta * (error - reg1 * bi[i]);

        // use movieNumber, userNumber to index matrices for SVD
        // compute gradients of current user/ movie matrix
        for (int j = 0; j < k; j++){
          grad_pu[j] = reg1 * pu[u][j] - error * qi[i][j];
        }
        for (int j = 0; j < k; j++){
          grad_qi[j] = reg1 * qi[i][j] - error * pu[u][j];
        }
        for (int j = 0; j < k; j++){
          pu[u][j] = pu[u][j] - eta*grad_pu[j];
        }
        for (int j = 0; j < k; j++){
          qi[i][j] = qi[i][j] - eta*grad_qi[j];
        }

        double new_error = y - predict(pu[u], qi[i], bu[u], bi[i]);
        //printf("new error = %f \n", new_error);
        counter += 1;
        // if (counter % 1000000 == 0){
        //   cout << "in train " << counter << "\n";
        // }

        }
      }
      delete[] grad_pu;
      delete[] grad_qi;
  return 0;

}

/*
 * This function creates a prediction rating for a given user/movie pair using
 * the pu and qi matrices and the bias bu and bi matrices.
 *
 * @param curr_pu : pointer to the corresponding user matrix
 * @param curr_qi : pointer to the corresponding movie matrix
 * @param curr_bu : corresponding user bias term
 * @param curr_bi : corresponding movie bias term
 */
double SVD::predict(double *curr_pu, double *curr_qi, double curr_bu, double curr_bi){
    double product = 0;
    for (int i = 0; i < k; i++){
        product += (curr_pu[i] * curr_qi[i]);
    }
    product += (curr_bu + curr_bi + GLOBAL_MEAN);
    //cout << "predict_result" << product << endl;
    return product;
}


// void dotMatrices(int i, int j, double mat1[10][10], double mat2[10][10]) {
//
// double mat3[10][10];
// for (int r = 0; r < i; r++) {
//     for (int c = 0; c < j; c++) {
//         for (int in = 0; in < i; in++) {
//             mat3[r][c] += mat1[r][in] * mat2[in][c];
//         }
//         cout << mat3[r][c] << "  ";
//     }
//     cout << "\n";
// }
//


/*
 * This function writes the predicted ratings for the qual.dta data points.
 *
 * @param write_file : the name of the file to write ratings to
 */
void SVD::write_results(string write_file, string in_file){
  ifstream qual_data(in_file);
  ofstream qual_results;
  qual_results.open(write_file);
  string qual_line;
  double rating = 0.0;
  double error_sum = 0.0;
  int counter = 0;
  int u, i, d, y;

  while(getline(qual_data, qual_line)){
    istringstream iss(qual_line);

    //cout << u << " " << i << " "  << d << " " << y << "\n";
    if (!(iss >> u >> i >> d >> y)) { break; }
    u = u - 1;
    i = i - 1;
    // cout << qual_line << "\n";
    rating = predict(pu[u], qi[i], bu[u], bi[i]);
    // cout << "prediction " << rating << "\n";
    if (rating > 5.0) {
      rating = 5.0;
    }
    else if (rating < 1.0) {
      rating = 1.0;
    }

    error_sum += (y - rating) * (y - rating);
    counter += 1;
    // if (counter % 10000 == 0){
    //   cout << "in test " << counter << "\n";
    //
    qual_results << rating << "\n";
    // cout << "rating" << rating << "\n";
  }
  error_sum /= counter;
  error_sum = sqrt(error_sum);
  cout << "in test " << "error = " << error_sum << "\n";

  qual_data.close();
  qual_results.close();
}

int main(int argc, char* argv[])
{
  int latent_factors = 200;
  int epochs = 5;
  double reg1 = 0.02;
  double reg2 = 0.02;
  double learning_rate = 0.005;
  SVD* test_svd = new SVD(latent_factors, reg1, reg2, learning_rate);

  test_svd->train(OUTPUT_FILE_PATH_1, epochs);
  test_svd->write_results(RESULTS_FILE_PATH_QUAL, OUTPUT_FILE_PATH_2);
  delete test_svd;
  return 0;
}
