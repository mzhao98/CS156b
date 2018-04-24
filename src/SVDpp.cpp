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

// #include "SVDpp.hpp"

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

struct movie_rating {
    int id;
    int rating;
};

/*
 * A class to perform SVD on the Netflix data to get user prediction ratings
 */
class SVDpp{
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
      double **y_factor; // matrix of item factors, users x movies size
      double **Ru_array; // user x factors size

      //double **y;  // matrix of factor vectors for y_i
      // int *Ru;

      // double **data; /* input data matrix */
      // std::string train_filename;

      SVDpp(int k_factors, double regularize1, double regularize2, double learning_rate); // Constructor
      ~SVDpp(); // destructor
      double predict(double *curr_pu, double *curr_qi, double curr_bu, double curr_bi, double * ru_factor_sum);
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

SVDpp::SVDpp(int k_factors, double regularize1, double regularize2, double learning_rate)
{
    reg1 = regularize1;
    reg2 = regularize2;
    eta = learning_rate;

    k = k_factors;
    bu = new double[NUM_USERS];   // user biases
    bi = new double[NUM_MOVIES];  // movie biases
    // Ru = new int[NUM_USERS];   // number of movies a user has rated
    Ru_array = new double*[NUM_USERS];
    pu = new double*[NUM_USERS];  // user matrix
    qi = new double*[NUM_MOVIES]; // movie matrix
    y_factor = new double*[NUM_MOVIES];


    // init the bu and bi bias arrays
    for (int i = 0; i < NUM_USERS; i++){
       bu[i] = 0.0;
    }

    for (int i = 0; i < NUM_MOVIES; i++){
      bi[i] = 0.0;

    }

    // Create random number generator for generating from -0.5 to 0.5
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> dis(-0.5, 0.5); // Uniform distribution

    // create 2D array pu
    for(int i = 0; i < NUM_USERS; i++){
      pu[i] = new double[k];
      Ru_array[i] = new double[k];
    }

    // initialize pu to small random variables and Ru_array to 0.0
    int n, m;
    for (n = 0; n < NUM_USERS; n++){
	    for (m = 0; m < k; m++){
	      pu[n][m] = dis(gen);
        Ru_array[n][m] = 0.0;
	    }
	   }

    // create 2D arrays qi and y_factor
    for(int i = 0; i < NUM_MOVIES; i++){
      qi[i] = new double[k];
      y_factor[i] = new double[k];
    }

    // init qi and y_factor to all small random variables and 0 respectively
    for (n = 0; n < NUM_MOVIES; n++){
	    for (m = 0; m < k; m++){
	      qi[n][m] = dis(gen);
        y_factor[n][m] = 0.0;
	    }
	   }
}

// Destructor
SVDpp::~SVDpp() {
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
double SVDpp::train(string train_file, int iters){

    double * factor_sum = new double[k];
    double * grad_pu = new double[k];
    double * grad_qi = new double[k];
    vector<movie_rating> actual_rating;
    vector<int> Ru;
    int u, i, d, y;
    int prev_user;
    int curr_user;
    bool not_finished; // signals that we have reached end of training file

    // Shuffle data to file SHUFFLED_DATA
    for (int m = 0; m < iters; m++)
    {
      prev_user = 0;
      curr_user = 0;
      not_finished = true;
      cout << "Iter: " << m << endl;

      /* The below commented code shuffles the lines in the training data */
      // ***********************************************************************
      // string command = "gshuf " + train_file + " > " + SHUFFLED_DATA;
      // system(command.c_str());
      // cout << "Shuffled data" << endl;
      // ***********************************************************************


      // ***********************************************************************
      // - Read in training file line by line.
      // - We keep all movie_id, rating pairs for a particular user in the
      //    move_rating vector, and all movie_ids a user rated in the Ru vector.
      // - After we have collected all the ratings for a user, we update the
      //    bi, bu, qi, qu, and factor_sum arrays according to the collected
      //    data.
      // ***********************************************************************
      ifstream infile(train_file);
      string line;
      int counter = 0;
      while (not_finished){
        if (!getline(infile, line)) {
          not_finished = false; // we have reached the end of the training file
          curr_user = -1; // set to arbitrary value so that prev_user != curr_user
        }
        else {
          // read in current line, separate line into 4 data points :
          // u : user number
          // i : movie number
          // d : date number
          // y : rating
          istringstream iss(line);

          if (!(iss >> u >> i >> d >> y)) { break; }
          // Change from 1-indexed to 0-indexed
          u = u - 1;
          i = i - 1;

          curr_user = u;
        }

        // start training when we have collected all ratings for a user
        if (prev_user != curr_user){
          int user_u = prev_user;

          int Ru_size = Ru.size(); // number of movies the current user has rated
          for (int i = 0; i < k; i++) { factor_sum[i] = 0.0; }
          double ru_sqrt = pow(double(Ru_size), -1.5);
          for (int i = 0; i < Ru_size; i++) {
            for (int j = 0; j < k; j++){
              factor_sum[j] += y_factor[Ru[i]][j];
            }
          }
          for (int j = 0; j < k; j++) { factor_sum[j] *= ru_sqrt; }

          // Update
          for (int movie_index = 0; movie_index < actual_rating.size(); movie_index++){

            int movie_i = actual_rating[movie_index].id;
            int rating_y = actual_rating[movie_index].rating;

            // to define
            double error = rating_y - predict(pu[user_u], qi[movie_i],
                           bu[user_u], bi[movie_i], factor_sum);

            bu[user_u] += eta * (error - reg1 * bu[user_u]);
            bi[movie_i] += eta * (error - reg1 * bi[movie_i]);


            for (int j = 0; j < k; j++){
              grad_pu[j] = reg2 * pu[user_u][j] - error * qi[movie_i][j];
            }
            for (int j = 0; j < k; j++){
              grad_qi[j] = reg2 * qi[movie_i][j] - error * (pu[user_u][j] +
                           factor_sum[j]);
            }
            for (int j = 0; j < k; j++){
              pu[user_u][j] = pu[user_u][j] - eta*grad_pu[j];
            }
            for (int j = 0; j < k; j++){
              qi[movie_i][j] = qi[movie_i][j] - eta*grad_qi[j];
            }

            // Update y for each movie j in Ru and  for each factor f in k
            // y_factor[j] += eta*(error * ru_sqrt * q[movie_i] - reg2 * y_factor[j])
            for (int j = 0; j < Ru_size; j++) {
              for (int f = 0; f < k; f++) {
                y_factor[j][f] += eta*(error * ru_sqrt * qi[movie_i][f]
                                       - reg2 * y_factor[j][f]);
              }
            }

            // Update Ru_array for user_u for each factor f
            for (int f = 0; f < k; f++) {
              Ru_array[user_u][f] = factor_sum[f];
            }

          }
          // clear our user and ratings arrays
          Ru.clear();
          actual_rating.clear();
        }

        prev_user = curr_user;
        counter += 1;
        movie_rating curr_rating;
        curr_rating.id = i;
        curr_rating.rating = y;

        actual_rating.push_back(curr_rating);
        // Add movie ID only if it doesn't already exist in Ru
        if (!(std::find(Ru.begin(), Ru.end(), i) != Ru.end())) {
          Ru.push_back(i);
        }


        // START OF TRAINING

        //printf("old error = %f \n", error);
        // update the biases


        // use movieNumber, userNumber to index matrices for SVD
        // compute gradients of current user/ movie matrix

        //printf("new error = %f \n", new_error);
        counter += 1;
        // if (counter % 1000000 == 0){
        //   cout << "in train " << counter << "\n";
        // }

      }

    infile.close();
  }

  delete[] grad_pu;
  delete[] grad_qi;
  delete[] factor_sum;
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
double SVDpp::predict(double *curr_pu, double *curr_qi, double curr_bu,
                      double curr_bi, double *ru_factor_sum) {
    double product = 0;

    for (int i = 0; i < k; i++){
        product += (curr_pu[i] * (curr_qi[i] + ru_factor_sum[i]));
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
void SVDpp::write_results(string write_file, string in_file){
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
    rating = predict(pu[u], qi[i], bu[u], bi[i], Ru_array[u]);
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
  double learning_rate = 0.01;
  SVDpp* test_svdpp = new SVDpp(latent_factors, reg1, reg2, learning_rate);

  test_svdpp->train(FILE_PATH_SMALL, epochs);
  test_svdpp->write_results(RESULTS_FILE_PATH_QUAL, OUTPUT_FILE_PATH_2);
  delete test_svdpp;
  return 0;
}
