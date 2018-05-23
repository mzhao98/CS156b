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
#include <chrono>

using namespace std;
using namespace std::chrono;

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
#define OUTPUT_FILE_PATH_6 "../data/output_noqual.dta"

#define RESULTS_FILE_PATH_QUAL "../data/results_qual.dta"
#define SHUFFLED_DATA "../data/shuf.dta"

#define GLOBAL_MEAN 3.60952

struct movie_rating {
    int movie;
    int rating;
};

struct user_data {
    int user_id;
    vector<movie_rating> ratings;
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
      vector<user_data> all_ratings;
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
      void getData(string train_file);
      double predict(double *curr_pu, double *curr_qi, double curr_bu, double curr_bi, double * ru_factor_sum);
      void train();
      double get_error();
      void validate(string valid_file);
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


 SVDpp::SVDpp(int k_factors, double regularize1, double regularize2, double learning_rate) {
   reg1 = regularize1;
   reg2 = regularize2;
   eta = learning_rate;

   k = k_factors;
   bu = new double[NUM_USERS];   // user biases
   bi = new double[NUM_MOVIES];  // movie biases
   // Ru = new int[NUM_USERS];   // number of movies a user has rated
   pu = new double*[NUM_USERS];  // user matrix
   qi = new double*[NUM_MOVIES]; // movie matrix
   y_factor = new double*[NUM_MOVIES];
   Ru_array = new double*[NUM_USERS];


   // init the bu and bi bias arrays
   for (int i = 0; i < NUM_USERS; i++){
      bu[i] = 0.0;
   }

   for (int i = 0; i < NUM_MOVIES; i++){
     bi[i] = 0.0;

   }

   // Create random number generator for generating from -0.5 to 0.5
   double upper_dis = 0.1 * (1.0 / (sqrt(k)));
   std::random_device rd;  //Will be used to obtain a seed for the random number engine
   std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
   std::uniform_real_distribution<double> dis(0, upper_dis); // Uniform distribution

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
       Ru_array[n][m] = dis(gen);
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
    delete[] Ru_array[i];
  }
  for(int i = 0; i < NUM_MOVIES; i++){
    delete[] qi[i];
    delete[] y_factor[i];
  }
  delete[] pu;
  delete[] qi;
  delete[] y_factor;
  all_ratings.clear();

}


void SVDpp::getData(string train_file){

  vector<movie_rating> curr_user_ratings;
  int prev_user = 0;
  int curr_user = 0;
  int u, i, d, y;
  bool not_finished = true;

  /* The below commented code shuffles the lines in the training data */
  // ***********************************************************************
  // string command = "gshuf " + train_file + " > " + SHUFFLED_DATA;
  // system(command.c_str());
  // cout << "Shuffled data" << endl;
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
      user_data curr_user_data;
      // create copy of user's movie data to add to giant array
      // of user information
      vector<movie_rating> curr_user_ratings_copy(curr_user_ratings);
      curr_user_data.user_id = prev_user;
      // add curr_user_data to the big array
      curr_user_data.ratings = curr_user_ratings_copy;
      // add user into big array
      all_ratings.push_back(curr_user_data);

      // clear the list of movie ratings
      curr_user_ratings.clear();
    }
    else {
      // create a movie_rating
      movie_rating mr;
      mr.movie = i;
      mr.rating = y;
      curr_user_ratings.push_back(mr);
    }
    prev_user = curr_user;
  }
}
void SVDpp::train(){
  double * factor_sum = new double[k];
  double * grad_pu = new double[k];
  double * grad_qi = new double[k];
  int movie_id;

  for (const auto& curr_user_data: all_ratings){
    int user_u = curr_user_data.user_id;
    int Ru_size = curr_user_data.ratings.size(); // number of movies the current user has rated

    // Calculate r(t) for user u and movie i
    for (int i = 0; i < k; i++) {
      factor_sum[i] = 0.0;
    }

    double ru_sqrt = 0.0;

    if (Ru_size > 1) {
      ru_sqrt = pow(double(Ru_size), -0.5);
    }

    // Sum for all movies rated by this user
    for (int i = 0; i < Ru_size; i++) {
      for (int j = 0; j < k; j++){
        factor_sum[j] += y_factor[curr_user_data.ratings[i].movie][j];
      }
    }
    for (int j = 0; j < k; j++) {
      factor_sum[j] *= ru_sqrt;
    }

    // Update parameters bu, bi, pu, qi, and y
    for (const auto& curr_movie_rating: curr_user_data.ratings){

      int movie_i = curr_movie_rating.movie;
      int rating_y = curr_movie_rating.rating;

      // to define
      double error = rating_y - predict(pu[user_u], qi[movie_i],
                     bu[user_u], bi[movie_i], factor_sum);

      bu[user_u] += eta * (error - reg1 * bu[user_u]);
      bi[movie_i] += eta * (error - reg1 * bi[movie_i]);

      for (int j = 0; j < k; j++){
        grad_pu[j] = reg2 * pu[user_u][j] - error * qi[movie_i][j];
        grad_qi[j] = reg2 * qi[movie_i][j] - error * (pu[user_u][j] +
                     factor_sum[j]);
        pu[user_u][j] = pu[user_u][j] - eta * grad_pu[j];
        qi[movie_i][j] = qi[movie_i][j] - eta * grad_qi[j];
      }

      // Update Ru_array for user_u for each factor f
      for (int f = 0; f < k; f++) {
        Ru_array[user_u][f] = factor_sum[f];
      }

    } // Finish updating parameters bu, bi, pu, qi, and y for every movie rating

    // Update y for each movie j in Ru and for each factor f in k
    // y_factor[j] += eta*(error * ru_sqrt * q[movie_i] - reg2 * y_factor[j])
    for (int j = 0; j < Ru_size; j++) {
      movie_rating mr = curr_user_data.ratings[j];
      double error = mr.rating - predict(
                     pu[user_u], qi[mr.movie],
                     bu[user_u], bi[mr.movie], factor_sum);
      for (int f = 0; f < k; f++) {
        y_factor[mr.movie][f] += eta * (error * ru_sqrt * qi[mr.movie][f]
                                - reg2 * y_factor[mr.movie][f]);
      }
    }

  } // Finish iterating through all users
  delete[] grad_pu;
  delete[] grad_qi;
  delete[] factor_sum;
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
      product += (curr_qi[i] * (curr_pu[i] + ru_factor_sum[i]));
  }
  product += (curr_bu + curr_bi + GLOBAL_MEAN);
  //cout << "predict_result" << product << endl;

  if (product > 5.0) {
    product = 5.0;
  }
  else if (product < 1.0) {
    product = 1.0;
  }

  return product;
}


/*
 * This function uses the validation set to determine test error
 *
 * @param valid_file : the name of the file to check ratings against
 */
void SVDpp::validate(string valid_file){
  ifstream valid_data(valid_file);
  ofstream qual_results;
  string valid_line;
  double rating = 0.0;
  double error_sum = 0.0;
  int counter = 0;
  int u, i, d, y;

  while(getline(valid_data, valid_line)){
    istringstream iss(valid_line);

    //cout << u << " " << i << " "  << d << " " << y << "\n";
    if (!(iss >> u >> i >> d >> y)) { break; }
    u = u - 1;
    i = i - 1;
    // cout << qual_line << "\n";
    rating = predict(pu[u], qi[i], bu[u], bi[i], Ru_array[u]);
    // cout << "prediction " << rating << "\n";

    error_sum += (y - rating) * (y - rating);
    counter += 1;
  }
  if (counter != 0) {
    error_sum /= counter;
  }
  error_sum = sqrt(error_sum);
  cout << "in test " << "error = " << error_sum << "\n";

  valid_data.close();
}


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
  int u, i, d;

  while(getline(qual_data, qual_line)){
    istringstream iss(qual_line);

    //cout << u << " " << i << " "  << d << " " << y << "\n";
    if (!(iss >> u >> i >> d)) { break; }
    u = u - 1;
    i = i - 1;
    // cout << qual_line << "\n";
    rating = predict(pu[u], qi[i], bu[u], bi[i], Ru_array[u]);
    // cout << "prediction " << rating << "\n";

    qual_results << rating << "\n";
  }
  cout << "finished writing predictions" << "\n";

  qual_data.close();
  qual_results.close();
}

int main(int argc, char* argv[])
{
  int latent_factors = 200;
  int epochs = 30;
  double reg1 = 0.005;
  double reg2 = 0.015;
  double learning_rate = 0.007;
  cout << "creating svdpp" << endl;
  SVDpp* test_svd = new SVDpp(latent_factors, reg1, reg2, learning_rate);
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  test_svd->getData(OUTPUT_FILE_PATH_6);
  cout << "data obtained. training now" << endl;
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>( t2 - t1 ).count();
  cout << (duration* (.000001)) << "\n";
  for (int iter = 0; iter < epochs; iter++) {
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    cout << iter << "\n";
    test_svd->train();
    test_svd->eta *= 0.9;
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    cout << (duration* (.000001)) << "\n";
  }

  test_svd->validate(OUTPUT_FILE_PATH_2);
  test_svd->write_results(RESULTS_FILE_PATH_QUAL, FILE_PATH_QUAL);
  delete test_svd;
  return 0;
}
