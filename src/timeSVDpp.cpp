#include <algorithm>
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
#include <unordered_map>
#include <math.h>

using namespace std;
using namespace std::chrono;

// #include "SVD.hpp"

#define NUM_USERS 458293
#define NUM_MOVIES 17770
#define MAX_DATE 2243
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

#define GLOBAL_MEAN 3.60952

struct movie_rating {
    int movie;
    int rating;
    int date;
};

struct user_data {
    int user_id;
    vector<movie_rating> ratings;
    double mean_date;
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
      double **but; // bias for users and time
      double **bibin; // bias for movies and time bins
      double *au; // time dependency term for each user
      double **pu; // matrix of users
      double **qi; // matrix of movies
      vector<user_data> all_ratings;
      unordered_map <int, int> user_data_map;
      int k;       // number of latent factors
      double beta; // regularization for time term
      double reg1; // first regularization term
      double reg2; // second regularization term
      double reg3; // third regularization term - used for bibin
      double reg4; // 4th regularization term - used for au
      double eta1;  // the learning
      double eta2; // used in au
      double **y_factor; // matrix of item factors, users x movies size
      double **Ru_array; // user x factors size
      int n_bins;
      int max_date;

      //double **y;  // matrix of factor vectors for y_i
      // int *Ru;

      // double **data; /* input data matrix */
      // std::string train_filename;

      SVDpp(int bins, int k_factors, double regularize1, double regularize2,
            double regularize3, double regularize4, double et1, double et2, double b); // Constructor
      ~SVDpp(); // destructor
      int getSign(double time_diff);
      int getBinNumber(int date);
      void getData(string train_file);
      double predict(int user, int movie, int date, double * ru_factor_sum);
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


 SVDpp::SVDpp(int bins, int k_factors, double regularize1, double regularize2,
              double regularize3, double regularize4, double et1, double et2,
              double b) {
   reg1 = regularize1;
   reg2 = regularize2;
   reg3 = regularize3;
   reg4 = regularize4;
   eta1 = et1;
   eta2 = et2;
   beta = b;

   n_bins = bins;
   k = k_factors;
   bu = new double[NUM_USERS];   // user biases
   bi = new double[NUM_MOVIES];  // movie biases

   but = new double*[NUM_USERS]; // bias for users and time
   bibin = new double*[NUM_MOVIES]; // bias for movies and time bins
   au = new double[NUM_USERS]; // time dependency term for each user

   // Ru = new int[NUM_USERS];   // number of movies a user has rated
   pu = new double*[NUM_USERS];  // user matrix
   qi = new double*[NUM_MOVIES]; // movie matrix
   y_factor = new double*[NUM_MOVIES];
   Ru_array = new double*[NUM_USERS];


   // init the bu, bi, but, bibin, and au bias arrays
   for (int i = 0; i < NUM_USERS; i++){
      bu[i] = 0.0;
      au[i] = 0.0;
      but[i] = new double[MAX_DATE];
      for (int j = 0; j < MAX_DATE; j++){
        but[i][j] = 0.0;
      }
   }

   for (int i = 0; i < NUM_MOVIES; i++) {
     bi[i] = 0.0;
     bibin[i] = new double[n_bins];
     for (int j = 0; j < n_bins; j++) {
       bibin[i][j] = 0.0;
     }
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
  delete[] au;
  for (int i = 0; i < NUM_USERS; i++){
    delete[] pu[i];
    delete[] Ru_array[i];
    delete[] but[i];
  }
  for(int i = 0; i < NUM_MOVIES; i++){
    delete[] qi[i];
    delete[] y_factor[i];
    delete[] bibin[i];
  }
  delete[] pu;
  delete[] qi;
  delete[] y_factor;
  delete[] but;
  delete[] bibin;
  all_ratings.clear();

}

int SVDpp::getSign(double time_diff) {
  if (time_diff > 0) { return 1; }
  else { return 0; }
}

int SVDpp::getBinNumber(int date) {
  int bin_size = (MAX_DATE / n_bins) + 1;
  return (date/bin_size);
}

void SVDpp::getData(string train_file){

  vector<movie_rating> curr_user_ratings;
  int prev_user = 0;
  int curr_user = 0;
  int u, i, d, y;
  bool not_finished = true;
  double sum_dates = 0;
  int user_data_counter = 0;

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
      d = d - 1;
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
      curr_user_data.mean_date = sum_dates / curr_user_ratings_copy.size();
      // add user into big array
      all_ratings.push_back(curr_user_data);
      user_data_map[user_u] = user_data_counter;

      // clear the list of movie ratings
      curr_user_ratings.clear();
      sum_dates = 0;
      user_data_counter += 1;
    }
    else {
      // create a movie_rating
      movie_rating mr;
      mr.movie = i;
      mr.rating = y;
      mr.date = d;
      sum_dates += d;
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

    if (Ru_size >= 1) {
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
      int date_d = curr_movie_rating.date;
      int bin_b = getBinNumber(date_d);

      // to define
      double error = rating_y - predict(user_u, movie_i, date_d, factor_sum);

      bu[user_u] += eta1 * (error - reg1 * bu[user_u]);
      bi[movie_i] += eta1 * (error - reg1 * bi[movie_i]);
      but[user_u][date_d] += eta1 * (error - reg1 * but[user_u][date_d]);
      bibin[movie_i][bin_b] += eta1 * (error - reg3 * bibin[movie_i][bin_b]);

      double time_diff = date_d - curr_user_data.mean_date;
      double devu = getSign(time_diff) * pow(abs(time_diff), beta);
      au[user_u] += eta2 * (error * devu - reg4 * au[user_u]);

      for (int j = 0; j < k; j++){
        grad_pu[j] = reg2 * pu[user_u][j] - error * qi[movie_i][j];
        grad_qi[j] = reg2 * qi[movie_i][j] - error * (pu[user_u][j] +
                     factor_sum[j]);
        pu[user_u][j] = pu[user_u][j] - eta1 * grad_pu[j];
        qi[movie_i][j] = qi[movie_i][j] - eta1 * grad_qi[j];
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
                     user_u, mr.movie, mr.date, factor_sum);
      for (int f = 0; f < k; f++) {
        y_factor[mr.movie][f] += eta1 * (error * ru_sqrt * qi[mr.movie][f]
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
double SVDpp::predict(int curr_user, int curr_movie, int curr_time,
                      double *ru_factor_sum) {
  double product = 0.0;
  double time_diff = 0.0;

  for (int i = 0; i < k; i++){
      product += (pu[curr_user][i] + ru_factor_sum[i]) * qi[curr_movie][i];
  }
  product += bu[curr_user] + bi[curr_movie] + GLOBAL_MEAN;

  if (user_data_map.find(curr_user) != user_data_map.end()) {
    time_diff = curr_time - all_ratings[user_data_map[curr_user]].mean_date;
  }
  double devu = getSign(time_diff) * pow(abs(time_diff), beta);
  int curr_bin = getBinNumber(curr_time);
  product += au[curr_user] * devu + but[curr_user][curr_time] +
             bibin[curr_movie][curr_bin];

  //cout << "predict_result" << product << endl;

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
    d = d - 1;
    // cout << qual_line << "\n";
    rating = predict(u, i, d, Ru_array[u]);
    // cout << "prediction " << rating << "\n";
    if (rating > 5.0) {
      rating = 5.0;
    }
    else if (rating < 1.0) {
      rating = 1.0;
    }

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
    d = d - 1;
    // cout << qual_line << "\n";
    rating = predict(u, i, d, Ru_array[u]);
    // cout << "prediction " << rating << "\n";
    if (rating > 5.0) {
      rating = 5.0;
    }
    else if (rating < 1.0) {
      rating = 1.0;
    }
    qual_results << rating << "\n";
  }
  cout << "finished writing predictions" << "\n";

  qual_data.close();
  qual_results.close();
}

int main(int argc, char* argv[])
{
  int latent_factors = 200;
  int epochs = 1;
  double reg1 = 0.005;
  double reg2 = 0.015;
  double reg3 = 0.08;
  double reg4 = 0.0004;
  double eta1 = 0.007;
  double eta2 = 0.00001;
  int bins = 30;
  double beta = 0.4;

  cout << "creating timesvdpp" << endl;
  SVDpp* test_svd = new SVDpp(bins, latent_factors, reg1, reg2, reg3, reg4, eta1,
                              eta2, beta);
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  test_svd->getData(OUTPUT_FILE_PATH_1);
  cout << "data obtained. training now" << endl;
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>( t2 - t1 ).count();
  cout << (duration* (.000001)) << "\n";
  for (int iter = 0; iter < epochs; iter++) {
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    cout << iter << "\n";
    test_svd->train();
    test_svd->eta1 *= 0.9; // scale down the learning rate
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    cout << (duration* (.000001)) << "\n";
  }

  test_svd->validate(OUTPUT_FILE_PATH_2);
  test_svd->write_results(RESULTS_FILE_PATH_QUAL, FILE_PATH_QUAL);
  delete test_svd;
  return 0;
}
