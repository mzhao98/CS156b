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

#define RESULTS_FILE_PATH_QUAL "../data/results_qual_knn.dta"

#define GLOBAL_MEAN 3.5126

/*
 * A class to perform SVD on the Netflix data to get user prediction ratings
 */
class KNN{
  public:
    int K;
    int ** users;
    double *averages;

    KNN(int Kval);
    ~KNN();
    void LoadData(string train_file);
    double PearsonCorrelationCoefficient(int *ratings_user1, int *ratings_user2);
    int *getNeighbors(int user, int movie);
    double getRating(int user, int movie, int *neighbors);
    void write_results(string test_file, string output_file);
};

KNN::KNN(int Kval){
  K = Kval;
  averages = new double[NUM_MOVIES];
  users = new int*[NUM_USERS];

  for(int i = 0; i < NUM_USERS; i++){
      users[i] = new int[NUM_MOVIES];
  }

    // init qi to all 0.1
  for (int n = 0; n < NUM_USERS; n++){
    for (int m = 0; m < NUM_MOVIES; m++){
      users[n][m] = 0;
    }
	}

  for (int n = 0; n < NUM_USERS; n++)
  {
    averages[n] = 0;
  }

}

// Destructor
KNN::~KNN() {
  delete[] averages;
  delete[] users;
  for (int i = 0; i < NUM_MOVIES; i++)
  {
    delete users[i];
  }
}


void KNN::LoadData(string train_file){
  // reading file line by line
  ifstream infile(train_file);
  string line;
  // while not the end of the file
  while (getline(infile, line)){
    // read in current line, separate line into 4 data points
    // user number, movie number, date number, rating
    istringstream iss(line);
    // u : user number
    // i : movie number
    // d : date number
    // y : rating
    int u, i, d, r;
    //cout << u << " " << i << " "  << d << " " << y << "\n";
    if (!(iss >> u >> i >> d >> r)) { break; }
    // Change from 1-indexed to 0-indexed
    u = u - 1;
    i = i - 1;
    users[u][i] = r;
  }
  infile.close();

  // calculate average rating for each movie
  for (int i = 0; i < NUM_MOVIES; i++){
    int sum = 0;
    int length = 0;
    for (int j = 0; j < NUM_USERS; i++){
      if(users[j][i] != 0){
        length += 1;
      }
      sum += users[j][i];
    }
    averages[i] = (sum + 0.0)/length;
  }

}



double KNN::PearsonCorrelationCoefficient(int * ratings_user1, int * ratings_user2){
  int sum_U1 = 0, sum_U2 = 0, sum_U12 = 0;
  int squareSum_U1 = 0, squareSum_U2 = 0;
  // keeps track of number of movies both users have rated
  int n = 0;
  double corr = 0;

  for (int i = 0; i < NUM_MOVIES; i++)
  {
      // sum of ratings of user 1
      sum_U1 = sum_U1 + ratings_user1[i];

      // sum of ratings of user 2
      sum_U2 = sum_U2 + ratings_user2[i];

      // sum of ratings of user 1 * ratings of user 2
      sum_U12 = sum_U12 + ratings_user1[i] * ratings_user2[i];

      // if both users have rated this movie, increment n
      if (ratings_user1[i] * ratings_user2[i] != 0){
        n ++;
      }
      // sum of square of ratings of user 1 and 2
      squareSum_U1 = squareSum_U1 + ratings_user1[i] * ratings_user1[i];
      squareSum_U2 = squareSum_U2 + ratings_user2[i] * ratings_user2[i];
  }

  // use formula for calculating correlation coefficient.
  if (n != 0)
  {
    corr = (double)(n * sum_U12 - sum_U1 * sum_U2)/ sqrt((n * squareSum_U1 - sum_U1 * sum_U1) * (n * squareSum_U2 - sum_U2 * sum_U2));
  }

  return corr;
}



int* KNN::getNeighbors(int user, int movie){
  double *distances;
  int *neighbors;
  double corr;
  distances = new double[NUM_USERS];
  neighbors = new int[K];

  for (int i = 0; i < NUM_USERS; i++)
  {
    if(i == user){
      corr = 0;
    }
    else{
      corr = PearsonCorrelationCoefficient(users[user], users[i]);
    }
    distances[i] = corr;
  }
  for (int i = 0; i < K; i++)
  {
    int max_val = distance(distances, max_element(distances, distances + NUM_USERS));
    neighbors[i] = max_val;
    distances[max_val] = 0;
  }
  delete[] distances;
  return neighbors;
}

double KNN::getRating(int user, int movie, int* neighbors){
  int sum = 0;
  int count = 0;
  for (int i = 0; i < K; i++){
    if (users[neighbors[i]][movie]!= 0){
      count += 1;
    }
    sum += users[neighbors[i]][movie];
  }
  if(count == 0){
    return averages[movie];
  }
  return (sum + 0.0)/count;
}


void KNN::write_results(string test_file, string output_file){
  ifstream qual_data(test_file);
  ofstream qual_results;
  qual_results.open(output_file);
  string qual_line;
  double rating = 0.0;
  double error_sum = 0.0;
  int counter = 0;
  int u, i, d, y;

  while(getline(qual_data, qual_line)){
    istringstream iss(qual_line);

    //cout << u << " " << i << " "  << d << " " << y << "\n";
    if (!(iss >> u >> i >> d)) { break; }
    u = u - 1;
    i = i - 1;
    // cout << qual_line << "\n";
    rating = getRating(u, i, getNeighbors(u, i));

    // cout << "prediction " << rating << "\n";
    if (rating > 5.0) {
      rating = 5.0;
    }
    else if (rating < 1.0) {
      rating = 1.0;
    }

    //error_sum += (y - rating) * (y - rating);
    //counter += 1;
    //cout << counter << "\n";
    qual_results << rating << "\n";
    // cout << "rating" << rating << "\n";
  }
  //error_sum /= counter;
  //error_sum = sqrt(error_sum);
  //cout << "error = " << error_sum << "\n";

  qual_data.close();
  qual_results.close();
}

int main(int argc, char* argv[])
{
  cout << "create KNN object" << endl;
  KNN test_KNN = KNN(5);
  cout << "load data" << endl;
  test_KNN.LoadData(FILE_PATH_SMALL);
  cout << "write results" << endl;

  test_KNN.write_results(FILE_PATH_QUAL, RESULTS_FILE_PATH_QUAL);
  return 0;

}
