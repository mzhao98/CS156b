#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <numeric>

using namespace std;

#define IDX_FILE           "../data/all.idx"
#define FILE_PATH_ALL      "../data/all.dta"
#define FILE_PATH_SMALLER  "../data/smaller.dta"
#define OUTPUT_FILE_PATH_1 "../data/output_base.dta"
#define OUTPUT_FILE_PATH_2 "../data/output_valid.dta"
#define OUTPUT_FILE_PATH_3 "../data/output_hidden.dta"
#define OUTPUT_FILE_PATH_4 "../data/output_probe.dta"
#define NUM_RATINGS 102416306.0


/*
 * This function uses the all.idx file to separate the all.dta into
 * individual files by data type [base, valid, hidden, probe, qual].
 */
int parse_by_index() {
  ifstream all_data_file(FILE_PATH_ALL);
  ifstream idx_file(IDX_FILE);
  ofstream out_file_1;
  ofstream out_file_2;
  ofstream out_file_3;
  ofstream out_file_4;

  string data_line;
  string idx_line;
  int index;

  // if (!out_file_1) {
  //       cout << "Unable to open file";
  //       exit(1);
  //   }
  out_file_1.open(OUTPUT_FILE_PATH_1);
  out_file_2.open(OUTPUT_FILE_PATH_2);
  out_file_3.open(OUTPUT_FILE_PATH_3);
  out_file_4.open(OUTPUT_FILE_PATH_4);

  while (getline(all_data_file, data_line) && getline(idx_file, idx_line)) {
    istringstream iss_idx(idx_line);
    istringstream iss_data(data_line);

    if (!(iss_idx >> index)) { break; }

    if (index == 1) {
      out_file_1 << data_line << "\n";
    }
    else if (index == 2) {
      out_file_2 << data_line << "\n";
    }
    else if (index == 3) {
      out_file_3 << data_line << "\n";
    }
    else if (index == 4) {
      out_file_4 << data_line << "\n";
    }
  }

  out_file_1.close();
  out_file_2.close();
  out_file_3.close();
  out_file_4.close();
  idx_file.close();
  all_data_file.close();

  return 0;
}

/*
 * This function finds the overall mean of ratings in the all.dta file.
 */
int find_overall_mean() {
  double kahan_sum = 0.0;
  double correction = 0.0;
  double y_correction = 0.0;
  double t_correction = 0.0;

  ifstream infile(FILE_PATH_ALL);
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
    int u, i, d, y;
    if (!(iss >> u >> i >> d >> y)) { break; }
    y_correction = y  - correction;
    t_correction = kahan_sum + y_correction;
    correction = (t_correction - kahan_sum) - y_correction;
    kahan_sum = t_correction;
  }
  double average_rating = kahan_sum / NUM_RATINGS;
  cout << average_rating << endl;

  return 0;
}



int main(int argc, char* argv[])
{
  parse_by_index();
  return 0;

}
