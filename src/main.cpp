#include "jlann_ds.hpp"
#include "ANN/ANN.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>
#include <string>
#include <boost/program_options.hpp>
#include <sstream>
#include <chrono>
#include <sys/resource.h>

using namespace std;
namespace po = boost::program_options;

using dtype = double;
using JLMatrix = typename jlann::JLAnnDS<dtype>::JLMatrix;
using JLVector = typename jlann::JLAnnDS<dtype>::JLVector;

class Timer {
public:
	Timer() { start_time = std::chrono::high_resolution_clock::now(); }

	void start() {
		start_time = std::chrono::high_resolution_clock::now();
	}

	double end() {
		return elapsed_seconds();
	}

	double elapsed_seconds() {
		auto end_time = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
		return elapsed.count();
	}
private:
	decltype(std::chrono::high_resolution_clock::now()) start_time;
};

template <typename T>
void parse_csv(string filename, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dataset) {
	ifstream fin(filename.c_str());
	int numberOfPoints;
	int dimension;
	fin >> numberOfPoints >> dimension;

	dataset.resize(numberOfPoints, dimension);

	string line;
	string line2;
	getline(fin, line);
	int i=0;
	while (getline(fin, line)) {
		istringstream iss(line);
		int j=0;
		while (getline(iss, line2, ',')) {
			T tmp;
			istringstream(line2) >> tmp;
			dataset(i, j++) = tmp;
			//point_array[i][j++] = tmp;
		}
		i++;
		if (i==numberOfPoints) {
			break;
		}
	}
	fin.close();
}

template <typename T>
void read_file(string filename, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dataset) {
	if (filename.find(".csv")!=string::npos) {
		parse_csv<T>(filename, dataset);
	}
}

void queryJl(jlann::JLAnnDS<dtype>* jlAnnDs, double epsilon, JLMatrix& dataset, JLMatrix& queries,
				Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& groundtruth) {
	int numberOfQueries = queries.rows();
	int trueNn = 0;
	int ann = 0;
	double annQueryTime = 0;
	//for (int i=0; i<numberOfQueries; i++) {
	//	Timer timer;
	//	JLVector query = queries.row(i);
	//	int nnIndex = jlAnnDs->findNearestNeighbor(query, epsilon);
	//	annQueryTime += timer.elapsed_seconds();

	//	double nnDist = (dataset.row(groundtruth(i, 0))-query).norm();
	//	double annDist = (dataset.row(nnIndex)-query).norm();
	//	//cout << annDist << " <= " << "(1+" << epsilon << ")*" << nnDist << " = " << (1+epsilon)*nnDist << endl;

	//	if (nnIndex==groundtruth(i, 0)) {
	//		++trueNn;
	//	}
	//	for (int j=1; j<groundtruth.cols(); j++) {
	//		if (nnIndex==groundtruth(i, j)) {
	//			++ann;
	//		}
	//	}
	//}
	Timer batchTimer;
	std::vector<int> nns = jlAnnDs->findNearestNeighbors(queries, epsilon);
	annQueryTime = batchTimer.elapsed_seconds();

	for (int i=0; i<numberOfQueries; i++) {
		double nnDist = (dataset.row(groundtruth(i, 0))-queries.row(i)).norm();
		//double annDist = (dataset.row(nnIndex)-query).norm();
		//cout << annDist << " <= " << "(1+" << epsilon << ")*" << nnDist << " = " << (1+epsilon)*nnDist << endl;

		if (nns[i]==groundtruth(i, 0)) {
			++trueNn;
		}
		for (int j=1; j<groundtruth.cols(); j++) {
			if (nns[i]==groundtruth(i, j)) {
				++ann;
			}
		}
	}

	cout << "Total query time: " << annQueryTime << " -- Average query time: " << annQueryTime/numberOfQueries << endl;
	cout << "True nns percentage: " << (double)trueNn/numberOfQueries*100 << "%" << endl;
	cout << "Misses percentage: " << (double)(numberOfQueries-ann)/numberOfQueries*100 << "%" << endl;
}

int main(int argc, char* argv[]) {
	boost::program_options::options_description desc;
	desc.add_options()
		("input,i", po::value<string>(), "The input file (can be .ine, .csv); mandatory.")
		("query,q", po::value<string>(), "The query file (can be .ine, .csv); mandatory.")
		("groundtruth,g", po::value<string>(), "The groundtruth file (can be .ine, .csv); mandatory.")
		("e", po::value<double>()->value_name("Val")->default_value(0.1), "The approximation factor")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("input")==0) {
		cout << desc << endl;
		return 1;
	}

	if (vm.count("query")==0) {
		cout << desc << endl;
		return 1;
	}

	if (vm.count("groundtruth")==0) {
		cout << desc << endl;
		return 1;
	}

	if (vm.count("e")==0) {
		cout << desc << endl;
		return 1;
	}

	string filename = vm["input"].as<string>();
	string query_filename = vm["query"].as<string>();
	string groundtruth_filename = vm["groundtruth"].as<string>();
	double epsilon = vm["e"].as<double>();

	cout << "Reading files. " << endl;
	JLMatrix dataset;
	read_file<dtype>(filename, dataset);
	cout << "Read dataset." << endl;

	JLMatrix queries;
	read_file<dtype>(query_filename, queries);
	cout << "Read queries." << endl;

	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> groundtruth;
	read_file<int>(groundtruth_filename, groundtruth);
	cout << "Read groundtruth." << endl;

	Timer timer;
	std::cout << "Creating jlann ds." << std::endl;
	int projDimension = 16;//(int)2*std::ceil(std::log2(dataset.rows())/(std::log2(std::log2(dataset.rows()))));
	int projSearchSize = (int)std::sqrt(dataset.rows());
	jlann::JLAnnDS<dtype>* jlAnnDs = new jlann::JLAnnDS<dtype>(dataset, 1, projDimension, projSearchSize);
	jlAnnDs->createDs();
	double jlannPreprocessingTime = timer.end();
	cout << "Constructing jlann dataset took " << jlannPreprocessingTime << "s" << endl;

	queryJl(jlAnnDs, epsilon, dataset, queries, groundtruth);

    delete jlAnnDs;
	annClose();
}
