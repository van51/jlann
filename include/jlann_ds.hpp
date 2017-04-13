/**
 * Copyright 2017 Evangelos Anagnostopoulos
   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __JLANN_DS_HPP__
#define __JLANN_DS_HPP__

#include <Eigen/Dense>
#include <ANN/ANN.h>
#include <iostream>
#include <cmath>
#include <random>

#include <sys/resource.h>

namespace jlann{

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DMatrix;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FMatrix;

template<typename T>
class JLAnnDS {
public:
	typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> JLMatrix;
	typedef Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor> JLVector;

	JLAnnDS(JLMatrix& dataPoints, int numOfDs=1, int projectedDimension=0, int searchSize=0):
		originalPoints(dataPoints),
		numberOfDs(numOfDs),
		projDimension(projectedDimension),
		projSearchSize(searchSize) {

		numberOfPoints = originalPoints.rows();
		dimension = originalPoints.cols();

		if (projDimension<=0) {
			projDimension = (int)std::ceil(std::log2(numberOfPoints)/(std::log2(std::log2(numberOfPoints))));
		}
		if (numberOfDs<=0) {
			numberOfDs = 1;
		}
		if (projSearchSize<=0) {
			projSearchSize = (int)std::ceil(std::sqrt(numberOfPoints));
		}

		annIdx = new ANNidx[projSearchSize];
		dists = new ANNdist[projSearchSize];
		annQueryPt = annAllocPt(projDimension);


		//createDs();
	}

	void createDs() {
		this->bdTree.reserve(numberOfDs);
		this->projectionMatrices.reserve(numberOfDs);
        std::mt19937 generator;
        std::normal_distribution<double> distribution(0.0,1.0);
		struct rlimit limit;

		for (int currentDs=0; currentDs<numberOfDs; currentDs++) {
			JLMatrix* projMatrix = new JLMatrix(dimension, projDimension);

        	for (int i=0; i<dimension; i++) {
        	    for (int j=0; j<projDimension; j++) {
        	        (*projMatrix)(i, j) = distribution(generator);
        	    }
        	}
			JLMatrix projectedData = originalPoints*(*projMatrix);

        	ANNpointArray point_array = annAllocPts(numberOfPoints, projDimension);
			auto dataPointer = projectedData.data();
       	    for (int i=0; i<numberOfPoints; i++) {
	        	for (int j=0; j<projDimension; j++) {
        	        point_array[i][j] = (*dataPointer);
					++dataPointer;
        	    }
        	}

			this->projectionMatrices.push_back(projMatrix);
			this->pointArray.push_back(point_array);
			ANNbd_tree* bdtree = new ANNbd_tree(
        	    point_array,
        	    numberOfPoints,
        	   	projDimension,
				2
        	);
			this->bdTree.push_back(bdtree);
		}
	}

	std::vector<int> findNearestNeighbors(JLMatrix& queries, double epsilon) {
		std::vector<int> nns;
		std::vector<double> nnDists;
		nns.reserve(queries.rows());
		nnDists.reserve(queries.rows());
		for (int i=0; i<queries.rows(); i++){
			nns.push_back(-1);
			nnDists.push_back(-1);
		}
		for (int currentDs=0; currentDs<numberOfDs; currentDs++) {
			JLMatrix projectedQueries = queries * (*projectionMatrices[currentDs]);
			auto queriesData = projectedQueries.data();
			for (int i=0; i<projectedQueries.rows(); i++) {
				auto queryPoint = queries.row(i);
				for (int j=0; j<projDimension; j++) {
					annQueryPt[j] = *(queriesData);
					queriesData++;
				}

	        	this->bdTree[currentDs]->annkSearch(
	        	    annQueryPt,
	        	    projSearchSize,
	        	    annIdx,
	        	    dists,
	        	    epsilon
	        	);
	
	        	for (int k=0; k<projSearchSize; k++) {
	        	    if (annIdx[k]==-1) {
	        	        break;
	        	    }
					double distance = (queryPoint-originalPoints.row(annIdx[k])).norm();
	        	    if (distance<nnDists[i] || nnDists[i]<0) {
	        	        nnDists[i] = distance;
	        	        nns[i] = annIdx[k];
	        	    }
	        	}
			}
		}
		return nns;
	}

	int findNearestNeighbor(JLVector& queryPoint, double epsilon) {

		int minDistance=-1;
		int nearestNeighborIdx=-1;
		for (int currentDs=0; currentDs<numberOfDs; currentDs++) {
        	for (int i=0; i<projDimension; i++) {
				annQueryPt[i] = 0;
        	    for (int j=0; j<dimension; j++) {
        	        annQueryPt[i] += queryPoint(j) * (*projectionMatrices[currentDs])(j, i);
        	    }
        	}
        	this->bdTree[currentDs]->annkSearch(
        	    annQueryPt,
        	    projSearchSize,
        	    annIdx,
        	    dists,
        	    epsilon
        	);

        	for (int i=0; i<projSearchSize; i++) {
        	    if (annIdx[i]==-1) {
        	        break;
        	    }
				double distance = (queryPoint-originalPoints.row(annIdx[i])).norm();
        	    if (distance<minDistance || minDistance<0) {
        	        minDistance = distance;
        	        nearestNeighborIdx = annIdx[i];
        	    }
        	}
		}

		return nearestNeighborIdx;
	}

	~JLAnnDS() {
		for (int i=0; i<pointArray.size(); i++) {
			annDeallocPts(pointArray[i]);
		}
		for (int i=0; i<projectionMatrices.size(); i++) {
			delete projectionMatrices[i];
		}
		for (int i=0; i<bdTree.size(); i++) {
			delete bdTree[i];
		}
		annDeallocPt(annQueryPt);
		delete []annIdx;
		delete []dists;
	}

private:
	std::vector<JLMatrix*> projectionMatrices;
	std::vector<ANNbd_tree*> bdTree;
	JLMatrix& originalPoints;
	std::vector<ANNpointArray> pointArray;
	ANNidxArray annIdx;
	ANNdistArray dists;
	ANNpoint annQueryPt;
	int numberOfDs;
	int projDimension;
	int projSearchSize;
	int dimension;
	int numberOfPoints;
}; //class JLAnnDs
} //namespace jlann
#endif //__JLANN_DS_HPP__
