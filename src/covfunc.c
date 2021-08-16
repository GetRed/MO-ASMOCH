/*
 * covfunc.c
 * Covariance functions
 *
 *  Created on: Sep 18 2018
 *      Author: gongwei, BNU
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "constant.h"
#include "utility.h"
#include "covfunc.h"

int CalcCovMatrix(int nInput, int nSample, int CovIdx, int nhyp, double *hyp, double **X, double **K) {
	// calculate covariance matrix using given cov-function and hyper-parameters
	// X is the input data matrix, nSample*nInput
	// K is the returned cov-matrix, nSample*nSample
	int i, j;
	for (i = 0; i < nSample; i++) {
		for (j = 0; j <= i; j++) {
			K[i][j] = CalcCovFunc(CovIdx, nhyp, hyp, nInput, X[i], X[j]);
		}
	}
	return 1;
}

int CalcCovVector(int nInput, int nSample, int CovIdx, int nhyp, double *hyp, double **X, double *x, double *k) {
	// calculate the value cov-function using given hyper-parameters
	int i;
	for (i = 0; i < nSample; i++) {
		k[i] = CalcCovFunc(CovIdx, nhyp, hyp, nInput, X[i], x);
	}
	return 1;
}

double CalcCovFunc(int CovIdx, int nhyp, double *hyp, int nInput, double *x1, double *x2) {
	// calculate the value cov-function using given hyper-parameters
	double k;
	switch (CovIdx) {
		case 1:
			k = CovMatern3(nhyp, hyp, nInput, x1, x2);
			break;
		case 2:
			k = CovMatern5(nhyp, hyp, nInput, x1, x2);
			break;
		case 3:
			k = CovSE(nhyp, hyp, nInput, x1, x2);
			break;
		case 4:
			k = CovSEnoisefree(nhyp, hyp, nInput, x1, x2);
			break;
        case 5:
			k = CovNN(nhyp, hyp, nInput, x1, x2);
			break;
		default:
			k = CovMatern3(nhyp, hyp, nInput, x1, x2);
			break;
	}
	return k;
}

double CovMatern3(int nhyp, double *hyp, int nInput, double *x1, double *x2) {
	// Matern covariance function, GPML page85 eq4,17
	double l, sf, r, k;
	int i;
	l = hyp[0];	sf = hyp[1]; r = 0;
	for (i = 0; i < nInput; i++) {
		r += (x1[i] - x2[i]) * (x1[i] - x2[i]);
	}
	r = sqrt(r);
	k = sf*sf * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l);
	return k;
}

double CovMatern5(int nhyp, double *hyp, int nInput, double *x1, double *x2) {
	// Matern covariance function, GPML page85 eq4.17
	double l, sf, r, k; 
	int i;
	l = hyp[0];	sf = hyp[1]; r = 0;
	for (i = 0; i < nInput; i++) {
		r += (x1[i] - x2[i]) * (x1[i] - x2[i]);
	}
	r = sqrt(r);
	k = sf*sf * (1 + sqrt(5)*r/l + (5*r*r)/(3*l*l)) * exp(-sqrt(5)*r/l);
	return k;
}

double CovSE(int nhyp, double *hyp, int nInput, double *x1, double *x2) {
	// squared exponential covariance function
	double l, sf, sn, r, k;
	int i;
	l = hyp[0];	sf = hyp[1]; sn = hyp[2]; r = 0;
	for (i = 0; i < nInput; i++) {
		r += (x1[i] - x2[i]) * (x1[i] - x2[i]);
	}
	if (r > 0) {
    	k = sf*sf * exp(-0.5 * r/l/l) + sn*sn;
	} else {
		k = sf*sf * exp(-0.5 * r/l/l);
	}
	return k;
}

double CovSEnoisefree(int nhyp, double *hyp, int nInput, double *x1, double *x2) {
	// squared exponential covariance function
	double l, sf, r, k;
	int i;
	l = hyp[0];	sf = hyp[1]; r = 0;
	for (i = 0; i < nInput; i++) {
		r += (x1[i] - x2[i]) * (x1[i] - x2[i]);
	}
	k = sf*sf * exp(-0.5 * r/l/l);
	return k;
}

double CovNN(int nhyp, double *hyp, int nInput, double *x1, double *x2) {
	// neural network covariance function, GPML page91, eq4.29
    double s0, s, sum1, sum2, sum12, k;
	int i;
    s0 = hyp[0]; s = hyp[1];
    sum1 = 0; sum2 = 0; sum12 = 0;
    for (i = 0; i < nInput; i++) {
        sum1  += x1[i] * x1[i];
        sum2  += x2[i] * x2[i];
        sum12 += x1[i] * x2[i];
    }
    sum1 *= s;  sum2 *= s;  sum12 *= s;
    sum1 += s0; sum2 += s0; sum12 += s0;
    k = 2 / PI * asin( 2*sum12 / sqrt( (1+2*sum1) * (1+2*sum2) ) );
	return k;
}
