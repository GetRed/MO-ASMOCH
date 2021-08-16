/*
 * gpr.c
 * Gaussian Processes Regression
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
#include "gpr.h"

double GPtrain(int CovIdx, int nhyp, double *hyp, double noise, int nInput, int nSample, 
	double **X, double *y, double **K, double **L, double *a)
{
	// train Gaussian Processes Regression
	double m;
	int i;

	CalcCovMatrix(nInput, nSample, CovIdx, nhyp, hyp, X, K);
	for (i = 0; i < nSample; i++) K[i][i] += noise*noise;

	chol(nSample, K, L);

	for (i = 0; i < nSample; i++) a[i] = y[i];
	LUsym(nSample, a, L);

	m = mgnlike(nSample, y, a, L);

	return m;
}

double GPtrain_r(int CovIdx, int nhyp, double *hyp, double noise, 
	int nInput, int nSample1, int nSample2, double **X1, double **X2, double **XX,
	double *y1, double *y2, double *yy, double **K, double **L, double *a,
	double **KK, double **LL, double *aa)
{
	// train Gaussian Processes Regression with reinforcement
	double m;
	int i, j, nSample;

	nSample = nSample1 + nSample2;
	for (i = 0; i < nSample; i++) {
		for (j = 0; j < nInput; j++) {
			if (i < nSample1) {
				XX[i][j] = X1[i][j];
			} else {
				XX[i][j] = X2[i-nSample1][j];
			}
		}
	}

	for (i = 0; i < nSample; i++) {
		for (j = 0; j < nSample; j++) {
			if (j > i) {
				KK[i][j] = 0;
			} else if (i < nSample1) {
				KK[i][j] = K[i][j];
			} else {
				KK[i][j] = CalcCovFunc(CovIdx, nhyp, hyp, nInput, XX[i], XX[j]);
			}
		}
	}
	
	for (i = nSample1; i < nSample2; i++) KK[i][i] += noise*noise;

	chol_r(nSample1, nSample, KK, L, LL);

	for (i = 0; i < nSample1; i++) yy[i] = y1[i];
	for (i = nSample1; i < nSample; i++) yy[i] = y2[i-nSample1];
	for (i = 0; i < nSample; i++) aa[i] = yy[i];
	LUsym(nSample, aa, LL);

	m = mgnlike(nSample, yy, aa, LL);

	return m;
}

int GPpredict(int CovIdx, int nhyp, double *hyp, 
	int nInput, int nSample1, int nSample2, double **X1, double **X2, 
	double *y, double *pv, double **L, double *a)
{
	// predict Gaussian Processes Regression
	double *kstar, *v, k;
	int i, j;

	kstar = DoubleVector(nSample1);
	v     = DoubleVector(nSample1);

	for (j = 0; j < nSample2; j++) {
		k = CalcCovFunc(CovIdx, nhyp, hyp, nInput, X2[j], X2[j]);
		CalcCovVector(nInput, nSample1, CovIdx, nhyp, hyp, X1, X2[j], kstar);
		y[j] = 0;
		for (i = 0; i < nSample1; i++) {
			y[j] += kstar[i] * a[i];
			v[i] = kstar[i];
		}
		Linv(nSample1, v, L);
		pv[j] = k;
		for (i = 0; i < nSample1; i++) pv[j] -= v[i]*v[i];
	}
    
	FreeDoubleVector(kstar);
	FreeDoubleVector(v);
	return 1;
}

int chol(int n, double **A, double **L) {
	// cholesky decomposition
	double s;
	int i, j, k;
	for (j = 0; j < n; j++) {
		s = 0;
		for (k = 0; k < j; k++) {
			s += L[j][k] * L[j][k];
		}
		L[j][j] = sqrt(A[j][j] - s);
		for (i = j+1; i < n; i++) {
			s = 0;
			for (k = 0; k < j; k++) {
				s += L[i][k] * L[j][k];
			}
			L[i][j] = (A[i][j] - s) / L[j][j];
		}
	}
	return 1;
}

int chol_r(int n1, int n2, double **A, double **L1, double **L2) {
	// cholesky decomposition with reinforcement
	// n1: dimension of original matrix (has been decomposed)
	// n2: dimension of updated matrix (needs decomposition)
	// n2-n1: dimension of reinforcement
	// A: matrix needs decomposition n2*n2
	// L1: n1*n1
	// L2: n2*n2
	double s;
	int i, j, k;
	for (j = 0; j < n1; j++) {
		for (i = 0; i < n1; i++) {
			L2[i][j] = L1[i][j];
		}
	}

	for (j = 0; j < n1; j++) {
		for (i = n1; i < n2; i++) {
			s = 0;
			for (k = 0; k < j; k++) {
				s += L2[i][k] * L2[j][k];
			}
			L2[i][j] = (A[i][j] - s) / L2[j][j];
		}
	}

	for (j = n1; j < n2; j++) {
		s = 0;
		for (k = 0; k < j; k++) {
			s += L2[j][k] * L2[j][k];
		}
		L2[j][j] = sqrt(A[j][j] - s);
		for (i = j+1; i < n2; i++) {
			s = 0;
			for (k = 0; k < j; k++) {
				s += L2[i][k] * L2[j][k];
			}
			L2[i][j] = (A[i][j] - s) / L2[j][j];
		}
	}

	return 1;
}

int LUsym(int n, double *b, double **L) {
	// solve linear equation Ax=b, in which A=LL'
	// L is lower triangle
	int i, j;
	// forward iteration
	// Ly = b
	// y is stored in b
	for (j = 0; j < n-1; j++) {
		b[j] /= L[j][j];
		for (i = j+1; i < n; i++) {
			b[i] -= L[i][j] * b[j];
		}
	}
	b[n-1] /= L[n-1][n-1];
	// backward iteration
	// Ux = y, U = L'
	for (j = n-1; j >= 1; j--) {
		b[j] /= L[j][j];
		for (i = j-1; i >= 0; i--) {
			b[i] -= L[j][i] * b[j];
		}
	}
	b[0] /= L[0][0];
	// result x is stored in b
	return 1;
}

int LUsolve(int n, double *b, double **L, double **U) {
	// solve linear equation Ax=b, in which A=LU
	// L is lower triangle
	// U is upper triangle
	int i, j;
	// forward iteration
	// Ly = b
	// y is stored in b
	for (j = 0; j < n-1; j++) {
		b[j] /= L[j][j];
		for (i = j+1; i < n; i++) {
			b[i] -= L[i][j] * b[j];
		}
	}
	b[n-1] /= L[n-1][n-1];
	// backward iteration
	// Ux = y
	for (j = n-1; j >= 1; j--) {
		b[j] /= U[j][j];
		for (i = j-1; i >= 0; i--) {
			b[i] -= U[i][j] * b[j];
		}
	}
	b[0] /= U[0][0];
	// result x is stored in b
	return 1;
}

int Linv(int n, double *b, double **L) {
	// solve linear equation Lx=b
	// L is lower triangle
	int i, j;
	for (j = 0; j < n-1; j++) {
		b[j] /= L[j][j];
		for (i = j+1; i < n; i++) {
			b[i] -= L[i][j] * b[j];
		}
	}
	b[n-1] /= L[n-1][n-1];
	// result x is stored in b
	return 1;
}

double mgnlike(int nSample, double *y, double *a, double **L) {
	// compute marginal likelihood of Gaussian Processes Regression
	double m, t1, t2, t3;
	int i;
	m = 0; t1 = 0; t2 = 0; t3 = 0;

	for (i = 0; i < nSample; i++) {
		t1 -= y[i] * a[i];
	} t1 *= 0.5;

	for (i = 0; i < nSample; i++) {
		t2 -= log(L[i][i]);
	}

	t3 = -0.5*nSample*log(2*PI);

	m = t1 + t2 + t3;

	return m;
}
