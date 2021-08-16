/*
 * gpr.h
 * Gaussian Processes Regression
 *
 *  Created on: Sep 18 2018
 *      Author: gongwei, BNU
 */

#ifndef GPR_H_
#define GPR_H_
 
double GPtrain(int CovIdx, int nhyp, double *hyp, double noise, int nInput, int nSample, 
	double **X, double *y, double **K, double **L, double *a);
double GPtrain_r(int CovIdx, int nhyp, double *hyp, double noise, 
	int nInput, int nSample1, int nSample2, double **X1, double **X2, double **XX,
	double *y1, double *y2, double *yy, double **K, double **L, double *a,
	double **KK, double **LL, double *aa);
int GPpredict(int CovIdx, int nhyp, double *hyp, 
	int nInput, int nSample1, int nSample2, double **X1, double **X2, 
	double *y, double *pv, double **L, double *a);
int chol(int n, double **A, double **L);
int chol_r(int n1, int n2, double **A, double **L1, double **L2);
int LUsym(int n, double *b, double **L);
int LUsolve(int n, double *b, double **L, double **U);
int Linv(int n, double *b, double **L);
double mgnlike(int nSample, double *y, double *a, double **L);

#endif /* GPR_H_ */
