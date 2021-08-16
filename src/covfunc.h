/*
 * covfunc.h
 *
 *  Created on: Sep 18 2018
 *      Author: gongwei, BNU
 */
 
#ifndef COVFUNC_H_
#define COVFUNC_H_

int CalcCovMatrix(int nInput, int nSample, int CovIdx, int nhyp, double *hyp, double **X, double **K);
int CalcCovVector(int nInput, int nSample, int CovIdx, int nhyp, double *hyp, double **X, double *x, double *k);
double CalcCovFunc(int CovIdx, int nhyp, double *hyp, int nInput, double *x1, double *x2);
double CovMatern3(int nhyp, double *hyp, int nInput, double *x1, double *x2);
double CovMatern5(int nhyp, double *hyp, int nInput, double *x1, double *x2);
double CovSE(int nhyp, double *hyp, int nInput, double *x1, double *x2);
double CovSEnoisefree(int nhyp, double *hyp, int nInput, double *x1, double *x2);
double CovNN(int nhyp, double *hyp, int nInput, double *x1, double *x2);

#endif /* COVFUNC_H_ */
