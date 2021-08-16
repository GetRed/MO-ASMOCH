/* call GPtrain with python
 *  by Wei Gong, Sep 2018
 */

#include <stdlib.h>
#include "constant.h"
#include "utility.h"
#include "gpr.h"
#include "covfunc.h"

double callGPtrain( \
        /* input variables */
        double *X_in, int nrowX, int ncolX, \
        double *y, int ny, \
        int CovIdx, \
        double *hyp, int nhyp, \
        double noise, \
        /* output variables, should be pre-allocated */
        double *K_out, int nrowK, int ncolK, \
        double *L_out, int nrowL, int ncolL, \
        double *a, int na )
{
    int nSample, nInput;
    double m;
    int i, j, idx;
    double **X, **K, **L;
    
    /* check the size of arrays */
    nSample = nrowX;
    nInput = ncolX;
    if (nSample == 1) {
        PrintError("X must be a nSample*nInput double matrix.");
    }
    if (nSample != ny) {
        PrintError("X, y must be the same length.");
    }
    if ((nSample != nrowK) || (nSample != ncolK)) {
        PrintError("K must be a nSample*nSample double matrix.");
    }
    if ((nSample != nrowL) || (nSample != ncolL)) {
        PrintError("L must be a nSample*nSample double matrix.");
    }
    if (nSample != na) {
        PrintError("a must be a nSample double vector.");
    }

    /* prepare input matrix */
    X = DoubleMatrix(nSample, nInput);
    for (i = 0; i < nSample; i++) {
    	for (j = 0; j < nInput; j++) {
            idx = i*nInput + j;
    		X[i][j] = X_in[idx];
    	}
    }
    /* prepare output matrix */
    K = DoubleMatrix(nSample, nSample);
    L = DoubleMatrix(nSample, nSample);
    for (i = 0; i < nSample; i++) {
    	for (j = 0; j < nSample; j++) {
    		K[i][j] = 0;
			L[i][j] = 0;
    	}
    }

    /* run GPtrain to train a GP model */
    m = GPtrain(CovIdx, nhyp, hyp, noise, nInput, nSample, X, y, K, L, a);

    /* copy output matrix */
    for (i = 0; i < nSample; i++) {
    	for (j = 0; j < nSample; j++) {
            idx = i*nSample + j;
            K_out[idx] = K[i][j];
			L_out[idx] = L[i][j];
    	}
    }

    /* free allocated memory */
	FreeDoubleMatrix(X, nSample);
	FreeDoubleMatrix(K, nSample);
	FreeDoubleMatrix(L, nSample);

    return m;
}


double callGPtrain_r( \
        /* input variables */
        double *X1_in, int nrowX1, int ncolX1, \
        double *X2_in, int nrowX2, int ncolX2, \
        double *y1, int ny1, \
        double *y2, int ny2, \
        double *K_in, int nrowK, int ncolK, \
        double *L_in, int nrowL, int ncolL, \
        double *a, int na, \
        int CovIdx, \
        double *hyp, int nhyp, \
        double noise, \
        /* output variables, should be pre-allocated */
        double *XX_out, int nrowXX, int ncolXX, \
        double *yy, int nyy, \
        double *KK_out, int nrowKK, int ncolKK, \
        double *LL_out, int nrowLL, int ncolLL, \
        double *aa, int naa )
{
    int nSample, nSample1, nSample2, nInput;
    double m;
    int i, j, idx;
    double **X1, **X2, **K, **L, **XX, **KK, **LL;
    
    /* check the size of arrays */
    nSample1 = nrowX1;
    nSample2 = nrowX2;
    nSample = nSample1 + nSample2;
    nInput = ncolX1;
    if (nSample1 == 1) {
        PrintError("X1 must be a nSample1*nInput double matrix.");
    }
    if (nInput != ncolX2) {
        PrintError("X1 and X2 must have the same number of columns.");
    }
    if (nSample1 != ny1) {
        PrintError("X1, y1 must be the same length.");
    }
    if (nSample2 != ny2) {
        PrintError("X2, y2 must be the same length.");
    }
    if ((nSample1 != nrowK) || (nSample1 != ncolK)) {
        PrintError("K must be a nSample1*nSample1 double matrix.");
    }
    if ((nSample1 != nrowL) || (nSample1 != ncolL)) {
        PrintError("L must be a nSample1*nSample1 double matrix.");
    }
    if (nSample1 != na) {
        PrintError("a must be a nSample1 double vector.");
    }
    if (nSample != nrowXX) {
        PrintError("XX must be a nSample*nInput double matrix.");
    }
    if (nInput != ncolXX) {
        PrintError("XX must be a nSample*nInput double matrix.");
    }
    if (nSample != nyy) {
        PrintError("yy must be a nSample double vector.");
    }
    if ((nSample != nrowKK) || (nSample != ncolKK)) {
        PrintError("KK must be a nSample*nSample double matrix.");
    }
    if ((nSample != nrowLL) || (nSample != ncolLL)) {
        PrintError("LL must be a nSample*nSample double matrix.");
    }
    if (nSample != naa) {
        PrintError("aa must be a nSample1 double vector.");
    }
    
    /* prepare input matrix */
    X1 = DoubleMatrix(nSample1, nInput);
    for (i = 0; i < nSample1; i++) {
    	for (j = 0; j < nInput; j++) {
            idx = i*nInput + j;
    		X1[i][j] = X1_in[idx];
    	}
    }
    X2 = DoubleMatrix(nSample2, nInput);
    for (i = 0; i < nSample2; i++) {
    	for (j = 0; j < nInput; j++) {
            idx = i*nInput + j;
    		X2[i][j] = X2_in[idx];
    	}
    }
    K = DoubleMatrix(nSample1, nSample1);
    L = DoubleMatrix(nSample1, nSample1);
    for (i = 0; i < nSample1; i++) {
    	for (j = 0; j < nSample1; j++) {
            idx = i*nSample1 + j;
    		K[i][j] = K_in[idx];
    		L[i][j] = L_in[idx];
    	}
    }
    /* prepare output matrix */
    XX = DoubleMatrix(nSample, nInput);
    for (i = 0; i < nSample; i++) {
    	for (j = 0; j < nInput; j++) {
    		XX[i][j] = 0;
    	}
    }
    KK = DoubleMatrix(nSample, nSample);
    LL = DoubleMatrix(nSample, nSample);
    for (i = 0; i < nSample; i++) {
    	for (j = 0; j < nSample; j++) {
    		KK[i][j] = 0;
			LL[i][j] = 0;
    	}
    }

    /* run GPtrain_r to train a GP model */
    m = GPtrain_r(CovIdx, nhyp, hyp, noise, nInput, nSample1, nSample2, \
            X1, X2, XX, y1, y2, yy, K, L, a, KK, LL, aa);

    /* copy output matrix */
    for (i = 0; i < nSample; i++) {
    	for (j = 0; j < nInput; j++) {
            idx = i*nInput + j;
    		XX_out[idx] = XX[i][j];
    	}
    }
    for (i = 0; i < nSample; i++) {
    	for (j = 0; j < nSample; j++) {
            idx = i*nSample + j;
            KK_out[idx] = KK[i][j];
			LL_out[idx] = LL[i][j];
    	}
    }

    /* free allocated memory */
	FreeDoubleMatrix(X1, nSample1);
	FreeDoubleMatrix(X2, nSample2);
	FreeDoubleMatrix(K, nSample1);
	FreeDoubleMatrix(L, nSample1);
	FreeDoubleMatrix(XX, nSample);
	FreeDoubleMatrix(KK, nSample);
	FreeDoubleMatrix(LL, nSample);

    return m;
}

void callGPpredict( \
        /* input variables */
        double *X1_in, int nrowX1, int ncolX1, \
        double *X2_in, int nrowX2, int ncolX2, \
        int CovIdx, \
        double *hyp, int nhyp, \
        double *L_in, int nrowL, int ncolL, \
        double *a, int na, \
        /* output variables, should be pre-allocated */
        double *f, int nf, \
        double *pv, int npv )
{
    int nSample1, nSample2, nSample, nInput;
    int i, j, idx;
    double **X1, **X2, **L;
    
    /* check the size of arrays */
    nSample1 = nrowX1;
    nSample2 = nrowX2;
    nSample = nSample1 + nSample2;
    nInput = ncolX1;
    if (nSample1 == 1) {
        PrintError("X must be a nSample*nInput double matrix.");
    }
    if (nInput != ncolX2) {
        PrintError("X1 and X2 must have the same number of columns.");
    }
    if ((nSample1 != nrowL) || (nSample1 != ncolL)) {
        PrintError("L must be a nSample1*nSample1 double matrix.");
    }
    if (nSample1 != na) {
        PrintError("a must be a nSample1 double vector.");
    }
    if (nSample2 != nf) {
        PrintError("f must be a nSample2 double vector.");
    }
    if (nSample2 != npv) {
        PrintError("pv must be a nSample2 double vector.");
    }

    /* prepare input matrix */
    X1 = DoubleMatrix(nSample1, nInput);
    for (i = 0; i < nSample1; i++) {
    	for (j = 0; j < nInput; j++) {
            idx = i*nInput + j;
    		X1[i][j] = X1_in[idx];
    	}
    }
    X2 = DoubleMatrix(nSample2, nInput);
    for (i = 0; i < nSample2; i++) {
    	for (j = 0; j < nInput; j++) {
            idx = i*nInput + j;
    		X2[i][j] = X2_in[idx];
    	}
    }
    L = DoubleMatrix(nSample1, nSample1);
    for (i = 0; i < nSample1; i++) {
    	for (j = 0; j < nSample1; j++) {
            idx = i*nSample1 + j;
    		L[i][j] = L_in[idx];
    	}
    }
    
    /* run GPpredict with a trained GP model */
    GPpredict(CovIdx, nhyp, hyp, nInput, nSample1, nSample2, X1, X2, f, pv, L, a);

    /* free allocated memory */
	FreeDoubleMatrix(X1, nSample1);
	FreeDoubleMatrix(X2, nSample2);
	FreeDoubleMatrix(L, nSample1);

}
