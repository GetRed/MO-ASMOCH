/*
 * covfunc.h
 *
 *  Created on: Sep 18 2018
 *      Author: gongwei, BNU
 */

#ifndef CGP_H_
#define CGP_H_

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
        double *a, int na );

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
        double *aa, int naa );

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
        double *pv, int npv );

#endif /* CGP_H_ */
