%module cgp

%{
    #define SWIG_FILE_WITH_INIT
    #include "cgp.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double *X_in, int nrowX, int ncolX)}
%apply (double* IN_ARRAY1, int DIM1) {(double *y, int ny)}
%apply (double* IN_ARRAY1, int DIM1) {(double *hyp, int nhyp)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *K_out, int nrowK, int ncolK)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *L_out, int nrowL, int ncolL)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double *a, int na)}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double *X1_in, int nrowX1, int ncolX1)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double *X2_in, int nrowX2, int ncolX2)}
%apply (double* IN_ARRAY1, int DIM1) {(double *y1, int ny1)}
%apply (double* IN_ARRAY1, int DIM1) {(double *y2, int ny2)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double *K_in, int nrowK, int ncolK)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double *L_in, int nrowL, int ncolL)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *XX_out, int nrowXX, int ncolXX)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *KK_out, int nrowKK, int ncolKK)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *LK_out, int nrowLL, int ncolLL)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double *aa, int naa)}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double *f, int nf)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double *pv, int npv)}

%include "cgp.h"
