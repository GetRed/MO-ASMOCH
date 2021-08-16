/* ============================================
prototypes of utility functions described in utility.c
==============================================*/

#ifndef _UTILITY_H_
#define _UTILITY_H_

int    PrintError(char* );
FILE   *Fopen(char*, char*);
int    *IntVector(int);
double *DoubleVector(int);
int    **IntMatrix(int, int);
double **DoubleMatrix(int, int);
char   **CharMatrix(int, int);
double ***Double3Dim(int, int, int);
int    FreeDoubleVector(double*);
int    FreeIntVector(int*);
int    FreeDoubleMatrix(double**, int);
int    FreeIntMatrix(int**, int);
int    FreeCharMatrix(char**, int);
int    FreeDouble3Dim(double ***, int, int);
double MinVector(int n, double *v);
double MaxVector(int n, double *v);
double Round(double);
double Probks(double);
double Mean(int n, double *x);
double Std(int n, double *x);
double Var(int n, double *x);
double CorrCoef(int nx, double *x, int ny, double *y);
void   Mean2D(int m, int n, double **v, double *ave);
void   Var2D(int m, int n, double **v, double *var);
double **transpose(int nrow, int ncol, double **A);
#endif
