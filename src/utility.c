/* ============== Utility.c ======================
 Some utility functions used in this program
 ================================================= */
#include <stdlib.h>
#include <math.h>
#include "constant.h"
#include "utility.h"

int PrintError(char* error_text) {
	fprintf(stderr, "%s\n", error_text);
	fprintf(stderr, "...now exiting to system...\n");
	exit(1);
	return 1;
}

FILE* Fopen(char* fname, char* mode) {
	FILE* fp;
	char str[MAX_LINE_LEN];

	if ((fp = (fopen(fname, mode))) == NULL) {
		sprintf(str, "%s %s\n", "Can not open file:", fname);
		PrintError(str);
	}
	return fp;
}

int *IntVector(int n) {
	int *v;

	v = (int *) calloc((size_t) n, (size_t) sizeof(int));
	if (!v)
		PrintError("allocation failure in IntVector()");
	return v;
}

double *DoubleVector(int n) {
	double *v;

	v = (double *) calloc((size_t) n, (size_t) sizeof(double));
	if (!v)
		PrintError("allocation failure in DoubleVector()");
	return v;
}

int **IntMatrix(int m, int n) {
	register int i;
	int **x;

	x = (int **) calloc((size_t) m, (size_t) sizeof(int *));
	if (!x)
		PrintError("allocation failure 1 in IntMatrix()");
	for (i = 0; i < m; i++) {
		x[i] = (int *) calloc((size_t) n, (size_t) sizeof(int));
		if (!x[i])
			PrintError("allocation failure 2 in IntMatrix()");
	}
	return x;
}

char **CharMatrix(int m, int n) {
	register int i;
	char **x;

	x = (char **) calloc((size_t) m, (size_t) sizeof(char *));
	if (!x)
		PrintError("allocation failure 1 in CharMatrix()");
	for (i = 0; i < m; i++) {
		x[i] = (char *) calloc((size_t) n, (size_t) sizeof(char));
		if (!x[i])
			PrintError("allocation failure 2 in CharMatrix()");
	}
	return x;
}

double **DoubleMatrix(int m, int n) {
	register int i;
	double **x;

	x = (double **) calloc((size_t) m, (size_t) sizeof(double *));
	if (!x)
		PrintError("allocation failure 1 in DoubleMatrix()");
	for (i = 0; i < m; i++) {
		x[i] = (double *) calloc((size_t) n, (size_t) sizeof(double));
		if (!x[i])
			PrintError("allocation failure 2 in DoubleMatrix()");
	}
	return x;
}

double ***Double3Dim(int m, int n, int k) {
	register int i, j;
	double ***x;

	x = (double ***) calloc((size_t) m, (size_t) sizeof(double **));
	if (!x)
		PrintError("allocation failure 1 in Double3Dim()");
	for (i = 0; i < m; i++) {
		x[i] = (double **) calloc((size_t) n, (size_t) sizeof(double *));
		if (!x[i])
			PrintError("allocation failure 2 in Double3Dim()");
		for (j = 0; j < n; j++) {
			x[i][j] = (double *) calloc((size_t) k, (size_t) sizeof(double));
			if (!x[i][j])
				PrintError("allocation failure 3 in Double3Dim()");
		}
	}
	return x;
}

int FreeDoubleVector(double *v) {
	free((double *) v);
	return 0;
}

int FreeIntVector(int *v) {
	free((int *) v);
	return 0;
}

int FreeIntMatrix(int **x, int m) {
	register int i;

	for (i = 0; i < m; i++)
		free((int *) x[i]);
	free((int *) x);
	return 0;
}

int FreeCharMatrix(char **x, int m) {
	register int i;

	for (i = 0; i < m; i++)
		free((char *) x[i]);
	free((char *) x);
	return 0;
}

int FreeDoubleMatrix(double **x, int m) {
	register int i;

	for (i = 0; i < m; i++)
		free((double *) x[i]);
	free((double *) x);
	return 0;
}

int FreeDouble3Dim(double ***x, int m, int n) {
	register int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++)
			free((double *) x[i][j]);
		free((double *) x[i]);
	}
	free((double *) x);
	return 0;
}

double MaxVector(int n, double *v) {

	int i;
	double max1;

	max1 = NINF;
	for (i = 0; i < n; i++)
		if (v[i] > max1)
			max1 = v[i];

	return max1;
}

double MinVector(int n, double *v) {

	int i;
	double min1;

	min1 = INF;
	for (i = 0; i < n; i++)
		if (v[i] < min1)
			min1 = v[i];

	return min1;
}

double Round(double x) {
	double y;
	y = ceil(x);
	if ((ceil(x) - x) <= 0.5)
		return (y);
	return (floor(x));
}

double Probks(double alam) {
	register int j;
	double a2, fac = 2.0, sum = 0.0, term, termbf = 0.0;

	a2 = -2.0 * alam * alam;
	for (j = 1; j <= 100; j++) {
		term = fac * exp(a2 * j * j);
		sum += term;
		if (fabs(term) <= EPS1 * termbf || fabs(term) <= EPS2 * sum)
			return sum;
		fac = -fac;
		termbf = fabs(term);
	}
	return 1.0;
}

double Mean(int n, double *x) {
	double ave = 0; int i;
	for (i = 0; i < n; i++) ave += x[i];
	ave /= (double)n;
	return ave;
}

double Std(int n, double *x) {
	double std = 0, ave;
	int i;
	ave = Mean(n, x);
	for (i = 0; i < n; i++) {
		std += pow(x[i] - ave, 2.0);
	}
	std = sqrt(std / (n - 1));
	return std;
}

double Var(int n, double *x) {
	double var = 0, ave;
	int i;
	ave = Mean(n, x);
	for (i = 0; i < n; i++) {
		var += pow(x[i] - ave, 2.0);
	}
	var = var / n;
	return var;
}

double CorrCoef(int nx, double *x, int ny, double *y) {
	double sx, sy, xy, x2, y2, r;
	int n, i;
    if (nx != ny) {
        printf("Error! length of x and y must be the same!\n");
        return 0;
    }
    n = nx;
    xy = 0; x2 = 0; y2 = 0; sx = 0; sy = 0;
	for (i = 0; i < n; i++) {
        xy += x[i] * y[i];
        x2 += x[i] * x[i];
        y2 += y[i] * y[i];
        sx += x[i];
        sy += y[i];
	}
	r = (n * xy - sx * sy) / (sqrt((n * x2 - sx * sx) * (n * y2 - sy * sy)));
	return r;
}

void Mean2D(int m, int n, double **v, double *ave) {

	int i, j;

	for (i = 0; i < n; i++) {
		ave[i] = 0.;
		for (j = 0; j < m; j++)
			ave[i] = ave[i] + v[j][i];
		ave[i] = ave[i] / m;
	}

}

void Var2D(int m, int n, double **v, double *var) {

	int i, j;
	double *ave;

	ave = DoubleVector(n);
	Mean2D(m, n, v, ave);
	for (i = 0; i < n; i++) {
		var[i] = 0.;
		for (j = 0; j < m; j++)
			var[i] = var[i] + 1.0 * pow(v[j][i] - ave[i], 2.0) / (m - 1);
	}

	FreeDoubleVector(ave);
}

double **transpose(int nrow, int ncol, double **A) {
	int i, j;
	double **AT;
	AT = DoubleMatrix(ncol, nrow);
	for (i = 0; i < nrow; i++) {
		for (j = 0; j < ncol; j++) {
			AT[j][i] = A[i][j];
		}
	}
	return AT;
}