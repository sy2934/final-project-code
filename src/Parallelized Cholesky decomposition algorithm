#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <ctime>
#include <iostream>

using namespace std;

extern "C" void cholesky_dll(double *A, double *L, int n);


//Function used to create a symmetric positive definite matrix
void gemm_ATA(double *A, double *C, int n) {
    for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			double sum = 0;
			for(int k=0; k<n; k++) {
				sum += A[i*n+k]*A[j*n+k];
			}
			C[i*n+j] = sum;
		}
	}
}


// Parallelized Cholesky decomposition algorithm
double *cholesky4(double *A, int n) {

   double *L = (double*)calloc(n * n, sizeof(double));
    if (L == NULL)
        exit(EXIT_FAILURE);

	for (int j = 0; j <n; j++) {

		double s = 0;
                for (int k = 0; k < j; k++) {
			s += L[j * n + k] * L[j * n + k];
		}
		L[j * n + j] = sqrt(A[j * n + j] - s);
		#pragma omp for
		for (int i = j+1; i <n; i++) {
                    double s = 0;
                    for (int k = 0; k < j; k++) {
                         s += L[i * n + k] * L[j * n + k];
	            }
                    L[i * n + j] = (1.0 / L[j * n + j] * (A[i * n + j] - s));
                }
	}
    return L;
}


//Function used to print out the matrix A
void show_matrix(double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%2.5f ", A[i * n + j]);
        printf("\n");
    }
}

int main()
{


   //Specify the number of threads being used in parallelization
   #ifdef _OPENMP
      num_threads = 12;
      omp_set_num_threads(num_threads);
      cout << "Using OpenMP with " << num_threads << " threads.\n";
   #endif

   //Set the size of the coefficient matrix A
   int n = 3000;

        // Create a symme
	double *m3 = (double*)malloc(sizeof(double)*n*n);
	for(int i=0; i<n; i++) {
		for(int j=i; j<n; j++) {
			double element = 1.0*rand()/RAND_MAX;
			m3[i*n+j] = element;
			m3[j*n+i] = element;

		}
	}
	double *m4 = (double*)malloc(sizeof(double)*n*n);
	gemm_ATA(m3, m4, n); //make a positive-definite matrix
	printf("\n");
	//show_matrix(m4,n);

        //set the starting time
        double start_s = clock();


        //Call the Cholesky decomposition
	double *c3 = cholesky4(m4, n);

        //set the ending time
        double stop_s = clock();

        //compute the execution time
        cout << (stop_s - start_s)/double(CLOCKS_PER_SEC);

    return 0;

}
