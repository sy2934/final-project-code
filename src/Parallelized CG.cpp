#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <ctime>


using namespace std;

const double NEARZERO = 1.0e-10;       // interpretation of "zero"

using vec    = vector<double>;         // vector
using matrix = vector<vec>;            // matrix (=collection of (row) vectors)


//Relevant functions used in this algorithm
void print( string title, const vec &V );
void print( string title, const matrix &A );
vec matrixTimesVector( const matrix &A, const vec &V );
vec vectorCombination( double a, const vec &U, double b, const vec &V );
double innerProduct( const vec &U, const vec &V );
double vectorNorm( const vec &V );
vec conjugateGradientSolver( const matrix &A, const vec &B );


//Make a symmetric positive-definite matrix
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

int main()
{

   int num_threads;

   //Specify the number of threads being used in parallelization
   #ifdef _OPENMP
      num_threads = 12;
      omp_set_num_threads(num_threads);
      cout << "Using OpenMP with " << num_threads << " threads.\n";
   #endif

   //Set up the starting time
   int start_s = clock();

   //The input matrix A has to be a symmetric positive definite matrix
   //please use the function "gemm_ATA" to generate one for different size n
   vec X = conjugateGradientSolver( A, B );


   cout << "Solves AX = B\n";
   print( "\nA:", A );
   print( "\nB:", B );
   print( "\nX:", X );
   print( "\nCheck AX:", matrixTimesVector( A, X ) );

   //Set up the ending time
   int stop_s = clock();
   
   //Compute the execution time of the code
   cout << (stop_s - start_s)/double(CLOCKS_PER_SEC);


   return 0;

}




// Function used to print out a vector
void print( string title, const vec &V )
{
   cout << title << '\n';

   int n = V.size();
   for ( int i = 0; i < n; i++ )
   {
      double x = V[i];   if ( abs( x ) < NEARZERO ) x = 0.0;
      cout << x << '\t';
   }
   cout << '\n';
}



// Function used to print out a matirx
void print( string title, const matrix &A )
{
   cout << title << '\n';

   int m = A.size(), n = A[0].size();
   for ( int i = 0; i < m; i++ )
   {
      for ( int j = 0; j < n; j++ )
      {
         double x = A[i][j];   if ( abs( x ) < NEARZERO ) x = 0.0;
         cout << x << '\t';
      }
      cout << '\n';
   }
}



// Parallelized function which does the matrix-vector product
vec matrixTimesVector( const matrix &A, const vec &V )
{
   int n = A.size();
   vec C( n );

   #pragma omp for
      for ( int i = 0; i < n; i++ )
        C[i] = innerProduct( A[i], V );

   #pragma omp barrier

   return C;
}



// Parallelized function which does the vector-vector addition
vec vectorCombination( double a, const vec &U, double b, const vec &V )
{
   int n = U.size();
   vec W( n );

   #pragma omp for
   for ( int j = 0; j < n; j++ )
      W[j] = a * U[j] + b * V[j];

   #pragma omp barrier

   return W;
}




// Parallelized function which does the vector-vector dot product
double innerProduct( const vec &U, const vec &V )
{
   double product = 0;
   int n = U.size();

   #pragma omp parallel
   {

     #pragma omp for reduction(+: product)
       for (int i = 0; i < n; i++ )
         product = product + U[i]*V[i];

   }

   #pragma omp barrier

   return product;
}



// Function which produces the L^2 norm of a vector
double vectorNorm( const vec &V )
{
   return sqrt( innerProduct( V, V ) );
}



// Conjugate gradient iteration algorithm
vec conjugateGradientSolver( const matrix &A, const vec &B )
{
   double TOLERANCE = 1.0e-10;
   int Iter_Max = 1000;
   int n = A.size();
   vec X( n, 0.0 );

   vec R = B;
   vec P = R;
   int k = 0;

   while ( k < Iter_Max )
   {
      // Store previous residual
      vec Rold = R;

      vec AP = matrixTimesVector( A, P );

      double alpha = innerProduct( R, R ) / max( innerProduct( P, AP ), NEARZERO );

      // Next estimate of the solution
      X = vectorCombination( 1.0, X, alpha, P );
      // update the residual
      R = vectorCombination( 1.0, R, -alpha, AP );

      // Convergence test
      if ( vectorNorm( R ) < TOLERANCE )
           break;

      // update varibles
      double beta = innerProduct( R, R ) / max( innerProduct( Rold, Rold ), NEARZERO );
      P = vectorCombination( 1.0, R, beta, P );

      // Check for failure
      if (k >= Iter_Max)
         cout << " CG method fails to converge!\n";

      k++;
   }

   return X;
}
