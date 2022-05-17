/* ky2072 KeYang 2022-05-17 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//	The following code is a copy of the 1st examplefrom 
//	section 3.6 Device API Examples, cuRand, CUDA Toolkit v11.6.1
//
//	These examples build executables that can be run from the 
//	Windows command prompt (or from a terminal on Linux)
//
//	use argument -m for MRG32k3a, or -p for the Philox RNG
//	the default (no argument) uses the XORWOW RNG


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cuda.h>
#include <curand_kernel.h>

using namespace std;

namespace fre {
    enum option_type { barrier = 1, lookback = 2, asian = 3 };

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
        printf("Error at %s:%d\n",__FILE__,__LINE__); \
        printf("CUDA Error: %s\n", cudaGetErrorString(x));\
        return EXIT_FAILURE;}} while(0)

    __device__ double barrier_payoff(double S0, double minSt, double maxSt, double ST, int callput, double K, double B);

    __device__ double lookback_payoff(double minSt, double maxSt, int callput, double K);

    __device__ double asian_payoff(double avgSt, int callput, double K);

    // callput = 1 : call; callput = 2 : put
    __device__ double payoff(double S0, double minSt, double maxSt, double avgSt,
        double ST, int callput, option_type option, double K, double B);

    __global__ void setup_kernel(curandStateMRG32k3a* state, int m, int nSim);

    __global__ void one_simulation_kernel(curandStateMRG32k3a* state, int nSim,
        double T, int m, double S0, double sigma, double r, int callput, option_type option, double K, double B,
        double* result, double* Ss);
}