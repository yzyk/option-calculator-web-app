/* ky2072 KeYang 2022-05-17*/
#include "price_one_option.cuh"

using namespace std;
namespace fre {
    double price_one_option_function(int nSim, double T, int m, double S0, double sigma, double r, int callput, double K, double B, int option_i)
    {
        curandStateMRG32k3a* devMRGStates;
        double* devResults, * hostResults;
        double* devSs;

        option_type option;

        option = static_cast<option_type>(option_i);

        /* Allocate space for results on host */
        hostResults = (double*)calloc(nSim, sizeof(double));

        /* Allocate space for results on device */
        CUDA_CALL(cudaMalloc((void**)&devResults, nSim *
            sizeof(double)));

        CUDA_CALL(cudaMalloc((void**)&devSs, m * nSim *
            sizeof(double)));

        CUDA_CALL(cudaMalloc((void**)&devMRGStates, nSim *
            sizeof(curandStateMRG32k3a)));

        /* Set results to 0 */
        CUDA_CALL(cudaMemset(devResults, 0.0, nSim *
            sizeof(double)));

        CUDA_CALL(cudaMemset(devSs, 0.0, m * nSim *
            sizeof(double)));

        setup_kernel << <1 + nSim / 128, 128 >> > (devMRGStates, m, nSim);

        /* Generate and use normal pseudo-random  */

        one_simulation_kernel << <1 + nSim / 128, 128 >> > (devMRGStates, nSim,
            T, m, S0, sigma, r, callput, option, K, B,
            devResults, devSs);

        CUDA_CALL(cudaGetLastError());

        CUDA_CALL(cudaDeviceSynchronize());

        /* Copy device memory to host */
        CUDA_CALL(cudaMemcpy(hostResults, devResults, nSim *
            sizeof(double), cudaMemcpyDeviceToHost));

        /* Show result */
        double H = 0.0;
        double Hsq = 0.0;
        double price;
        for (int i = 0; i < nSim; i++) {
            H = (i * H + hostResults[i]) / (i + 1.0);
            Hsq = (i * Hsq + hostResults[i] * hostResults[i]) / (i + 1.0);
        }
        price = exp(-r * T) * H;

        /* Cleanup */
        CUDA_CALL(cudaFree(devMRGStates));
        CUDA_CALL(cudaFree(devResults));
        CUDA_CALL(cudaFree(devSs));
        free(hostResults);

        return price;
    }
}