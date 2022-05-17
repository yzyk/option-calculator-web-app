/* ky2072 KeYang 2022-05-17 */
#include "kernels.cuh"

using namespace std;

namespace fre {

    __device__ double barrier_payoff(double S0, double minSt, double maxSt, double ST, int callput, double K, double B) {
        // call
        if (callput == 1) {
            // down and out
            if (B <= S0) {
                if (minSt > B) { return max(ST - K, 0.0); }
                else return 0.0;
            }
            // up and out
            else {
                if (maxSt < B) { return max(ST - K, 0.0); }
                else return 0.0;
            }
        }
        // put
        else if (callput == 2) {
            // down and out
            if (B <= S0) {
                if (minSt > B) { return max(K - ST, 0.0); }
                else return 0.0;
            }
            // up and out
            else {
                if (maxSt < B) { return max(K - ST, 0.0); }
                else return 0.0;
            }
        }
        else {
            return 0.0;
        }
    }

    __device__ double lookback_payoff(double minSt, double maxSt, int callput, double K) {

        if (callput == 1) {
            return max(maxSt - K, 0.0);
        }
        else if (callput == 2) {
            return max(K - minSt, 0.0);
        }
    }

    __device__ double asian_payoff(double avgSt, int callput, double K) {
        if (callput == 1) {
            return max(avgSt - K, 0.0);
        }
        else if (callput == 2) {
            return max(K - avgSt, 0.0);
        }
    }

    // callput = 1: call; callput = 2: put
    __device__ double payoff(double S0, double minSt, double maxSt, double avgSt,
        double ST, int callput, option_type option, double K, double B) {
        switch (option) {
        case barrier:
            return barrier_payoff(S0, minSt, maxSt, ST, callput, K, B);
            break;
        case lookback:
            return lookback_payoff(minSt, maxSt, callput, K);
            break;
        case asian:
            return asian_payoff(avgSt, callput, K);
            break;
        }
    }

    __global__ void setup_kernel(curandStateMRG32k3a* state, int m, int nSim)
    {
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        if (id >= nSim) return;
        /* Each thread gets same seed, a different sequence
        number, no offset */
        curand_init(0, id, 0, &state[id]);
    }

    __global__ void one_simulation_kernel(curandStateMRG32k3a* state, int nSim,
        double T, int m, double S0, double sigma, double r, int callput, option_type option, double K, double B,
        double* result, double* Ss)
    {
        int id = threadIdx.x + blockIdx.x * blockDim.x;


        if (id >= nSim) return;

        /* Copy state to local memory for efficiency */
        curandStateMRG32k3a localState = state[id];

        /* Generate pseudo-random normals */
        // use double array to hold a sample path
        double* S = Ss + m * id;
        double minSt = S0;
        double maxSt = S0;
        double avgSt = 0.0;
        double St = S0;
        double norm;
        for (int k = 0; k < m; k++) {
            norm = curand_normal_double(&localState);
            S[k] = St * exp((r - sigma * sigma * 0.5) * (T / m) + sigma * sqrt(T / m) * norm);
            minSt = min(S[k], minSt);
            maxSt = max(S[k], maxSt);
            avgSt = (k * avgSt + S[k]) / (k + 1.0);
            St = S[k];
        }
        double ST = St;

        double H = payoff(S0, minSt, maxSt, avgSt, ST, callput, option, K, B);

        state[id] = localState;
        /* Store results */
        result[id] = H;

    }
}