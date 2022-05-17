/* ky2072 KeYang 2022-05-17*/
#include "kernels.cuh"

using namespace std;
namespace fre {
    double price_one_option_function(int nSim, double T, int m, double S0, double sigma, double r, int callput, double K, double B, int option_i);
}