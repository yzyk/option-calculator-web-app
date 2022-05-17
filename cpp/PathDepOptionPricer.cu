/* ky2072 KeYang 2022-05-17*/
#include <jni.h>
#include "PathDepOptionPricer.cuh"

using namespace std;
using namespace fre;

JNIEXPORT jdouble JNICALL Java_com_ky2072_fre_entity_PathDepOptionPricer_priceByGPU
(JNIEnv* env, jobject obj, jint nSim, jdouble T, jint m, jdouble S0, jdouble sigma, jdouble r, jint callput, jdouble K, jdouble B, jint option_i) {
	return price_one_option_function(nSim, T, m, S0, sigma, r, callput, K, B, option_i);
}