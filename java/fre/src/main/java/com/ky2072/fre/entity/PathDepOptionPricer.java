// edited by ky2072 KeYang 2022-05-17
package com.ky2072.fre.entity;

import com.ky2072.fre.utils.FreUtils;

public class PathDepOptionPricer {
    /*
      This class links its native method with CUDA C/C++ codes in dll file
     */

    static {
        System.load(FreUtils.getFreContext() + "/src/main/resources/PathDepOptionPricer.dll");
    }

    public native double priceByGPU(int nSim, double T, int m, double S0, double sigma, double r,
                                    int callput, double K, double B, int option_i);

}
