// edited by ky2072 KeYang 2022-05-17
package com.ky2072.fre.entity;

public class PathDepOption implements Runnable {
    /**
     * This is unique class to describe path-dependent options
     * This version implements Runnable interface and override run() method to call priceByGPU() method,
     consider improvement to separate this functionality to another class by using this another class's run() to
     call the PathDepOption object
     * priceByGPU() will call the PathDepOptionPricer object to interact with GPU
     */

    private int nSim;
    private double T;
    private int m;
    private double S0;
    private double sigma;
    private double r;
    private int callput;
    private double K;
    private double B;
    private int option_i;
    private double price;

    public void priceByGPU() {
        PathDepOptionPricer pathDepOptionPricer = new PathDepOptionPricer();
        price = pathDepOptionPricer.priceByGPU(nSim, T, m, S0, sigma, r, callput, K, B, option_i);
    }

    public double getPrice() {
        return price;
    }

    @Override
    public void run() {
        priceByGPU();
    }

    public PathDepOption(int nSim, double t, int m, double s0, double sigma, double r, int callput, double k, double b, int option_i) {
        this.nSim = nSim;
        T = t;
        this.m = m;
        S0 = s0;
        this.sigma = sigma;
        this.r = r;
        this.callput = callput;
        K = k;
        B = b;
        this.option_i = option_i;
    }

    @Override
    public String toString() {
        return "PathDepOption{" +
                "nSim=" + nSim +
                ", T=" + T +
                ", m=" + m +
                ", S0=" + S0 +
                ", sigma=" + sigma +
                ", r=" + r +
                ", callput=" + callput +
                ", K=" + K +
                ", B=" + B +
                ", option_i=" + option_i +
                ", price=" + price +
                '}';
    }

    public int getnSim() {
        return nSim;
    }

    public void setnSim(int nSim) {
        this.nSim = nSim;
    }

    public double getT() {
        return T;
    }

    public void setT(double t) {
        T = t;
    }

    public int getM() {
        return m;
    }

    public void setM(int m) {
        this.m = m;
    }

    public double getS0() {
        return S0;
    }

    public void setS0(double s0) {
        S0 = s0;
    }

    public double getSigma() {
        return sigma;
    }

    public void setSigma(double sigma) {
        this.sigma = sigma;
    }

    public double getR() {
        return r;
    }

    public void setR(double r) {
        this.r = r;
    }

    public int getCallput() {
        return callput;
    }

    public void setCallput(int callput) {
        this.callput = callput;
    }

    public double getK() {
        return K;
    }

    public void setK(double k) {
        K = k;
    }

    public double getB() {
        return B;
    }

    public void setB(double b) {
        B = b;
    }

    public int getOption_i() {
        return option_i;
    }

    public void setOption_i(int option_i) {
        this.option_i = option_i;
    }
}
