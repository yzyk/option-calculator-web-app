// edited by ky2072 KeYang 2022-05-17
package com.ky2072.fre.service;

import com.ky2072.fre.dao.StockMapper;
import com.ky2072.fre.entity.PathDepOption;
import com.ky2072.fre.entity.PathDepOptionPricer;
import com.ky2072.fre.entity.Stock;
import com.ky2072.fre.utils.FreConstants;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

@Service
public class OptionPricingService implements FreConstants {
    /**
     * This class is designed to implement all services for option pricing
     */

    @Autowired
    private StockMapper stockMapper;

    // This method is for test use only
    // This method is to price one path-dependent option
    public double pricePathDepOption(int nSim, double T, int m, double S0, double sigma, double r,
                              int callput, double K, double B, int option_i) {
        PathDepOptionPricer pathDepOptionPricer = new PathDepOptionPricer();
        return pathDepOptionPricer.priceByGPU(nSim, T, m, S0, sigma, r, callput, K, B, option_i);
    }

    // This method is to be called by controller methods for calculators
    // This method is to calculate prices for all path-dependent options (to be added) with multi threads
    // Each thread handle one option determined by option type and callput type
    public Map<String, PathDepOption> pricePathDepOptions(int nSim, double T, int m, double S0, double sigma, double r,
                                                 double K, double B) {
        Map<String, PathDepOption> res = new HashMap<>();

        PathDepOption lookBackCall = new PathDepOption(nSim, T, m, S0, sigma, r, TYPE_CALL, K, B, TYPE_OPTION_LOOKBACK);
        PathDepOption lookBackPut = new PathDepOption(nSim, T, m, S0, sigma, r, TYPE_PUT, K, B, TYPE_OPTION_LOOKBACK);
        PathDepOption asianCall = new PathDepOption(nSim, T, m, S0, sigma, r, TYPE_CALL, K, B, TYPE_OPTION_ASIAN);
        PathDepOption asianPut = new PathDepOption(nSim, T, m, S0, sigma, r, TYPE_PUT, K, B, TYPE_OPTION_ASIAN);

        Thread t1 = new Thread(lookBackCall);
        Thread t2 = new Thread(lookBackPut);
        Thread t3 = new Thread(asianCall);
        Thread t4 = new Thread(asianPut);

        t1.start();
        t2.start();
        t3.start();
        t4.start();

        try {
            t1.join();
            t2.join();
            t3.join();
            t4.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        res.put("lookBackCall", lookBackCall);
        res.put("lookBackPut", lookBackPut);
        res.put("asianCall", asianCall);
        res.put("asianPut", asianPut);

        return res;
    }

    // This method is to be called by controller methods for data-pipeline
    // This method first query a list of stocks based on user's parameters,
    // then for each stock, it generates multiple parameter sets for K and T,
    // use multi threads to calculate option prices for different parameter set for this stock,
    // where each thread calculate one parameter set
    public ArrayList<Stock> findStockOptionsByTickerTime(String startDate, String endDate, String ticker,
                                                         int optionType, int callPut,
                                                         int nSim, double sigma, double r) {

        ArrayList<Stock> stocks = stockMapper.selectStockByTickerTime(startDate, endDate, ticker);

        for (Stock stock : stocks) {
            double S0 = stock.getPrice();
            double[] tChoices = new double[]{0.5, 1.0, 2.0, 3.0};
            double[] kChoices = new double[]{Math.floor(Math.floor(S0) * 0.5 / 10) * 10,
                    Math.floor(S0),
                    Math.floor(Math.floor(S0) * 1.5 / 10) * 10};
            ExecutorService executorService = Executors.newFixedThreadPool(tChoices.length * kChoices.length);

            for (double T : tChoices) {
                for (double K : kChoices) {
                    int m = (int) (T * 30);
                    PathDepOption temp = new PathDepOption(nSim, T, m, S0, sigma, r, callPut, K, 0, optionType);
                    stock.setPathDepOptions(temp);
                    executorService.submit(temp);
                }
            }
            executorService.shutdown();
            try {
                executorService.awaitTermination(10, TimeUnit.MINUTES);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        return stocks;

    }
}
