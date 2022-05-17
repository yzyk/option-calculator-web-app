// edited by ky2072 KeYang 2022-05-17
package com.ky2072.fre.controller;

import com.ky2072.fre.entity.PathDepOption;
import com.ky2072.fre.entity.Stock;
import com.ky2072.fre.service.OptionPricingService;
import com.ky2072.fre.utils.FileWriter;
import com.ky2072.fre.utils.FreConstants;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

@Controller
public class OptionPricingController implements FreConstants {
    /**
     * This implements controller for all service related to option pricing
     * The current implementation is about two separate web pages:
     * One page presents a calculator for user to interact with
     * Another page presents a data-pipeline for user to query
     */

    @Autowired
    private OptionPricingService optionPricingService;

    @Autowired
    private FileWriter fileWriter;

    // return page for the calculator
    @RequestMapping(path = "/path-dependent/calculator", method = {RequestMethod.GET, RequestMethod.POST})
    public String getPathDepCalculatorPage(Model model) {
        return "path-dependent-calculator";
    }

    // post to calculate the price of path dependent options
    @RequestMapping(path = "/path-dependent/calculator/exec", method = RequestMethod.POST)
    public String getPrice(int nSim, double T, int m, double S0, double sigma, double r, double K,
                           Model model) {

        long startTime = System.currentTimeMillis();
        Map<String, PathDepOption> res = optionPricingService.pricePathDepOptions(nSim, T, m, S0, sigma, r, K, 0);
        long endTime = System.currentTimeMillis();

        model.addAttribute("lookBackCallV", res.get("lookBackCall").getPrice());
        model.addAttribute("lookBackPutV", res.get("lookBackPut").getPrice());
        model.addAttribute("asianCallV", res.get("asianCall").getPrice());
        model.addAttribute("asianPutV", res.get("asianPut").getPrice());
        model.addAttribute("nSimV", nSim);
        model.addAttribute("TV", T);
        model.addAttribute("mV", m);
        model.addAttribute("S0V", S0);
        model.addAttribute("sigmaV", sigma);
        model.addAttribute("rV", r);
        model.addAttribute("KV", K);
        model.addAttribute("exeTimeV", String.format("%.2f", (endTime - startTime) / 1000.0));

        return "forward:/path-dependent/calculator";
    }

    // return page for data pipeline
    @RequestMapping(path = "/path-dependent/time-series-data", method = {RequestMethod.GET, RequestMethod.POST})
    public String getTimeSeriesPage() {
        return "path-dependent-time-series";
    }

    // post to return a table as the user's query
    @RequestMapping(path = "/path-dependent/time-series-data/exec", method = RequestMethod.POST)
    public String getTimeSeries(@DateTimeFormat(pattern = "yyyy-MM-dd") LocalDate startDate,
                                @DateTimeFormat(pattern = "yyyy-MM-dd") LocalDate endDate,
                                int nSim, String ticker, double sigma, double r, String optionTypeS, String callPutS,
                                Model model) {

        long startTime = System.currentTimeMillis();

        int optionType = 0;
        int callPut = 0;
        switch (optionTypeS) {
            case "barrier":
                optionType = TYPE_OPTION_BARRIER;
                break;
            case "lookBack":
                optionType = TYPE_OPTION_LOOKBACK;
                break;
            case "asian":
                optionType = TYPE_OPTION_ASIAN;
                break;
        }
        switch (callPutS) {
            case "call":
                callPut = TYPE_CALL;
                break;
            case "put":
                callPut = TYPE_PUT;
                break;
        }

        ArrayList<Stock> stocks = optionPricingService.findStockOptionsByTickerTime(
                startDate.toString(), endDate.toString(), ticker, optionType, callPut, nSim, sigma, r
        );
        ArrayList<Map<String, String>> optionVOList = new ArrayList<>();
        for (Stock stock : stocks) {
            for (PathDepOption option : stock.getPathDepOptions()) {
                Map<String, String> optionVO = new HashMap<>();
                optionVO.put("date", stock.getDate().toString());
                optionVO.put("ticker", stock.getTicker());
                optionVO.put("stockPrice", String.format("%.4f", stock.getPrice()));
                optionVO.put("T", String.format("%.1f", option.getT()));
                optionVO.put("K", String.format("%.2f", option.getK()));
                optionVO.put("optionType", optionTypeS);
                optionVO.put("callPut", callPutS);
                optionVO.put("optionPrice", String.format("%.4f", option.getPrice()));
                optionVOList.add(optionVO);
            }
        }
        long endTime = System.currentTimeMillis();

        model.addAttribute("optionVOList", optionVOList);
        model.addAttribute("startDateV", startDate);
        model.addAttribute("endDateV", endDate);
        model.addAttribute("tickerV", ticker);
        model.addAttribute("nSimV", nSim);
        model.addAttribute("sigmaV", sigma);
        model.addAttribute("rV", r);
        model.addAttribute("optionTypeS", optionTypeS);
        model.addAttribute("callPutS", callPutS);
        model.addAttribute("exeTimeV", String.format("%.2f", (endTime - startTime) / 1000.0));

        fileWriter.writeToCSV(optionVOList);

        return "forward:/path-dependent/time-series-data";
    }



}
