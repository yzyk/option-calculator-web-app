package com.ky2072.fre;

import com.ky2072.fre.entity.PathDepOption;
import com.ky2072.fre.entity.PathDepOptionPricer;
import com.ky2072.fre.entity.Stock;
import com.ky2072.fre.service.OptionPricingService;
import com.ky2072.fre.utils.FreConstants;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringRunner;

import java.util.ArrayList;
import java.util.Date;
import java.util.Map;

@RunWith(SpringRunner.class)
@SpringBootTest
@ContextConfiguration(classes = FreApplication.class)
public class OptionPricingServiceTests {

    @Autowired
    private OptionPricingService optionPricingService;

    @Test
    public void testPricePathDepOptions() {
        int nSim = 300000;
        double T = 1.0 / 12.0;
        int m = 30;
        double S0 = 100.0;
        double sigma = 0.2;
        double r = 0.03;
        int callput = 1;
        double K = 100.0;
        double B = 0;
        int option_i = 3;
        Map<String, PathDepOption> res = optionPricingService.pricePathDepOptions(nSim, T, m, S0, sigma, r, K, B);

        for (Map.Entry<String, PathDepOption> e : res.entrySet()) {
            System.out.println(e.getKey() + ": " + e.getValue().getPrice());
        }
    }

    @Test
    public void testFindStockOptionsByTickerTime() {
        ArrayList<Stock> stocks = optionPricingService.findStockOptionsByTickerTime("2020-01-01", "2020-01-07", "AAPL", FreConstants.TYPE_OPTION_ASIAN,
                FreConstants.TYPE_CALL, 30000, 0.2, 0.03);
        for (Stock stock : stocks) {
            for (PathDepOption option : stock.getPathDepOptions()) {
                System.out.println(stock.toString() + option.toString());
            }
        }
    }


}
