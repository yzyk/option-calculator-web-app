package com.ky2072.fre;

import com.ky2072.fre.entity.PathDepOptionPricer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringRunner;

@RunWith(SpringRunner.class)
@SpringBootTest
@ContextConfiguration(classes = FreApplication.class)
public class PathDepOptionPricerTests {

    @Test
    public void testPathDepOptionPricer() {
        PathDepOptionPricer pathDepOptionPricer = new PathDepOptionPricer();
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
        double price = pathDepOptionPricer.priceByGPU(nSim, T, m, S0, sigma, r, callput, K, B, option_i);
        System.out.println(price);
    }
}
