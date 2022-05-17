package com.ky2072.fre;

import com.ky2072.fre.dao.StockMapper;
import com.ky2072.fre.entity.Stock;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringRunner;

import java.util.ArrayList;

@RunWith(SpringRunner.class)
@SpringBootTest
@ContextConfiguration(classes = FreApplication.class)
public class StockMapperTests {

    @Autowired
    private StockMapper stockMapper;

    @Test
    public void testSelectStockByTickerTime() {
        ArrayList<Stock> stocks = stockMapper.selectStockByTickerTime("2020-01-01", "2020-01-07", "AAPL");
        for (Stock stock : stocks) {
            System.out.println(stock);
        }
    }

}
