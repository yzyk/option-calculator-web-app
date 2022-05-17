// edited by ky2072 KeYang 2022-05-17
package com.ky2072.fre.dao;

import com.ky2072.fre.entity.Stock;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.ArrayList;

@Mapper
public interface StockMapper {
    /**
     * This interface provides a mapper between the Stock object and IVY database's query
     */

    @Select("select s.Ticker as ticker, s.SecurityID as security_id, sp.Date as date, sp.ClosePrice as price\n" +
            "from XFDATA.DBO.SECURITY_PRICE as sp\n" +
            "join XFDATA.DBO.SECURITY as s on sp.SecurityID = s.SecurityID\n" +
            "where s.Ticker=#{ticker} and sp.Date between #{startDate} and #{endDate}")
    public ArrayList<Stock> selectStockByTickerTime(String startDate, String endDate, String ticker);
}
