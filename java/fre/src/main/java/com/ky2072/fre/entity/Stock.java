// edited by ky2072 KeYang 2022-05-17
package com.ky2072.fre.entity;

import java.time.LocalDate;
import java.util.ArrayList;

public class Stock {
    /**
     * This class is unique class to describe a stock at a given date
     * One stock at a given date can have multiple options, so we have a list of PathDepOptions here
     */

    private String ticker;
    private int securityId;
    private LocalDate date;
    private double price;

    ArrayList<PathDepOption> pathDepOptions;

    public Stock(String ticker, int securityId, LocalDate date, double price) {
        this.ticker = ticker;
        this.securityId = securityId;
        this.date = date;
        this.price = price;
        this.pathDepOptions = new ArrayList<>();
    }

    @Override
    public String toString() {
        return "Stock{" +
                "ticker='" + ticker + '\'' +
                ", securityId=" + securityId +
                ", date=" + date +
                ", price=" + price +
                '}';
    }

    public String getTicker() {
        return ticker;
    }

    public void setTicker(String ticker) {
        this.ticker = ticker;
    }

    public int getSecurityId() {
        return securityId;
    }

    public void setSecurityId(int securityId) {
        this.securityId = securityId;
    }

    public LocalDate getDate() {
        return date;
    }

    public void setDate(LocalDate date) {
        this.date = date;
    }

    public double getPrice() {
        return price;
    }

    public void setPrice(double price) {
        this.price = price;
    }

    public void setPathDepOptions(PathDepOption option) {
        this.pathDepOptions.add(option);
    }

    public ArrayList<PathDepOption> getPathDepOptions() {
        return (ArrayList<PathDepOption>) this.pathDepOptions.clone();
    }
}
