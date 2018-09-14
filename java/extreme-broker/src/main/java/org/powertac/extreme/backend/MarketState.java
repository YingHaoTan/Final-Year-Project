package org.powertac.extreme.backend;

import java.nio.ByteBuffer;

import org.powertac.common.ClearedTrade;
import org.powertac.common.MarketPosition;
import org.powertac.common.Orderbook;
import org.powertac.common.OrderbookOrder;
import org.powertac.common.Timeslot;
import org.powertac.common.msg.MarketBootstrapData;

public class MarketState implements ISerialize {
	private float[] ctprice;
	private float[] ctqty;
	private float[] ubqty;
	private float[] avgubprice;
	private float[] uaqty;
	private float[] avguaprice;
	private float[] power;
	
	public MarketState() {
		this.ctprice = new float[24];
		this.ctqty = new float[24];
		this.ubqty = new float[24];
		this.avgubprice = new float[24];
		this.uaqty = new float[24];
		this.avguaprice = new float[24];
		this.power = new float[24];
	}
	
	public void bootstrap(MarketBootstrapData data, Timeslot currentTimeslot) {
		int starthour = currentTimeslot.slotInDay() + 24;
		
		double[] prices = data.getMarketPrice();
		double[] quantities = data.getMwh();
		for(int i = 0; i < ctprice.length; i++) {
			int index = prices.length - i - 1;
			int hour = starthour - i;
			
			this.ctprice[hour % 24] = (float) prices[index];
			this.ctqty[hour % 24] = (float) quantities[index];
		}
	}
	
	public void setCTPrice(ClearedTrade trade) {
		int slot = trade.getTimeslot().slotInDay();
		this.ctprice[slot] = Math.abs((float) trade.getExecutionPrice());
		this.ctqty[slot] = Math.abs((float) trade.getExecutionMWh());
	}
	
	public void setUnclearedOrders(Orderbook orderbook) {
		int slot = orderbook.getTimeslot().slotInDay();
		
		this.avgubprice[slot] = 0.0f;
		for(OrderbookOrder order: orderbook.getBids()) {
			this.ubqty[slot] += order.getMWh();
			if(order.getLimitPrice() != null)
				this.avgubprice[slot] += order.getLimitPrice().floatValue();
		}
		
		this.avguaprice[slot] = 0.0f;
		for(OrderbookOrder order: orderbook.getBids()) {
			this.uaqty[slot] += order.getMWh();
			if(order.getLimitPrice() != null)
				this.avguaprice[slot] += order.getLimitPrice().floatValue();
		}
		
		if(this.ubqty[slot] > 0)
			this.avgubprice[slot] /= this.ubqty[slot];
		if(this.uaqty[slot] > 0)
			this.avguaprice[slot] /= this.uaqty[slot];
	}
	
	public void setPowerBalance(MarketPosition posn) {
		int slot = posn.getTimeslot().slotInDay();
		this.power[slot] = (float) posn.getOverallBalance();
	}

	@Override
	public int getSizeInBytes() {
		return 672;
	}

	@Override
	public void serialize(ByteBuffer buffer) {
		for(float f: this.ctprice)
			buffer.putFloat(f);
		for(float f: this.ctqty)
			buffer.putFloat(f);
		for(float f: this.ubqty)
			buffer.putFloat(f);
		for(float f: this.avgubprice)
			buffer.putFloat(f);
		for(float f: this.uaqty)
			buffer.putFloat(f);
		for(float f: this.avguaprice)
			buffer.putFloat(f);
		for(float f: this.power)
			buffer.putFloat(f);
	}

}
