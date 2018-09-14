package org.powertac.extreme.backend;

import java.nio.ByteBuffer;

import org.powertac.common.Timeslot;

public class AgentState implements ISerialize {
	private float cash;
	private Timeslot timeslot;
	private MarketState marketstate;
	
	public AgentState() {
		this.marketstate = new MarketState();
	}
	
	public void setCash(float cash) {
		this.cash = cash;
	}
	
	public float getCash() {
		return this.cash;
	}
	
	public MarketState getMarketState() {
		return this.marketstate;
	}
	
	public void advance(Timeslot timeslot) {
		this.timeslot = timeslot;
	}
	
	public Timeslot getCurrentTimeslot() {
		return this.timeslot;
	}

	@Override
	public int getSizeInBytes() {
		return 20 + this.marketstate.getSizeInBytes();
	}

	@Override
	public void serialize(ByteBuffer buffer) {
		float hour = (timeslot.slotInDay() + 1) / 24f;
		float day = (timeslot.dayOfWeek() + 1) / 7f;
		
		buffer.putFloat(cash);
		buffer.putFloat((float) Math.cos(2 * Math.PI * hour));
		buffer.putFloat((float) Math.cos(2 * Math.PI * day));
		buffer.putFloat((float) Math.sin(2 * Math.PI * hour));
		buffer.putFloat((float) Math.sin(2 * Math.PI * day));
		
		this.marketstate.serialize(buffer);
	}

}
