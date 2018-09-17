package org.powertac.extreme.models;

import java.nio.ByteBuffer;

import org.powertac.common.CashPosition;
import org.powertac.common.repo.TimeslotRepo;
import org.powertac.extreme.backend.ISerialize;
import org.powertac.samplebroker.core.BrokerPropertiesService;
import org.powertac.samplebroker.interfaces.BrokerContext;
import org.powertac.samplebroker.interfaces.Initializable;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class AgentState implements ISerialize, Initializable {
	private float cash;
	@Autowired
	private MarketState marketstate;
	@Autowired
	private TariffState tariffstate;
	@Autowired
	private TimeslotRepo timeservice;
	@Autowired
	private BrokerPropertiesService configurator;
	
	public AgentState() {
		this.marketstate = new MarketState();
	}
	
	@Override
	public void initialize(BrokerContext broker) {
		configurator.configureMe(this);
	}
	
	public synchronized void handleMessage(CashPosition posn) {
		this.cash = (float) posn.getBalance();
	}
	
	public MarketState getMarketState() {
		return this.marketstate;
	}

	@Override
	public int getSizeInBytes() {
		return 20 + this.marketstate.getSizeInBytes() + this.tariffstate.getSizeInBytes();
	}

	@Override
	public void serialize(ByteBuffer buffer) {
		float hour = timeservice.currentTimeslot().slotInDay() / 24f;
		float day = timeservice.currentTimeslot().dayOfWeek() / 7f;
		
		buffer.putFloat(cash);
		buffer.putFloat((float) Math.cos(2 * Math.PI * hour));
		buffer.putFloat((float) Math.cos(2 * Math.PI * day));
		buffer.putFloat((float) Math.sin(2 * Math.PI * hour));
		buffer.putFloat((float) Math.sin(2 * Math.PI * day));
		
		this.marketstate.serialize(buffer);
		this.tariffstate.serialize(buffer);
	}

}
