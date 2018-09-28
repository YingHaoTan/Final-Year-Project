package org.powertac.extreme.actors;

import java.nio.ByteBuffer;
import java.util.List;

import org.powertac.common.Competition;
import org.powertac.common.Timeslot;
import org.powertac.extreme.models.AgentState;
import org.powertac.samplebroker.interfaces.BrokerContext;

public class AgentActor extends Actor<AgentState> {
	private MarketActor mktactor;
	private TariffActor tariffactor;
	
	public AgentActor(BrokerContext context) {
		super(context);
		this.mktactor = new MarketActor(context);
		this.tariffactor = new TariffActor(context);
	}

	@Override
	public void deserialize(ByteBuffer buffer) {
		this.mktactor.deserialize(buffer);
		this.tariffactor.deserialize(buffer);
	}

	@Override
	public int getSizeInBytes() {
		return this.mktactor.getSizeInBytes() + this.tariffactor.getSizeInBytes();
	}

	@Override
	public void act(Competition competition, List<Timeslot> enabledTimeslots, AgentState state) {
		this.mktactor.act(competition, enabledTimeslots, state.getMarketState());
		this.tariffactor.act(competition, enabledTimeslots, state.getTariffState());
	}

}
