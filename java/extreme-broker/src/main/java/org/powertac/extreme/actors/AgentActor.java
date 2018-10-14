package org.powertac.extreme.actors;

import java.nio.ByteBuffer;
import java.util.List;

import org.powertac.common.Competition;
import org.powertac.common.Timeslot;
import org.powertac.extreme.models.AgentState;
import org.powertac.extreme.models.TariffConstants;
import org.powertac.samplebroker.interfaces.BrokerContext;

public class AgentActor extends Actor<AgentState> {
	private MarketActor mktactor;
	private TariffActor tariffactors[];
	
	public AgentActor(BrokerContext context) {
		super(context);
		this.mktactor = new MarketActor(context);
		this.tariffactors = new TariffActor[TariffConstants.NUM_TARIFF_ACTOR];
		for(int i = 0; i < TariffConstants.NUM_TARIFF_ACTOR; i++)
			this.tariffactors[i] = new TariffActor(context, i);
	}

	@Override
	public void deserialize(ByteBuffer buffer) {
		this.mktactor.deserialize(buffer);
		for(TariffActor actor: this.tariffactors)
			actor.deserialize(buffer);
	}

	@Override
	public int getSizeInBytes() {
		int size = this.mktactor.getSizeInBytes();
		for(TariffActor actor: this.tariffactors)
			size += actor.getSizeInBytes();
		
		return size;
	}

	@Override
	public void act(Competition competition, List<Timeslot> enabledTimeslots, AgentState state) {
		this.mktactor.act(competition, enabledTimeslots, state.getMarketState());
		for(TariffActor actor: this.tariffactors)
			actor.act(competition, enabledTimeslots, state.getTariffState());
	}

}
