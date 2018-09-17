package org.powertac.extreme.actors;

import java.util.List;

import org.powertac.common.Competition;
import org.powertac.common.Timeslot;
import org.powertac.extreme.backend.IDeserialize;
import org.powertac.extreme.backend.ISerialize;
import org.powertac.samplebroker.interfaces.BrokerContext;

public abstract class Actor<T extends ISerialize> implements IDeserialize {
	
	private BrokerContext context;
	
	public Actor(BrokerContext context) {
		this.context = context;
	}

	public BrokerContext getContext() {
		return context;
	}

	public abstract void act(Competition competition, List<Timeslot> enabledTimeslots, T state);

}
