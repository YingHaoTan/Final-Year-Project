package org.powertac.extreme;

import java.io.IOException;
import java.nio.ByteBuffer;

import org.powertac.common.Competition;
import org.powertac.common.config.ConfigurableValue;
import org.powertac.common.repo.TimeslotRepo;
import org.powertac.extreme.actors.AgentActor;
import org.powertac.extreme.backend.AgentConnection;
import org.powertac.extreme.models.AgentState;
import org.powertac.samplebroker.interfaces.Activatable;
import org.powertac.samplebroker.interfaces.BrokerContext;
import org.powertac.samplebroker.interfaces.Initializable;
import org.powertac.samplebroker.core.BrokerPropertiesService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class Agent implements Initializable, Activatable {
	@Autowired
	private BrokerPropertiesService configurator;
	@Autowired
	private TimeslotRepo timeslotRepo;
	@ConfigurableValue(valueType = "Integer")
	private int backendPortNumber = 16000;
	@Autowired
	private AgentState state;
	
	private Competition competition;
	private AgentConnection connection;
	private AgentActor actor;

	@Override
	public synchronized void activate(int timeslotIndex) {
		try {
			connection.write(state.serialize());
			actor.deserialize(connection.read(actor.getSizeInBytes()));
			actor.act(competition, timeslotRepo.enabledTimeslots(), state);
		} catch(IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void initialize(BrokerContext broker) {
		configurator.configureMe(this);
		
		this.actor = new AgentActor(broker);
		
		try {
			this.connection = new AgentConnection(backendPortNumber);
			this.connection.write(ByteBuffer.wrap(new byte[] {0}));
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	public synchronized void handleMessage(Competition comp) {
		this.competition = comp;
	}

}
