package org.powertac.extreme;

import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;
import org.powertac.common.CashPosition;
import org.powertac.common.ClearedTrade;
import org.powertac.common.Competition;
import org.powertac.common.MarketPosition;
import org.powertac.common.Orderbook;
import org.powertac.common.Timeslot;
import org.powertac.common.config.ConfigurableValue;
import org.powertac.common.msg.MarketBootstrapData;
import org.powertac.common.repo.TimeslotRepo;
import org.powertac.extreme.actors.Actor;
import org.powertac.extreme.actors.MarketActor;
import org.powertac.extreme.backend.AgentConnection;
import org.powertac.extreme.backend.AgentState;
import org.powertac.extreme.backend.MarketState;
import org.powertac.samplebroker.interfaces.Activatable;
import org.powertac.samplebroker.interfaces.BrokerContext;
import org.powertac.samplebroker.interfaces.Initializable;
import org.powertac.samplebroker.core.BrokerPropertiesService;
import org.powertac.samplebroker.core.interfaces.MarketManager;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class Agent implements MarketManager, Initializable, Activatable {
	@Autowired
	private BrokerPropertiesService propertiesService;
	@Autowired
	private TimeslotRepo timeslotRepo;
	@ConfigurableValue(valueType = "Integer")
	private int backendPortNumber = 16000;
	
	private Competition competition;
	private BrokerContext context;
	private AgentConnection connection;
	private AgentState state;
	private Actor actor;
	private double meanMktPrice;

	@Override
	public synchronized void activate(int timeslotIndex) {
		Timeslot currentTimeslot = timeslotRepo.currentTimeslot();
		MarketState mktstate = state.getMarketState();
		for(Timeslot timeslot: timeslotRepo.enabledTimeslots())
			mktstate.setPowerBalance(context.getBroker().findMarketPositionByTimeslot(timeslot.getSerialNumber()));
		
		state.advance(currentTimeslot);
		try {
			connection.write(state.serialize());
			actor.deserialize(connection.read(actor.getSizeInBytes()));
			actor.act(competition, timeslotRepo.enabledTimeslots());
		} catch(IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void initialize(BrokerContext broker) {
		propertiesService.configureMe(this);
		
		this.context = broker;
		this.state = new AgentState();
		this.actor = new MarketActor(broker);
		
		try {
			this.connection = new AgentConnection(backendPortNumber);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public double getMeanMarketPrice() {
		return meanMktPrice;
	}
	
	public synchronized void handleMessage(Competition comp) {
		this.competition = comp;
	}
	
	public synchronized void handleMessage(ClearedTrade ct) {
		double smoothing = 0.5;
		meanMktPrice = smoothing * meanMktPrice + (1 - smoothing) * ct.getExecutionPrice();
		state.getMarketState().setCTPrice(ct);
	}
	
	public synchronized void handleMessage(Orderbook orderbook) {
		state.getMarketState().setUnclearedOrders(orderbook);
	}
	
	public synchronized void handleMessage(MarketPosition posn) {
		context.getBroker().addMarketPosition(posn, posn.getTimeslotIndex());
	}
	
	public synchronized void handleMessage(CashPosition posn) {
		state.setCash((float) posn.getBalance());
	}
	
	public synchronized void handleMessage(MarketBootstrapData data) {
		meanMktPrice = Arrays.asList(ArrayUtils.toObject(data.getMarketPrice()))
				.stream().mapToDouble(d -> d).average().orElse(0.0);
		
		state.getMarketState().bootstrap(data, timeslotRepo.currentTimeslot());
	}

}
