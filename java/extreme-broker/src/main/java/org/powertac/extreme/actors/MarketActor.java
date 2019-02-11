package org.powertac.extreme.actors;

import java.nio.ByteBuffer;
import java.util.List;

import org.powertac.common.Competition;
import org.powertac.common.Order;
import org.powertac.common.Timeslot;
import org.powertac.extreme.models.MarketState;
import org.powertac.samplebroker.interfaces.BrokerContext;

public class MarketActor extends Actor<MarketState> {
	
	private enum MarketAction {
		NoOp, Transact
	}
	
	private MarketAction[] actions;
	private double[] price;
	private double[] quantity;
	
	public MarketActor(BrokerContext context) {
		super(context);
		this.actions = new MarketAction[24];
		this.price = new double[24];
		this.quantity = new double[24];
	}

	@Override
	public int getSizeInBytes() {
		return 288;
	}

	@Override
	public void deserialize(ByteBuffer buffer) {
		for(int i = 0; i < actions.length; i++)
			actions[i] = MarketAction.values()[(int) buffer.getFloat()];
		for(int i = 0; i < price.length; i++)
			price[i] = buffer.getFloat();
		for(int i = 0; i < quantity.length; i++)
			quantity[i] = buffer.getFloat();
	}

	@Override
	public void act(Competition competition, List<Timeslot> enabledTimeslots, MarketState state) {
		BrokerContext context = this.getContext();
		for (int i = 0; i < actions.length; i++) {
			double p = price[i];
			double q = quantity[i];
			
			if(actions[i] != MarketAction.NoOp && Math.abs(q) >= competition.getMinimumOrderQuantity()) {
				Order order;
				order = new Order(context.getBroker(), enabledTimeslots.get(i).getSerialNumber(), q, p);
				
				context.sendMessage(order);
			}
		}
	}

}
