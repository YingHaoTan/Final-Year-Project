package org.powertac.extreme.actors;

import java.nio.ByteBuffer;
import java.util.List;

import org.powertac.common.Competition;
import org.powertac.common.Order;
import org.powertac.common.Timeslot;
import org.powertac.extreme.models.MarketState;
import org.powertac.samplebroker.interfaces.BrokerContext;

public class MarketActor extends Actor<MarketState> {
	private float[] price;
	private float[] quantity;
	
	public MarketActor(BrokerContext context) {
		super(context);
		this.price = new float[24];
		this.quantity = new float[24];
	}

	@Override
	public int getSizeInBytes() {
		return 192;
	}

	@Override
	public void deserialize(ByteBuffer buffer) {
		for(int i = 0; i < price.length; i++)
			price[i] = buffer.getFloat();
		for(int i = 0; i < quantity.length; i++)
			quantity[i] = buffer.getFloat();
	}

	@Override
	public void act(Competition competition, List<Timeslot> enabledTimeslots, MarketState state) {
		BrokerContext context = this.getContext();
		for (int i = 0; i < quantity.length; i++) {
			if (Math.abs(quantity[i]) >= competition.getMinimumOrderQuantity())
				context.sendMessage(new Order(context.getBroker(), 
						enabledTimeslots.get(i).getSerialNumber(), 
						(double) quantity[i] % 0.001, (double) price[i] % 0.01));
		}
	}

}
