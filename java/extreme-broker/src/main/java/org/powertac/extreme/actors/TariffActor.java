package org.powertac.extreme.actors;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import org.powertac.common.Competition;
import org.powertac.common.Rate;
import org.powertac.common.TariffSpecification;
import org.powertac.common.Timeslot;
import org.powertac.common.enumerations.PowerType;
import org.powertac.common.msg.BalancingOrder;
import org.powertac.common.msg.TariffRevoke;
import org.powertac.extreme.models.TariffConstants;
import org.powertac.extreme.models.TariffInfo;
import org.powertac.extreme.models.TariffState;
import org.powertac.samplebroker.interfaces.BrokerContext;

public class TariffActor extends Actor<TariffState> {
	
	private enum TariffAction {
		NONE, REVOKE, BALANCE_ORDER, ACTIVATE
	}
	
	private int index;
	private TariffAction action;
	private PowerType type;
	private double variable_rate[];
	private double variable_ratio[];
	private double fixed_rate;
	private double curtail_ratio;
	private double pp;
	private double ewp;
	private double md;

	public TariffActor(BrokerContext context) {
		super(context);
		this.variable_rate = new double[TariffConstants.TIME_OF_USE_SLOTS];
		this.variable_ratio = new double[TariffConstants.TIME_OF_USE_SLOTS];
	}

	@Override
	public void deserialize(ByteBuffer buffer) {
		this.index = (int) buffer.getFloat();
		this.action = TariffAction.values()[(int) buffer.getFloat()];
		this.type = TariffConstants.POWER_TYPE_LIST.get((int) buffer.getFloat());
		for(int i = 0; i < this.variable_rate.length; i++)
			this.variable_rate[i] = buffer.getFloat();
		for(int i = 0; i < this.variable_ratio.length; i++)
			this.variable_ratio[i] = buffer.getFloat();
		this.fixed_rate = buffer.getFloat();
		this.pp = buffer.getFloat();
		this.ewp = buffer.getFloat();
		this.md = buffer.getFloat();
		this.curtail_ratio = buffer.getFloat();
	}

	@Override
	public int getSizeInBytes() {
		return 128;
	}

	@Override
	public void act(Competition competition, List<Timeslot> enabledTimeslots, TariffState state) {
		BrokerContext context = this.getContext();

		TariffInfo info = state.getTariffInfo(this.index);
		if(this.action == TariffAction.REVOKE) {
			if(info.getTariff() != null) {
				TariffRevoke revoke = new TariffRevoke(context.getBroker(), info.getTariff());
				context.sendMessage(revoke);
				info.setTariff(null);
			}
		}
		else if(this.action == TariffAction.BALANCE_ORDER) {
			TariffSpecification spec = info.getTariff();
			if(spec != null && spec.getPowerType().isConsumption() && spec.getPowerType().isInterruptible()) {
				BalancingOrder order = new BalancingOrder(context.getBroker(), spec, curtail_ratio, fixed_rate);
				context.sendMessage(order);
			}
		}
		else if(this.action == TariffAction.ACTIVATE) {
			if(this.type.isConsumption()) {
				TariffSpecification spec = new TariffSpecification(context.getBroker(), this.type)
						.withPeriodicPayment(-this.pp)
						.withEarlyWithdrawPayment(-this.ewp)
						.withMinDuration((long) (this.md));
				
				List<Rate> rates = new ArrayList<Rate>();
				int begin = 0;
				int offset = 24 / this.variable_rate.length;
				for(int i = 0; i < this.variable_rate.length; i++) {
					rates.add(new Rate().withValue(-this.variable_rate[i])
							.withDailyBegin(begin)
							.withDailyEnd(begin + offset));
					
					begin = (begin + offset) % this.variable_rate.length;
				}
				
				rates.add(new Rate().withValue(-this.fixed_rate));
				
				if (this.type.isInterruptible()) {
					for(int i = 0; i < this.variable_ratio.length; i++)
						rates.set(i, rates.get(i).withMaxCurtailment(this.variable_ratio[i]));
					rates.set(this.variable_ratio.length, rates.get(this.variable_ratio.length).withMaxCurtailment(this.curtail_ratio));
				}
				
				for(Rate rate: rates)
					spec.addRate(rate);
				
				if(info.getTariff() != null) {
					TariffSpecification oldspec = info.getTariff();
					if(oldspec.getPowerType().equals(spec.getPowerType()))
						spec.addSupersedes(info.getTariff().getId());
					context.sendMessage(spec);
					
					TariffRevoke revoke = new TariffRevoke(context.getBroker(), oldspec);
					context.sendMessage(revoke);
				}
				else {
					context.sendMessage(spec);
				}
				
				info.setTariff(spec);
			}
		}
		
		for(int i = 0; i < TariffConstants.MAX_NUM_TARIFF; i++)
			state.getTariffInfo(i).resetPowerUsageCounter();
	}

}
