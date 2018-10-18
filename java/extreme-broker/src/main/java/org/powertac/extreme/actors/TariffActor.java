package org.powertac.extreme.actors;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import org.powertac.common.Competition;
import org.powertac.common.Rate;
import org.powertac.common.RateCore;
import org.powertac.common.RegulationRate;
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
		NONE, REVOKE, UP_BALANCE_ORDER, DOWN_BALANCE_ORDER, ACTIVATE
	}
	
	private int actorIdx;
	private int tariffIdx;
	private TariffAction action;
	private PowerType type;
	private double variable_rate[];
	private double variable_ratio[];
	private double fixed_rate;
	private double curtail_ratio;
	private double up_regulation;
	private double down_regulation;
	private double pp;

	public TariffActor(BrokerContext context, int idx) {
		super(context);
		this.actorIdx = idx;
		this.variable_rate = new double[TariffConstants.TIME_OF_USE_SLOTS];
		this.variable_ratio = new double[TariffConstants.TIME_OF_USE_SLOTS];
	}

	@Override
	public void deserialize(ByteBuffer buffer) {
		this.tariffIdx = ((int) buffer.getFloat()) + (TariffConstants.TARIFF_PER_ACTOR * this.actorIdx);
		this.action = TariffAction.values()[(int) buffer.getFloat()];
		this.type = TariffConstants.POWER_TYPE_LIST.get((int) buffer.getFloat());
		for(int i = 0; i < this.variable_rate.length; i++)
			this.variable_rate[i] = buffer.getFloat();
		for(int i = 0; i < this.variable_ratio.length; i++)
			this.variable_ratio[i] = buffer.getFloat();
		this.fixed_rate = buffer.getFloat();
		this.curtail_ratio = buffer.getFloat();
		this.up_regulation = buffer.getFloat();
		this.down_regulation = buffer.getFloat();
		this.pp = buffer.getFloat();
	}

	@Override
	public int getSizeInBytes() {
		return 128;
	}

	@Override
	public void act(Competition competition, List<Timeslot> enabledTimeslots, TariffState state) {
		BrokerContext context = this.getContext();

		TariffInfo info = state.getTariffInfo(this.tariffIdx);
		if(this.action == TariffAction.REVOKE) {
			if(info.getTariff() != null) {
				TariffRevoke revoke = new TariffRevoke(context.getBroker(), info.getTariff());
				context.sendMessage(revoke);
				info.setTariff(null);
			}
		}
		else if(this.action == TariffAction.UP_BALANCE_ORDER || this.action == TariffAction.DOWN_BALANCE_ORDER) {
			TariffSpecification spec = info.getTariff();
			if(spec != null && spec.getPowerType().isStorage()) {
				BalancingOrder order;
				if(this.action == TariffAction.UP_BALANCE_ORDER) {
					order = new BalancingOrder(context.getBroker(), spec, 1.0, fixed_rate);
					info.setUpRegulationFee(fixed_rate);
				}
				else {
					order = new BalancingOrder(context.getBroker(), spec, -1.0, -fixed_rate);
					info.setDownRegulationFee(fixed_rate);
				}
				
				context.sendMessage(order);
			}
		}
		else if(this.action == TariffAction.ACTIVATE) {
			double signnum = this.type.isConsumption()? -1.0: 1.0;
			
			TariffSpecification spec = new TariffSpecification(context.getBroker(), this.type)
					.withPeriodicPayment(signnum * this.pp);
			List<RateCore> rates = new ArrayList<RateCore>();
			
			int begin = 0;
			int offset = 24 / TariffConstants.TIME_OF_USE_SLOTS;
			for(int i = 0; i < this.variable_rate.length; i++) {
				rates.add(new Rate().withValue(signnum * this.variable_rate[i])
						.withDailyBegin(begin)
						.withDailyEnd(begin + offset));
				
				begin = (begin + offset) % this.variable_rate.length;
			}
			
			rates.add(new Rate().withValue(signnum * this.fixed_rate));
			
			if(this.type.isStorage()) {
				rates.add(new RegulationRate().withUpRegulationPayment(this.up_regulation)
						.withDownRegulationPayment(-this.down_regulation));
			}
			else if (this.type.isInterruptible()) {
				for(int i = 0; i < this.variable_ratio.length; i++)
					rates.set(i, ((Rate)rates.get(i)).withMaxCurtailment(this.variable_ratio[i]));
				rates.set(this.variable_ratio.length, ((Rate)rates.get(this.variable_ratio.length)).withMaxCurtailment(this.curtail_ratio));
			}
			
			for(RateCore rate: rates)
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

}
