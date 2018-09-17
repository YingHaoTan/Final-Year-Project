package org.powertac.extreme.actors;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;

import org.powertac.common.Competition;
import org.powertac.common.Rate;
import org.powertac.common.TariffSpecification;
import org.powertac.common.Timeslot;
import org.powertac.common.enumerations.PowerType;
import org.powertac.common.msg.TariffRevoke;
import org.powertac.extreme.models.TariffState;
import org.powertac.samplebroker.interfaces.BrokerContext;

public class TariffActor extends Actor<TariffState> {
	
	private static final List<PowerType> PTYPES = Arrays.asList(PowerType.BATTERY_STORAGE, PowerType.CHP_PRODUCTION, 
			PowerType.CONSUMPTION, PowerType.ELECTRIC_VEHICLE,
			PowerType.FOSSIL_PRODUCTION, PowerType.INTERRUPTIBLE_CONSUMPTION, PowerType.PRODUCTION, PowerType.PUMPED_STORAGE_PRODUCTION,
			PowerType.RUN_OF_RIVER_PRODUCTION, PowerType.SOLAR_PRODUCTION, PowerType.SOLAR_PRODUCTION, PowerType.STORAGE,
			PowerType.THERMAL_STORAGE_CONSUMPTION, PowerType.WIND_PRODUCTION);
	
	private enum ActionType {
		NONE, REVOKE, SUPERSEDE_ACTIVATE
	}
	
	private enum TariffType {
		NORMAL, INTERRUPTIBLE
	}
	
	private ActionType atype;
	private TariffType ttype;
	private PowerType ptype;
	private float rate;
	private float ewp;
	private float pp;
	private float md;
	private float mcv;

	public TariffActor(BrokerContext context) {
		super(context);
	}

	@Override
	public void deserialize(ByteBuffer buffer) {
		this.atype = ActionType.values()[(int) buffer.getFloat()];
		this.ttype = TariffType.values()[(int) buffer.getFloat()];
		this.ptype = PTYPES.get((int)buffer.getFloat());
		this.rate = buffer.getFloat();
		this.ewp = buffer.getFloat();
		this.pp = buffer.getFloat();
		this.md = buffer.getFloat();
		this.mcv = buffer.getFloat();
	}

	@Override
	public int getSizeInBytes() {
		return 28;
	}

	@Override
	public void act(Competition competition, List<Timeslot> enabledTimeslots, TariffState state) {
		BrokerContext context = this.getContext();
		
		if(this.atype == ActionType.REVOKE) {
			if(state.getTariffInfo().getTariff() != null) {
				TariffRevoke revoke = new TariffRevoke(context.getBroker(), state.getTariffInfo().getTariff());
				context.sendMessage(revoke);
				state.getTariffInfo().setTariff(null);
			}
		}
		else if(this.atype == ActionType.SUPERSEDE_ACTIVATE) {
			TariffSpecification spec = new TariffSpecification(context.getBroker(), this.ptype)
					.withEarlyWithdrawPayment(this.ewp)
					.withPeriodicPayment(this.pp)
					.withMinDuration((long) this.md);
			Rate rate = new Rate().withValue(this.rate);
			if (this.ttype == TariffType.INTERRUPTIBLE)
				rate = rate.withMaxCurtailment(this.mcv);
			
			spec.addRate(rate);
			
			if(state.getTariffInfo().getTariff() != null)
				spec.addSupersedes(state.getTariffInfo().getTariff().getId());
			
			context.sendMessage(spec);
			state.getTariffInfo().setTariff(spec);
		}
		
		state.getTariffInfo().resetPUCounters();
	}

}
