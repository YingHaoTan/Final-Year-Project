package org.powertac.extreme.models;

import java.nio.ByteBuffer;

import org.powertac.common.TariffSpecification;
import org.powertac.extreme.backend.ISerialize;

public class TariffInfo implements ISerialize {
	private int subscribers;
	private double powerusage;
	private TariffSpecification spec;
	private double uregfee;
	private double dregfee;
	
	public TariffInfo() {
		this.subscribers = 0;
		this.powerusage = 0;
	}

	public TariffSpecification getTariff() {
		return spec;
	}
	
	public void setTariff(TariffSpecification spec) {
		if(spec == null || this.spec == null || spec.getPowerType() != this.spec.getPowerType()) {
			this.subscribers = 0;
			this.uregfee = 0;
			this.dregfee = 0;
		}
		
		this.spec = spec;
	}
	
	public void subscribe(int subscribers) {
		this.subscribers += subscribers;
	}
	
	public void withdraw(int withdrawers) {
		this.subscribers -= withdrawers;
	}
	
	public void recordPowerUsage(double powerusage) {
		this.powerusage += powerusage;
	}
	
	public void setUpRegulationFee(double uregfee) {
		this.uregfee = uregfee;
	}
	
	public void setDownRegulationFee(double dregfee) {
		this.dregfee = dregfee;
	}
	
	public int getSubscribers() {
		return this.subscribers;
	}
	
	public float getPowerUsagePerCustomer() {
		return (float) (this.subscribers > 0? this.powerusage / this.subscribers: 0);
	}
	
	public void resetPowerUsageCounter() {
		this.powerusage = 0;
	}

	@Override
	public int getSizeInBytes() {
		return 144;
	}

	@Override
	public void serialize(ByteBuffer buffer) {
		for(int i = 0; i < TariffConstants.POWER_TYPE_LIST.size() + 1; i++) {
			if(spec == null) {
				if(i == 0)
					buffer.putFloat(1f);
				else
					buffer.putFloat(0f);
			}
			else {
				if(i == TariffConstants.POWER_TYPE_LIST.indexOf(spec.getPowerType()) + 1)
					buffer.putFloat(1f);
				else
					buffer.putFloat(0f);
			}
		}
		
		if(spec != null) {
			int subscribers = getSubscribers();
			if(subscribers < 0)
				throw new RuntimeException("WTF 0?");
			
			buffer.putFloat(getSubscribers());
			buffer.putFloat(getPowerUsagePerCustomer());
			for(int i = 0; i < TariffConstants.TIME_OF_USE_SLOTS; i++)
				buffer.putFloat((float) spec.getRates().get(i).getValue());
			for(int i = 0; i < TariffConstants.TIME_OF_USE_SLOTS; i++)
				buffer.putFloat((float) spec.getRates().get(i).getMaxCurtailment());
			buffer.putFloat((float) spec.getRates().get(TariffConstants.TIME_OF_USE_SLOTS).getValue());
			buffer.putFloat((float) spec.getRates().get(TariffConstants.TIME_OF_USE_SLOTS).getMaxCurtailment());
			if(spec.getPowerType().isStorage()) {
				buffer.putFloat((float) spec.getRegulationRates().get(0).getUpRegulationPayment());
				buffer.putFloat((float) spec.getRegulationRates().get(0).getDownRegulationPayment());
			}
			else {
				buffer.putFloat(0f);
				buffer.putFloat(0f);
			}
			buffer.putFloat((float) uregfee);
			buffer.putFloat((float) dregfee);
			buffer.putFloat((float) spec.getPeriodicPayment());
			buffer.putFloat((float) spec.getEarlyWithdrawPayment());
		}
		else {
			buffer.putFloat(0f);
			buffer.putFloat(0f);
			for(int i = 0; i < TariffConstants.TIME_OF_USE_SLOTS; i++)
				buffer.putFloat(0f);
			for(int i = 0; i < TariffConstants.TIME_OF_USE_SLOTS; i++)
				buffer.putFloat(0f);
			buffer.putFloat(0f);
			buffer.putFloat(0f);
			buffer.putFloat(0f);
			buffer.putFloat(0f);
			buffer.putFloat(0f);
			buffer.putFloat(0f);
			buffer.putFloat(0f);
			buffer.putFloat(0f);
		}
		
		this.powerusage = 0;
	}
}