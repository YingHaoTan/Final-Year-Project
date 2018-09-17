package org.powertac.extreme.models;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;

import org.powertac.common.CustomerInfo;
import org.powertac.common.TariffSpecification;
import org.powertac.common.TariffTransaction;
import org.powertac.common.enumerations.PowerType;
import org.powertac.common.msg.CustomerBootstrapData;
import org.powertac.common.repo.CustomerRepo;
import org.powertac.extreme.backend.ISerialize;
import org.powertac.samplebroker.core.BrokerPropertiesService;
import org.powertac.samplebroker.interfaces.BrokerContext;
import org.powertac.samplebroker.interfaces.Initializable;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class TariffState implements ISerialize, Initializable {
	
	public class TariffInfo {
		private int[] subscribers;
		private float[] cpowerusage;
		private float[] npowerusage;
		private TariffSpecification spec;
		
		public TariffInfo() {
			this.subscribers = new int[14];
			this.cpowerusage = new float[14];
			this.npowerusage = new float[14];
		}

		public TariffSpecification getTariff() {
			return spec;
		}
		
		public void setTariff(TariffSpecification spec) {
			this.spec = spec;
		}
		
		public void advancePUCounters() {
			for(int i = 0; i < this.cpowerusage.length; i++)
				if(this.subscribers[i] > 0 && this.npowerusage[i] != 0)
					this.cpowerusage[i] = this.npowerusage[i] / this.subscribers[i];
		}
		
		public void resetPUCounters() {
			Arrays.fill(this.npowerusage, 0f);
		}
	}
	
	@Autowired
	private BrokerPropertiesService configurator;
	@Autowired
	private CustomerRepo customerdb;
	private int[] powertypepop;
	private TariffInfo tariffinfo;
	
	public TariffState() {
		this.powertypepop = new int[14];
		this.tariffinfo = new TariffInfo();
	}
	
	private int convertPTIndex(PowerType pt) {
		List<PowerType> ptList = Arrays.asList(PowerType.BATTERY_STORAGE, PowerType.CHP_PRODUCTION, PowerType.CONSUMPTION, PowerType.ELECTRIC_VEHICLE,
				PowerType.FOSSIL_PRODUCTION, PowerType.INTERRUPTIBLE_CONSUMPTION, PowerType.PRODUCTION, PowerType.PUMPED_STORAGE_PRODUCTION,
				PowerType.RUN_OF_RIVER_PRODUCTION, PowerType.SOLAR_PRODUCTION, PowerType.SOLAR_PRODUCTION, PowerType.STORAGE,
				PowerType.THERMAL_STORAGE_CONSUMPTION, PowerType.WIND_PRODUCTION);
		
		return ptList.indexOf(pt);
	}
	
	@Override
	public void initialize(BrokerContext broker) {
		configurator.configureMe(this);
	}
	
	public synchronized void handleMessage(CustomerBootstrapData data) {
		CustomerInfo customer = customerdb.findByNameAndPowerType(data.getCustomerName(), data.getPowerType());
		double[] netusage = data.getNetUsage();
		
		int custpopulation = customer.getPopulation();
		int totalpopulation = powertypepop[convertPTIndex(data.getPowerType())] + custpopulation;
		float ousage = tariffinfo.npowerusage[convertPTIndex(data.getPowerType())];
		float ofrac = powertypepop[convertPTIndex(data.getPowerType())] / (float) totalpopulation;
		float cusage = (float) netusage[netusage.length - 1];
		float cfrac = customer.getPopulation() / (float) totalpopulation;
		
		powertypepop[convertPTIndex(data.getPowerType())] = totalpopulation;
		tariffinfo.cpowerusage[convertPTIndex(data.getPowerType())] = ousage * ofrac + cusage * cfrac;
	}
	
	public synchronized void handleMessage(TariffTransaction ttx) {
		TariffSpecification spec = ttx.getTariffSpec();
		
		if(spec != null) {
			TariffTransaction.Type txType = ttx.getTxType();
			CustomerInfo cinfo = ttx.getCustomerInfo();
			
			if (TariffTransaction.Type.SIGNUP == txType) {
				this.tariffinfo.subscribers[convertPTIndex(cinfo.getPowerType())] += ttx.getCustomerCount();
			}
			else if(TariffTransaction.Type.WITHDRAW == txType) {
				this.tariffinfo.subscribers[convertPTIndex(cinfo.getPowerType())] -= ttx.getCustomerCount();
			}
			else if (ttx.isRegulation() || TariffTransaction.Type.PRODUCE == txType || TariffTransaction.Type.CONSUME == txType) {
				this.tariffinfo.npowerusage[convertPTIndex(cinfo.getPowerType())] += ttx.getKWh();
			}
		}
	}
	
	public TariffInfo getTariffInfo() {
		return this.tariffinfo;
	}

	@Override
	public int getSizeInBytes() {
		return 200;
	}

	@Override
	public void serialize(ByteBuffer buffer) {
		this.tariffinfo.advancePUCounters();
		
		for(int i = 0; i < this.powertypepop.length; i++) {
			if(this.powertypepop[i] > 0)
				buffer.putFloat(this.tariffinfo.subscribers[i] / (float) this.powertypepop[i]);
			else
				buffer.putFloat(0f);
			
			buffer.putFloat(this.tariffinfo.cpowerusage[i]);
			buffer.putFloat(this.tariffinfo.subscribers[i]);
		}
		
		if(this.tariffinfo.spec != null) {
			TariffSpecification spec = this.tariffinfo.getTariff();
			buffer.putFloat((float) spec.getRates().get(0).getValue());
			buffer.putFloat((float) spec.getEarlyWithdrawPayment());
			buffer.putFloat((float) spec.getPeriodicPayment());
			buffer.putFloat((float) spec.getMinDuration());
			buffer.putFloat((float) spec.getRates().get(0).getMaxCurtailment());
			buffer.putFloat(1f);
		}
		else {
			buffer.putFloat(0f);
			buffer.putFloat(0f);
			buffer.putFloat(0f);
			buffer.putFloat(0f);
			buffer.putFloat(0f);
			buffer.putFloat(0f);
		}
	}

}
