package org.powertac.extreme.models;

import java.nio.ByteBuffer;

import org.powertac.common.CustomerInfo;
import org.powertac.common.TariffSpecification;
import org.powertac.common.TariffTransaction;
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
	
	@Autowired
	private BrokerPropertiesService configurator;
	@Autowired
	private CustomerRepo customerdb;
	private int[] powertypepop;
	private float[] bootstrapPowerUsage;
	private TariffInfo[] tariffinfo;
	
	public TariffState() {
		this.powertypepop = new int[TariffConstants.POWER_TYPE_LIST.size()];
		this.bootstrapPowerUsage = new float[TariffConstants.POWER_TYPE_LIST.size()];
		this.tariffinfo = new TariffInfo[TariffConstants.NUM_TARIFF_ACTOR * TariffConstants.TARIFF_PER_ACTOR];
		for(int i = 0; i < this.tariffinfo.length; i++)
			this.tariffinfo[i] = new TariffInfo();
	}
	
	@Override
	public void initialize(BrokerContext broker) {
		configurator.configureMe(this);
	}
	
	public synchronized void handleMessage(CustomerBootstrapData data) {
		CustomerInfo customer = customerdb.findByNameAndPowerType(data.getCustomerName(), data.getPowerType());
		double[] netusage = data.getNetUsage();
		
		int ptIndex = TariffConstants.POWER_TYPE_LIST.indexOf(data.getPowerType());
		int custpopulation = customer.getPopulation();
		int totalpopulation = powertypepop[ptIndex] + custpopulation;
		float ousage = bootstrapPowerUsage[ptIndex];
		float ofrac = powertypepop[ptIndex] / (float) totalpopulation;
		float cusage = (float) netusage[netusage.length - 1];
		float cfrac = customer.getPopulation() / (float) totalpopulation;
		
		powertypepop[ptIndex] = totalpopulation;
		bootstrapPowerUsage[ptIndex] = ousage * ofrac + cusage * cfrac;
	}
	
	public synchronized void handleMessage(TariffTransaction ttx) {
		TariffSpecification spec = ttx.getTariffSpec();
		
		if(spec != null) {
			TariffTransaction.Type txType = ttx.getTxType();
			TariffInfo info = this.getTariffInfo(spec);
						
			if(info != null) {
				if (TariffTransaction.Type.SIGNUP == txType) {
					info.subscribe(ttx.getCustomerCount());
				}
				else if(TariffTransaction.Type.WITHDRAW == txType) {
					info.withdraw(ttx.getCustomerCount());
				}
				else if (ttx.isRegulation() || TariffTransaction.Type.PRODUCE == txType || TariffTransaction.Type.CONSUME == txType) {
					info.recordPowerUsage(ttx.getKWh());
				}
			}
		}
	}
	
	public TariffInfo getTariffInfo(int index) {
		return this.tariffinfo[index];
	}
	
	private TariffInfo getTariffInfo(TariffSpecification spec) {
		TariffInfo info = null;
		
		for(TariffInfo tmp: this.tariffinfo) {
			TariffSpecification tmpSpec = tmp.getTariff();
			
			if(tmpSpec != null && tmpSpec.getId() == spec.getId()) {
				info = tmp;
				break;
			}
		}
		
		return info;
	}

	@Override
	public int getSizeInBytes() {
		int size =  104;
		
		for(TariffInfo info: this.tariffinfo)
			size = size + info.getSizeInBytes();
		
		return size;
	}

	@Override
	public void serialize(ByteBuffer buffer) {
		for(int i = 0; i < this.powertypepop.length; i++)
			buffer.putFloat(this.powertypepop[i]);
		for(int i = 0; i < this.bootstrapPowerUsage.length; i++)
			buffer.putFloat(this.bootstrapPowerUsage[i]);
		
		for(TariffInfo info: this.tariffinfo)
			info.serialize(buffer);
	}

}
