package org.powertac.extreme.models;

import java.nio.ByteBuffer;
import java.util.List;

import org.powertac.common.BalancingTransaction;
import org.powertac.common.CapacityTransaction;
import org.powertac.common.CashPosition;
import org.powertac.common.DistributionTransaction;
import org.powertac.common.WeatherForecast;
import org.powertac.common.WeatherForecastPrediction;
import org.powertac.common.WeatherReport;
import org.powertac.common.repo.TimeslotRepo;
import org.powertac.extreme.backend.ISerialize;
import org.powertac.samplebroker.core.BrokerPropertiesService;
import org.powertac.samplebroker.interfaces.BrokerContext;
import org.powertac.samplebroker.interfaces.Initializable;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class AgentState implements ISerialize, Initializable {
	private final static int NUM_WEATHER_FORECAST_SLOTS = 24;
	
	private float cash;
	private float balanceTxCharge;
	private float distributionTxCharge;
	private float capacityTxCharge;
	private float temperature[];
	private float cloud[];
	private float windspeed[];
	private float winddirection[];
	@Autowired
	private MarketState marketstate;
	@Autowired
	private TariffState tariffstate;
	@Autowired
	private TimeslotRepo timeservice;
	@Autowired
	private BrokerPropertiesService configurator;
	
	public AgentState() {
		this.marketstate = new MarketState();
		this.temperature = new float[NUM_WEATHER_FORECAST_SLOTS + 1];
		this.cloud = new float[NUM_WEATHER_FORECAST_SLOTS + 1];
		this.windspeed = new float[NUM_WEATHER_FORECAST_SLOTS + 1];
		this.winddirection = new float[NUM_WEATHER_FORECAST_SLOTS + 1];
	}
	
	@Override
	public void initialize(BrokerContext broker) {
		configurator.configureMe(this);
	}
	
	public synchronized void handleMessage(CashPosition posn) {
		this.cash = (float) posn.getBalance();
	}
	
	public synchronized void handleMessage(BalancingTransaction tx) {
		this.balanceTxCharge = (float) tx.getCharge();
	}
	
	public synchronized void handleMessage(DistributionTransaction tx) {
		this.distributionTxCharge = (float) tx.getCharge();
	}
	
	public synchronized void handleMessage(CapacityTransaction tx) {
		this.capacityTxCharge = (float) tx.getCharge();
	}
	
	public synchronized void handleMessage(WeatherForecast forecast) {
		List<WeatherForecastPrediction> predictions = forecast.getPredictions();
		for(int i = 0; i < predictions.size(); i++) {
			WeatherForecastPrediction prediction = predictions.get(i);
			this.temperature[i] = (float) prediction.getTemperature();
			this.cloud[i] = (float) prediction.getCloudCover();
			this.windspeed[i] = (float) prediction.getWindSpeed();
			this.winddirection[i] = (float) prediction.getWindDirection();
		}
	}
	
	public synchronized void handleMessage(WeatherReport report) {
		this.temperature[NUM_WEATHER_FORECAST_SLOTS] = (float) report.getTemperature();
		this.cloud[NUM_WEATHER_FORECAST_SLOTS] = (float) report.getCloudCover();
		this.windspeed[NUM_WEATHER_FORECAST_SLOTS] = (float) report.getWindSpeed();
		this.winddirection[NUM_WEATHER_FORECAST_SLOTS] = (float) report.getWindDirection();
	}
	
	public MarketState getMarketState() {
		return this.marketstate;
	}
	
	public TariffState getTariffState() {
		return this.tariffstate;
	}

	@Override
	public int getSizeInBytes() {
		return 532 + this.marketstate.getSizeInBytes() + this.tariffstate.getSizeInBytes();
	}

	@Override
	public void serialize(ByteBuffer buffer) {
		float hour = timeservice.currentTimeslot().slotInDay() / 24f;
		float day = timeservice.currentTimeslot().dayOfWeek() / 7f;
		
		buffer.putFloat(cash);
		buffer.putFloat((float) Math.cos(2 * Math.PI * hour));
		buffer.putFloat((float) Math.cos(2 * Math.PI * day));
		buffer.putFloat((float) Math.sin(2 * Math.PI * hour));
		buffer.putFloat((float) Math.sin(2 * Math.PI * day));
		buffer.putFloat(balanceTxCharge);
		buffer.putFloat(distributionTxCharge);
		buffer.putFloat(capacityTxCharge);
		for(int i = 0; i < AgentState.NUM_WEATHER_FORECAST_SLOTS + 1; i++) {
			buffer.putFloat(temperature[i]);
			buffer.putFloat(cloud[i]);
			buffer.putFloat(windspeed[i]);
			buffer.putFloat((float)Math.cos(Math.toRadians(winddirection[i])));
			buffer.putFloat((float)Math.sin(Math.toRadians(winddirection[i])));
		}
		
		this.marketstate.serialize(buffer);
		this.tariffstate.serialize(buffer);
		
		this.balanceTxCharge = 0;
		this.distributionTxCharge = 0;
		this.capacityTxCharge = 0;
	}

}
