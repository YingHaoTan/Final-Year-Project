package org.powertac.extreme.models;

import java.util.Arrays;
import java.util.List;

import org.powertac.common.enumerations.PowerType;

public class TariffConstants {
	public final static List<PowerType> POWER_TYPE_LIST = Arrays.asList(PowerType.BATTERY_STORAGE, PowerType.CHP_PRODUCTION, PowerType.CONSUMPTION, 
			PowerType.ELECTRIC_VEHICLE, PowerType.FOSSIL_PRODUCTION, PowerType.INTERRUPTIBLE_CONSUMPTION, PowerType.PRODUCTION, 
			PowerType.PUMPED_STORAGE_PRODUCTION, PowerType.RUN_OF_RIVER_PRODUCTION, PowerType.SOLAR_PRODUCTION, PowerType.STORAGE,
			PowerType.THERMAL_STORAGE_CONSUMPTION, PowerType.WIND_PRODUCTION);
	public final static int TIME_OF_USE_SLOTS = 6;
	public final static int TARIFF_PER_ACTOR = 4;
	public final static int NUM_TARIFF_ACTOR = 5;
	public final static int NUM_MIN_DAY_DURATION = 2;
}
