package org.powertac.extreme.backend;

import java.nio.ByteBuffer;

public interface IDeserialize extends ITransmittable {
	
	void deserialize(ByteBuffer buffer);

}
