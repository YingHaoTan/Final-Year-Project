package org.powertac.extreme.backend;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public interface ISerialize extends ITransmittable {
	
	void serialize(ByteBuffer buffer);
	
	default ByteBuffer serialize() {
		ByteBuffer buffer = ByteBuffer.allocate(this.getSizeInBytes()).order(ByteOrder.BIG_ENDIAN);
		this.serialize(buffer);
		
		buffer.rewind();
		
		return buffer;
	}

}
