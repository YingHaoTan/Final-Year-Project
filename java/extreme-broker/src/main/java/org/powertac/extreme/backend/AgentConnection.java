package org.powertac.extreme.backend;

import java.io.Closeable;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class AgentConnection implements Closeable {
	private Socket socket;
	private DataOutputStream writer;
	private DataInputStream reader;
	
	public AgentConnection(int port) throws UnknownHostException, IOException {
		socket = new Socket(InetAddress.getLocalHost().getHostName(), port);
		writer = new DataOutputStream(socket.getOutputStream());
		reader = new DataInputStream(socket.getInputStream());
	}
	
	public ByteBuffer read(int readbytes) throws IOException {
		ByteBuffer buffer = ByteBuffer.allocate(readbytes).order(ByteOrder.BIG_ENDIAN);
		reader.readFully(buffer.array());
		
		return buffer;
	}
	
	public void write(ByteBuffer payload) throws IOException {
		writer.write(payload.array());
	}
	
	@Override
	public void close() throws IOException {
		reader.close();
		writer.close();
		socket.close();
	}

}
