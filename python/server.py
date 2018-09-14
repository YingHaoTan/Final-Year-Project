import socket
import struct
import select
import utility


class ServerHook:

    def get_observation_structure(self) -> struct.Struct:
        raise NotImplementedError("Observation structure is undefined")

    def get_output_structure(self) -> struct.Struct:
        raise NotImplementedError("Output structure is undefined")

    def setup(self, server, **kwargs):
        raise NotImplementedError("setup method is not implemented")

    def on_start(self, **kwargs):
        raise NotImplementedError("on_start method is not implemented")

    def on_step(self, observations, **kwargs):
        raise NotImplementedError("on_step method is not implemented")

    def on_stop(self, **kwargs):
        raise NotImplementedError("on_stop method is not implemented")


class Server:

    def __init__(self, num_clients: int, port: int, hook: ServerHook):
        self.ClientCount = num_clients
        self.Port = port
        self.Hook = hook
        self.ServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.Active = False

    def stop(self):
        self.Active = False

    def serve(self, **kwargs):
        buffers = tuple(bytearray(self.Hook.get_observation_structure().size) for _ in range(self.ClientCount))

        self.ServerSocket.bind((socket.gethostname(), self.Port))
        self.ServerSocket.listen(self.ClientCount)
        self.Active = True
        while self.Active:
            self.Hook.setup(self, **kwargs)
            clients = [self.ServerSocket.accept()[0] for _ in range(self.ClientCount)]
            utility.apply(lambda client: client.setblocking(0), clients)

            self.Hook.on_start(**kwargs)
            try:
                while self.Active:
                    uncleared_sockets = [*clients]
                    observations = [None] * self.ClientCount
                    bviews = [memoryview(buffer) for buffer in buffers]

                    while not all(observations):
                        rsocketlist, _, _ = select.select(uncleared_sockets, [], [])
                        for rsocket in rsocketlist:
                            index = clients.index(rsocket)
                            read_bytes = rsocket.recv_into(bviews[index], len(bviews[index]))
                            bviews[index] = bviews[index][read_bytes:]
                            if len(bviews[index]) == 0:
                                uncleared_sockets.remove(rsocket)
                                observations[index] = self.Hook.get_observation_structure().unpack_from(buffers[index])

                    uncleared_sockets = [*clients]
                    outputs = [self.Hook.get_output_structure().pack(*output)
                               for output in self.Hook.on_step(observations, **kwargs)]
                    while any(outputs):
                        _, wsocketlist, _ = select.select([], uncleared_sockets, [])
                        for wsocket in wsocketlist:
                            index = clients.index(wsocket)
                            write_bytes = wsocket.send(outputs[index])
                            outputs[index] = outputs[index][write_bytes:]
                            if len(outputs[index]) == 0:
                                uncleared_sockets.remove(wsocket)
                                outputs[index] = None
            except ConnectionResetError:
                utility.apply(lambda client: client.close(), clients)
                self.Hook.on_stop(**kwargs)

