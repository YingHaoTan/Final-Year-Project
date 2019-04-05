import socket
import struct
import select
import utility
from socket import timeout


class ServerResetException(Exception):
    pass


class ServerHook:

    @property
    def observation_structure(self) -> struct.Struct:
        raise NotImplementedError("Observation structure is undefined")

    @property
    def output_structure(self) -> struct.Struct:
        raise NotImplementedError("Output structure is undefined")

    def setup(self, server, **kwargs):
        raise NotImplementedError("setup method is not implemented")

    def on_start(self, **kwargs):
        raise NotImplementedError("on_start method is not implemented")

    def on_step(self, observations, **kwargs):
        raise NotImplementedError("on_step method is not implemented")

    def on_stop(self, **kwargs):
        raise NotImplementedError("on_stop method is not implemented")

    def on_reset(self):
        raise NotImplementedError("on_stop method is not implemented")


class Server:

    def __init__(self, num_clients: int, port: int, hook: ServerHook):
        self.client_count = num_clients
        self.port = port
        self.hook = hook
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__active__ = False
        self.__reset__ = False

    def stop(self):
        self.__active__ = False

    def reset(self):
        self.__reset__ = True

    def serve(self, **kwargs):
        buffers = tuple(bytearray(self.hook.observation_structure.size) for _ in range(self.client_count))

        self.server_socket.bind(('127.0.0.1', self.port))
        self.server_socket.listen(self.client_count)
        self.__active__ = True
        while self.__active__:
            self.server_socket.settimeout(300)
            self.hook.setup(self, **kwargs)
            clients = None
            try:
                clients = [self.server_socket.accept()[0] for _ in range(self.client_count)]

                utility.apply(lambda client: client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True), clients)
                utility.apply(lambda client: client.settimeout(self.server_socket.gettimeout()), clients)
                utility.apply(lambda client: client.recv(1), clients)
                utility.apply(lambda client: client.setblocking(0), clients)

                self.hook.on_start(**kwargs)

                while self.__active__:
                    if self.__reset__:
                        raise ServerResetException('Server raised a reset flag')

                    uncleared_sockets = [*clients]
                    observations = [None] * self.client_count
                    bviews = [memoryview(buffer) for buffer in buffers]

                    while not all(observations):
                        rsocketlist, _, esocketlist = select.select(uncleared_sockets, [], [],
                                                                    self.server_socket.gettimeout())
                        for rsocket in rsocketlist:
                            index = clients.index(rsocket)
                            read_bytes = rsocket.recv_into(bviews[index], len(bviews[index]))
                            bviews[index] = bviews[index][read_bytes:]
                            if len(bviews[index]) == 0:
                                uncleared_sockets.remove(rsocket)
                                observations[index] = self.hook.observation_structure.unpack_from(buffers[index])

                    uncleared_sockets = [*clients]
                    outputs = [self.hook.output_structure.pack(*output)
                               for output in self.hook.on_step(observations, **kwargs)]

                    while any(outputs):
                        _, wsocketlist, _ = select.select([], uncleared_sockets, [], 60)
                        for wsocket in wsocketlist:
                            index = clients.index(wsocket)
                            write_bytes = wsocket.send(outputs[index])
                            outputs[index] = outputs[index][write_bytes:]
                            if len(outputs[index]) == 0:
                                uncleared_sockets.remove(wsocket)
                                outputs[index] = None
            except (ConnectionResetError, ServerResetException, socket.timeout) as e:
                if clients is not None:
                    utility.apply(lambda client: client.close(), clients)
                if isinstance(e, ServerResetException):
                    self.__reset__ = False
                    self.hook.on_reset()
                self.hook.on_stop(**kwargs)
