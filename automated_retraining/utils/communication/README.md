# Edge-Data Center Communication

This collection of utilities enable communication between the edge and data center systems in order to implment retraining. 

## Localhost Communication

As a proof-of-concept demonstration edge-data center retraining is run on a single system using network communication via the `localhost` or `127.0.0.1` IP address. 

### Client-Server

To communicate, a client and server are used where the edge system is the client and the data center system is the server. The data center instantiates the server first and waits for messages to be sent from the edge. 

#### Data Center
The data center server implements the class `AsyncServer`, which uses the `asyncio` Python library to run asyncronously. A `callback` function is passed to `AsyncServer` and runs when a message is received. This callback is an external function in the `DataCenterModule`. 

The `start_server` function starts the server and listens on the selected port, which can be changed if the default conflicts on the system. The server runs asyncronously such that the main program will not block when the server is started. 

The `receive_send` function receives messages from the client, calls the callback function, and sends the response back to the client. 

The `AsyncServer` expects to receive a `RequestBlock` object from the client, and responds with a `ResponseBlock` object. The `ReqestBlock` and `ResponseBlock` objects are described below.  

```python
class AsyncServer:
    """AsyncServer creates a server on localhost that
    awaits requests from AsyncClient. A member function
    handles formatting the data and receiveing/writting
    messages, while external callback function is used
    for processing data in the message.
    The AsyncServer serves until stopped, and handles
    multiple requests from the AsyncClient.
    """

    def __init__(self, callback: Callable, port: int = 9999) -> None:
        ...
   
    async def start_server(self) -> None:
        """Start serving on self.port. Will
        use self.receive_send function to
        format data for receive and reply.
        Call to this function needs to be wrapped
        in asyncio.run() to start server.
        """
        server = await asyncio.start_server(self.receive_send, "127.0.0.1", self.port)

        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
        print(f"Serving on {addrs}")

        async with server:
            await server.serve_forever()

    async def receive_send(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Format data for recieve, processing, and response.
        Called when server receives a message. Collects
        sent data, formats for the callback function, and
        calls callback. Return from the callback is formatted
        as a ResponseBlock, and sent to edge as a reply.
        Args:
            reader (asyncio.StreamReader): created when new connection received.
            Reads data transmitted from client.
            writer (asyncio.StreamWriter): created when new connection received.
            Writes data to be sent to client.
        """
        print("waiting for data")
        data = await reader.read(-1)

        request_block = pickle.loads(data)

        checkpoint = self.callback(
            message_type=request_block.request_type,
            query_dataframe=request_block.query_data,
            random_dataframe=request_block.random_data,
        )

        response_block = ResponseBlock(checkpoint=checkpoint)

        writer.write(pickle.dumps(response_block))
        await writer.drain()
        writer.close()
```

#### Edge
The edge system uses the `AsyncClient` class to create temporary clients that send a single message to the server and then close the connection. 

The `send_data` function formats the message in a `RequestBlock`, creates a connection with the server, sends the message and waits for the response, ending the connection when the response is received. 

The `AsyncClient` sends messages as a `RequestBlock` object, and expects a `ResponseBlock` object in the response from the server. The `ReqestBlock` and `ResponseBlock` objects are described below.  

```python
class AsyncClient:
    """AsyncClient creates an object to send
    data to server, and receive the response.
    AsyncClient intended to work for only one
    request/response communication, and closes
    connection after.
    """

    def __init__(self, port: int = 9999) -> None:
        ...

    async def send_data(
        self,
        message_type: str,
        query_data: Optional[pd.DataFrame] = None,
        random_data: Optional[pd.DataFrame] = None,
        port: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Client function to send data to waiting server.
        This function is used by the edge to send requests
        to the datacenter that is serving on localhost. A
        message type and potentially two dataframes are sent
        via a RequestBlock data structure.
        Args:
            message_type (str): sent by edge to indicate what type of message was sent, and direct
            the datacenter how to process the request.
            query_dataframe (Optional[pd.DataFrame], optional): dataframe with information necessary
            to process the samples from the edge selected with active learning query. Defaults to None.
            random_dataframe (Optional[pd.DataFrame], optional): dataframe with information necessary
            to process the randomly sampled data sent from the edge. Defaults to None.
            port (Optional[int], optional): port to send data to if different than default port. Defaults to None.
        Returns:
            Tuple[bool, str]: retraining_flag is flag indicating if retraining is required.
            checkpoint is string with path to the model checkpoint.
        """
        if port is None:
            port = self.port

        reader, writer = await asyncio.open_connection("127.0.0.1", port)

        request_block = RequestBlock(
            request_type=message_type, query_data=query_data, random_data=random_data
        )

        writer.write(pickle.dumps(request_block))
        writer.write_eof()

        data = await reader.read(-1)
        response_block = pickle.loads(data)

        writer.close()

        return response_block.checkpoint
```

### Request Response Objects

`RequestBlock` and `ResponseBlock` objects are used to ensure messages sent between client and server maintain a specific format. 

Data sent via network connection must be serializable, and specifically the Python `pickle` package is used to serialize the objects. Because of this `Pandas Dataframes` are sent rather than `Pytorch Datasets` which are not serializable. 

```python
class RequestBlock:
    """Data structure to hold data sent by edge to
    datacenter via request messages
    """

    def __init__(
        self,
        request_type: str,
        query_data: Optional[pd.DataFrame] = None,
        random_data: Optional[pd.DataFrame] = None,
    ) -> None:
        """Instantiate a RequestBlock
        Args:
            request_type (str): string message indicating how to
            process data in message. Valid request_types are:
            'handshake', 'normal', 'prep_retraining', 'retraining'
            query_dataframe (Optional[pd.DataFrame], optional): dataframe with information necessary
            to process the samples from the edge selected with active learning query. Defaults to None.
            random_dataframe (Optional[pd.DataFrame], optional): dataframe with information necessary
            to process the randomly sampled data sent from the edge. Defaults to None.
        """
        self.request_type = request_type
        self.query_data = query_data
        self.random_data = random_data
```

```python
class ResponseBlock:
    """Data structure to hold responses sent by datacenter
    to edge via response messages
    """

    def __init__(
        self, retraining_flag: Optional[bool] = True, checkpoint: Optional[str] = None
    ) -> None:
        """Instantiate a ResponseBlock
        Args:
            retraining_flag (bool): Boolean flag indicating if retraining is required.
            checkpoint (Optional[str], optional): String with path to new model
            checkpoint after retraining. Defaults to None.
        """
        self.retraining_flag: Optional[bool] = retraining_flag
        self.checkpoint: Optional[str] = checkpoint
```

