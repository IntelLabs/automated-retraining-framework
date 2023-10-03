# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import asyncio
import pickle
import sys
from typing import Callable, Optional, Tuple

import pandas as pd


class AsyncClient:
    """AsyncClient creates an object to send
    data to server, and receive the response.
    AsyncClient intended to work for only one
    request/response communication, and closes
    connection after.
    """

    def __init__(self, port: int = 9999) -> None:
        """Instantiates an AsyncClient

        Args:
            port (int, optional): Port on localhost
            that should be used for communicating with server. Defaults to 9999.
        """
        self.port = port

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
        """Instantiates an AsyncServer

        Args:
            callback (Callable): function that should be called
            when a request is received by the server.
            port (int, optional): port on localhost where the
            server will receive messages on. Defaults to 9999.
        """
        self.port = port
        self.callback = callback

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
