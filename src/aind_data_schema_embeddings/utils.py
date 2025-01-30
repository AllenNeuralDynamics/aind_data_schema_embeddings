"""Utilities for the AIND Metadata Embeddings service."""

import logging
import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from sshtunnel import SSHTunnelForwarder

load_dotenv()

escaped_username = quote_plus(os.getenv("DOC_DB_USERNAME"))
escaped_password = quote_plus(os.getenv("DOC_DB_PASSWORD"))

CONNECTION_STRING = (
    f"mongodb://{escaped_username}:{escaped_password}@localhost:27017/"
    "?directConnection=true&authMechanism=SCRAM-SHA-1&retryWrites=false"
)


def create_ssh_tunnel():
    """Create an SSH tunnel to the Document Database."""
    try:
        return SSHTunnelForwarder(
            ssh_address_or_host=(
                os.getenv("DOC_DB_SSH_HOST"),
                22,
            ),
            ssh_username=os.getenv("DOC_DB_SSH_USERNAME"),
            ssh_password=os.getenv("DOC_DB_SSH_PASSWORD"),
            remote_bind_address=(os.getenv("DOC_DB_HOST"), 27017),
            local_bind_address=(
                "localhost",
                27017,
            ),
        )
    except Exception as e:
        logging.error(f"Error creating SSH tunnel: {e}")


class ResourceManager:
    """Resource Manager to open and close ssh tunnel"""

    def __init__(self):
        """Constructor"""
        self.ssh_server = None
        self.client = None
        self.async_client = None

    def __enter__(self):
        """Creates ssh tunnel"""
        try:
            self.ssh_server = create_ssh_tunnel()
            self.ssh_server.start()
            logging.info("SSH tunnel opened")

            self.client = MongoClient(CONNECTION_STRING)
            self.async_client = AsyncIOMotorClient(CONNECTION_STRING)
            logging.info("Successfully connected to MongoDB")

            return self
        except Exception as e:
            logging.exception(e)
            self.__exit__()
            raise

    def __exit__(self):
        """Closes ssh tunnel"""
        if self.client:
            self.client.close()
        if self.async_client:
            self.async_client.close()
        if self.ssh_server:
            self.ssh_server.stop()
        logging.info("Resources cleaned up")
