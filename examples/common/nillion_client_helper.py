"""Client helpers"""

import os

import py_nillion_client as nillion

from examples.common.nillion_payments_helper import create_payments_config


def create_nillion_client(userkey: str, nodekey: str) -> nillion.NillionClient:
    """
    Creates Nillion network client from user and node key.

    Args:
        userkey (str): User key.
        nodekey (str): Node key.

    Returns:
        nillion.NillionClient: Nillion client object.
    """
    bootnodes = [os.getenv("NILLION_BOOTNODE_MULTIADDRESS")]
    payments_config = create_payments_config()

    return nillion.NillionClient(
        nodekey, bootnodes, nillion.ConnectionMode.relay(), userkey, payments_config
    )
