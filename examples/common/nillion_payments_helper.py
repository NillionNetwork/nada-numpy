"""Nillion payment helpers"""

import os

import py_nillion_client as nillion


def create_payments_config() -> nillion.PaymentsConfig:
    """
    Creates a payment config from environment variables.

    Returns:
        nillion.PaymentsConfig: Payments config.
    """
    return nillion.PaymentsConfig(
        os.getenv("NILLION_BLOCKCHAIN_RPC_ENDPOINT"),
        os.getenv("NILLION_WALLET_PRIVATE_KEY"),
        int(os.getenv("NILLION_CHAIN_ID")),
        os.getenv("NILLION_PAYMENTS_SC_ADDRESS"),
        os.getenv("NILLION_BLINDING_FACTORS_MANAGER_SC_ADDRESS"),
    )
