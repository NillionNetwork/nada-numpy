"""Dot Product example script"""

import asyncio
import os
import pytest

import nada_numpy.client as na_client
from config import DIM
import numpy as np
from nillion_client import (
    InputPartyBinding,
    Network,
    NilChainPayer,
    NilChainPrivateKey,
    OutputPartyBinding,
    Permissions,
    SecretInteger,
    VmClient,
    PrivateKey,
)
from dotenv import load_dotenv

home = os.getenv("HOME")
load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")


# 1 Party running simple addition on 1 stored secret and 1 compute time secret
async def main():
    # Use the devnet configuration generated by `nillion-devnet`
    network = Network.from_config("devnet")

    # Create payments config and set up Nillion wallet with a private key to pay for operations
    nilchain_key: str = os.getenv("NILLION_NILCHAIN_PRIVATE_KEY_0")  # type: ignore
    payer = NilChainPayer(
        network,
        wallet_private_key=NilChainPrivateKey(bytes.fromhex(nilchain_key)),
        gas_limit=10000000,
    )

    # Use a random key to identify ourselves
    signing_key = PrivateKey()
    client = await VmClient.create(signing_key, network, payer)
    party_names = na_client.parties(3)
    program_name = "dot_product"
    program_mir_path = f"./target/{program_name}.nada.bin"

    ##### STORE PROGRAM
    print("-----STORE PROGRAM")

    # Store program
    program_mir = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), program_mir_path), "rb").read()
    program_id = await client.store_program(program_name, program_mir).invoke()

    # Print details about stored program
    print(f"Stored program_id: {program_id}")

    ##### STORE SECRETS
    print("-----STORE SECRETS Party 0")

    # Create a secret
    A = na_client.array(np.ones([DIM]), "A", SecretInteger)

    # Create a permissions object to attach to the stored secret
    permissions = Permissions.defaults_for_user(client.user_id).allow_compute(
        client.user_id, program_id
    )

    # Store a secret, passing in the receipt that shows proof of payment
    values_A = await client.store_values(
        A, ttl_days=5, permissions=permissions
    ).invoke()

    print("Stored values_A: ", values_A)

    print("-----STORE SECRETS Party 1")

    # Create a secret
    B = na_client.array(np.ones([DIM]), "B", SecretInteger)

    # Create a permissions object to attach to the stored secret
    permissions = Permissions.defaults_for_user(client.user_id).allow_compute(
        client.user_id, program_id
    )

    # Store a secret, passing in the receipt that shows proof of payment
    values_B = await client.store_values(
        B, ttl_days=5, permissions=permissions
    ).invoke()
    
    print("Stored values_B: ", values_B)

    ##### COMPUTE
    print("-----COMPUTE")

    # Bind the parties in the computation to the client to set input and output parties
    input_bindings = [InputPartyBinding(party_names[0], client.user_id), InputPartyBinding(party_names[1], client.user_id)]
    output_bindings = [OutputPartyBinding(party_names[2], [client.user_id])]

    # Create a computation time secret to use
    compute_time_values = {
        #"my_int2": SecretInteger(10)
    }

    # Compute, passing in the compute time values as well as the previously uploaded value.
    print(f"Invoking computation using program {program_id} and values id {values_A}, {values_B}")
    compute_id = await client.compute(
        program_id,
        input_bindings,
        output_bindings,
        values=compute_time_values,
        value_ids=[values_A, values_B],
    ).invoke()

    # Print compute result
    print(f"The computation was sent to the network. compute_id: {compute_id}")
    result = await client.retrieve_compute_results(compute_id).invoke()
    print(f"✅  Compute complete for compute_id {compute_id}")
    print(f"🖥️  The result is {result}")
    return result


if __name__ == "__main__":
    asyncio.run(main())
