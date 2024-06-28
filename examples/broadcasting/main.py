"""Broadcasting example script"""

import asyncio
import py_nillion_client as nillion
import os
import sys
import pytest
import numpy as np

import nada_numpy.client as na_client
from py_nillion_client import NodeKey, UserKey
from dotenv import load_dotenv
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey

from examples.broadcasting.config import DIM

from nillion_python_helpers import (
    create_nillion_client,
    pay,
    quote,
    pay_with_quote,
    create_payments_config,
)

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
home = os.getenv("HOME")
load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")



# Main asynchronous function to coordinate the process
async def main() -> None:
    """Main nada program"""

    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    grpc_endpoint = os.getenv("NILLION_NILCHAIN_GRPC")
    chain_id = os.getenv("NILLION_NILCHAIN_CHAIN_ID")
    seed = "my_seed"
    userkey = UserKey.from_seed((seed))
    nodekey = NodeKey.from_seed((seed))
    client = create_nillion_client(userkey, nodekey)
    party_id = client.party_id
    user_id = client.user_id

    party_names = na_client.parties(3)
    program_name = "broadcasting"
    program_mir_path = f"target/{program_name}.nada.bin"


    # Create payments config and set up Nillion wallet with a private key to pay for operations
    payments_config = create_payments_config(chain_id, grpc_endpoint)
    payments_client = LedgerClient(payments_config)
    payments_wallet = LocalWallet(
        PrivateKey(bytes.fromhex(os.getenv("NILLION_NILCHAIN_PRIVATE_KEY_0"))),
        prefix="nillion",
    )

    ##### STORE PROGRAM
    print("-----STORE PROGRAM")


    quote_store_program = await quote(client, nillion.Operation.store_program(), cluster_id)

    receipt_store_program = await pay_with_quote(
        quote_store_program, payments_wallet, payments_client
    )

    # Store program, passing in the receipt that shows proof of payment
    action_id = await client.store_program(
        cluster_id, program_name, program_mir_path, receipt_store_program
    )

    # Print details about stored program
    program_id = f"{user_id}/{program_name}"
    print("Stored program. action_id:", action_id)
    print("Stored program_id:", program_id)


    ##### STORE SECRETS
    print("-----STORE SECRETS")
    A = np.ones([DIM])
    C = np.ones([DIM])

    # Create a secret
    stored_secret = nillion.NadaValues(
        na_client.concat(
            [
                na_client.array(A, "A", nillion.SecretInteger),
                na_client.array(C, "C", nillion.SecretInteger),
            ]
        )
    )

    # Create a permissions object to attach to the stored secret
    permissions = nillion.Permissions.default_for_user(client.user_id)
    permissions.add_compute_permissions({client.user_id: {program_id}})

    # Get cost quote, then pay for operation to store the secret
    receipt_store = await pay(
        client,
        nillion.Operation.store_values(stored_secret),
        payments_wallet,
        payments_client,
        cluster_id,
    )

    # Store a secret, passing in the receipt that shows proof of payment
    A_C_store_id = await client.store_values(
        cluster_id, stored_secret, permissions, receipt_store
    )
    
    # Create and store secrets for two parties

    B = np.ones([DIM])
    D = np.ones([DIM])

    # Create a secret
    stored_secret = nillion.NadaValues(
        na_client.concat(
            [
                na_client.array(B, "B", nillion.SecretInteger),
                na_client.array(D, "D", nillion.SecretInteger),
            ]
        )
    )

    # Create a permissions object to attach to the stored secret
    permissions = nillion.Permissions.default_for_user(client.user_id)
    permissions.add_compute_permissions({client.user_id: {program_id}})

    # Get cost quote, then pay for operation to store the secret
    receipt_store = await pay(
        client,
        nillion.Operation.store_values(stored_secret),
        payments_wallet,
        payments_client,
        cluster_id,
    )

    # Store a secret, passing in the receipt that shows proof of payment
    B_D_store_id = await client.store_values(
        cluster_id, stored_secret, permissions, receipt_store
    )
    # Set up the compute bindings for the parties
    compute_bindings = nillion.ProgramBindings(program_id)

    for party_name in party_names[:-1]:
        compute_bindings.add_input_party(party_name, party_id)
    compute_bindings.add_output_party(party_names[-1], party_id)

    print(f"Computing using program {program_id}")
    print(
        f"Use secret store_id: {A_C_store_id}, {B_D_store_id}"
    )

    computation_time_secrets = nillion.Secrets({"my_int2": nillion.SecretInteger(10)})

    # Perform the computation and return the result
    result = await compute(
        client,
        cluster_id,
        compute_bindings,
        [A_C_store_id, B_D_store_id],
        computation_time_secrets,
    )
    return result


# Run the main function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
