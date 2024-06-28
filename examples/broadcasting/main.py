"""Broadcasting example script"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import py_nillion_client as nillion
import numpy as np

import nada_numpy.client as na_client
from py_nillion_client import NodeKey, UserKey
from dotenv import load_dotenv
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey

from config import DIM

from nillion_python_helpers import (
    create_nillion_client,
    get_quote_and_pay,
    get_quote,
    pay_with_quote,
    create_payments_config,
)

from common.utils import (
    store_secret_array
)

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


    quote_store_program = await get_quote(client, nillion.Operation.store_program(program_mir_path), cluster_id)

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

    # Create a permissions object to attach to the stored secret
    permissions = nillion.Permissions.default_for_user(client.user_id)
    permissions.add_compute_permissions({client.user_id: {program_id}})

    # Create a secret
    store_id_A = await store_secret_array(
        client,
        payments_wallet,
        payments_client,
        cluster_id,
        program_id,
        A,
        "A",
        nillion.SecretInteger,
        1,
        permissions,
    )

    store_id_C = await store_secret_array(
        client,
        payments_wallet,
        payments_client,
        cluster_id,
        program_id,
        C,
        "C",
        nillion.SecretInteger,
        1,
        permissions,
    )
    
    # Create and store secrets for two parties

    B = np.ones([DIM])
    D = np.ones([DIM])

    store_id_B = await store_secret_array(
        client,
        payments_wallet,
        payments_client,
        cluster_id,
        program_id,
        B,
        "B",
        nillion.SecretInteger,
        1,
        permissions,
    )

    store_id_D = await store_secret_array(
        client,
        payments_wallet,
        payments_client,
        cluster_id,
        program_id,
        D,
        "D",
        nillion.SecretInteger,
        1,
        permissions,
    )

    # Set up the compute bindings for the parties
    compute_bindings = nillion.ProgramBindings(program_id)

    for party_name in party_names[:-1]:
        compute_bindings.add_input_party(party_name, party_id)
    compute_bindings.add_output_party(party_names[-1], party_id)

    print(f"Computing using program {program_id}")
    print(
        f"Use secret store_id: {store_id_A}, {store_id_B}, {store_id_C}, {store_id_D}"
    )

    ##### COMPUTE
    print("-----COMPUTE")

    # Bind the parties in the computation to the client to set input and output parties
    compute_bindings = nillion.ProgramBindings(program_id)
    compute_bindings.add_input_party(party_names[0], party_id)
    compute_bindings.add_input_party(party_names[1], party_id)
    compute_bindings.add_output_party(party_names[2], party_id)

    # Create a computation time secret to use
    computation_time_secrets = nillion.NadaValues({})

    # Get cost quote, then pay for operation to compute
    receipt_compute = await get_quote_and_pay(
        client,
        nillion.Operation.compute(program_id, computation_time_secrets),
        payments_wallet,
        payments_client,
        cluster_id,
    )

    # Compute, passing all params including the receipt that shows proof of payment
    uuid = await client.compute(
        cluster_id,
        compute_bindings,
        [store_id_A, store_id_B, store_id_C, store_id_D],
        computation_time_secrets,
        receipt_compute,
    )
    print(f"Computing using program {program_id}")
    print(f"Use secret store_id: {store_id_A}, {store_id_B}, {store_id_C}, {store_id_D}")

    # Print compute result
    print(f"The computation was sent to the network. compute_id: {uuid}")
    while True:
        compute_event = await client.next_compute_event()
        if isinstance(compute_event, nillion.ComputeFinishedEvent):
            print(f"✅  Compute complete for compute_id {compute_event.uuid}")
            print(f"🖥️  The result is {compute_event.result.value}")
            return compute_event.result.value



# Run the main function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
