# Import necessary libraries and modules
import asyncio
import os
import sys
import time

import numpy as np
import py_nillion_client as nillion
import pytest
from dotenv import load_dotenv

# Add the parent directory to the system path to import modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import helper functions for creating nillion client and getting keys
from nillion_python_helpers import (create_nillion_client, getNodeKeyFromFile,
                                    getUserKeyFromFile)

import nada_numpy.client as na_client

# Load environment variables from a .env file
load_dotenv()
from dot_product.config.parameters import DIM


# Main asynchronous function to coordinate the process
async def main():
    print(f"USING: {DIM}")
    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    userkey = getUserKeyFromFile(os.getenv("NILLION_USERKEY_PATH_PARTY_1"))
    nodekey = getNodeKeyFromFile(os.getenv("NILLION_NODEKEY_PATH_PARTY_1"))
    client = create_nillion_client(userkey, nodekey)
    party_id = client.party_id
    user_id = client.user_id
    party_names = na_client.parties(3)
    program_name = "main"
    program_mir_path = f"./target/{program_name}.nada.bin"

    # Store the program
    action_id = await client.store_program(cluster_id, program_name, program_mir_path)
    program_id = f"{user_id}/{program_name}"
    print("Stored program. action_id:", action_id)
    print("Stored program_id:", program_id)

    # Create and store secrets for two parties
    stored_secret = nillion.Secrets({"my_input_0": na_client.SecretRational(3.2)})
    secret_bindings = nillion.ProgramBindings(program_id)
    secret_bindings.add_input_party(party_names[0], party_id)

    # Store the secret for the specified party
    A_store_id = await client.store_secrets(
        cluster_id, secret_bindings, stored_secret, None
    )

    stored_secret = nillion.Secrets({"my_input_1": na_client.SecretRational(2.3)})
    secret_bindings = nillion.ProgramBindings(program_id)
    secret_bindings.add_input_party(party_names[1], party_id)

    # Store the secret for the specified party
    B_store_id = await client.store_secrets(
        cluster_id, secret_bindings, stored_secret, None
    )

    # Set up the compute bindings for the parties
    compute_bindings = nillion.ProgramBindings(program_id)
    [
        compute_bindings.add_input_party(party_name, party_id)
        for party_name in party_names[:-1]
    ]
    compute_bindings.add_output_party(party_names[-1], party_id)

    print(f"Computing using program {program_id}")
    print(f"Use secret store_id: {A_store_id}, {B_store_id}")

    computation_time_secrets = nillion.Secrets({"my_int2": nillion.SecretInteger(10)})

    # Perform the computation and return the result
    compute_id = await client.compute(
        cluster_id,
        compute_bindings,
        [A_store_id, B_store_id],
        computation_time_secrets,
        nillion.PublicVariables({}),
    )

    # Monitor and print the computation result
    print(f"The computation was sent to the network. compute_id: {compute_id}")
    while True:
        compute_event = await client.next_compute_event()
        if isinstance(compute_event, nillion.ComputeFinishedEvent):
            print(f"✅  Compute complete for compute_id {compute_event.uuid}")
            print(f"🖥️  The result is {compute_event.result.value}")
            return compute_event.result.value
    return result


# Run the main function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
