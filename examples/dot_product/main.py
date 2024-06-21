"""Dot product Nada example"""

import asyncio
import os
from typing import Dict

import numpy as np
import py_nillion_client as nillion
from dotenv import load_dotenv
# Import helper functions for creating nillion client and getting keys
from nillion_python_helpers import (create_nillion_client, getNodeKeyFromFile,
                                    getUserKeyFromFile)

import nada_numpy.client as na_client
from examples.common.utils import compute, store_program, store_secret_array
from examples.dot_product.config import DIM

# Load environment variables from a .env file
load_dotenv()


# Main asynchronous function to coordinate the process
async def main() -> Dict:
    """Main nada program script"""

    print(f"USING {DIM} dims")

    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    userkey = getUserKeyFromFile(os.getenv("NILLION_USERKEY_PATH_PARTY_1"))
    nodekey = getNodeKeyFromFile(os.getenv("NILLION_NODEKEY_PATH_PARTY_1"))
    client = create_nillion_client(userkey, nodekey)
    party_id = client.party_id
    user_id = client.user_id
    party_names = na_client.parties(3)
    program_name = "dot_product"
    program_mir_path = f"./target/{program_name}.nada.bin"

    # Store the program
    program_id = await store_program(
        client, user_id, cluster_id, program_name, program_mir_path
    )

    # Create and store secrets for two parties
    A = np.ones([DIM])
    A_store_id = await store_secret_array(
        client,
        cluster_id,
        program_id,
        party_id,
        party_names[0],
        A,
        "A",
        nillion.SecretInteger,
    )

    B = np.ones([DIM])
    B_store_id = await store_secret_array(
        client,
        cluster_id,
        program_id,
        party_id,
        party_names[1],
        B,
        "B",
        nillion.SecretInteger,
    )

    # Set up the compute bindings for the parties
    compute_bindings = nillion.ProgramBindings(program_id)

    for party_name in party_names[:-1]:
        compute_bindings.add_input_party(party_name, party_id)
    compute_bindings.add_output_party(party_names[-1], party_id)

    print(f"Computing using program {program_id}")
    print(f"Use secret store_id: {A_store_id}, {B_store_id}")

    computation_time_secrets = nillion.Secrets({"my_int2": nillion.SecretInteger(10)})

    # Perform the computation and return the result
    result = await compute(
        client,
        cluster_id,
        compute_bindings,
        [A_store_id, B_store_id],
        computation_time_secrets,
    )
    return result


# Run the main function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
