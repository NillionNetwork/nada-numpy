"""Rationals Nada example"""

import asyncio
import os

import py_nillion_client as nillion
from dotenv import load_dotenv

import nada_algebra.client as na_client
# Import helper functions for creating nillion client and getting keys
from examples.common.nillion_client_helper import create_nillion_client
from examples.common.nillion_keypath_helper import (getNodeKeyFromFile,
                                                    getUserKeyFromFile)
from examples.common.utils import compute, store_program, store_secrets

# Load environment variables from a .env file
load_dotenv()


# Main asynchronous function to coordinate the process
async def main() -> float:
    """Main nada program script"""

    cluster_id = os.getenv("NILLION_CLUSTER_ID")
    userkey = getUserKeyFromFile(os.getenv("NILLION_USERKEY_PATH_PARTY_1"))
    nodekey = getNodeKeyFromFile(os.getenv("NILLION_NODEKEY_PATH_PARTY_1"))
    client = create_nillion_client(userkey, nodekey)
    party_id = client.party_id
    user_id = client.user_id
    party_names = na_client.parties(3)
    program_name = "rational_numbers"
    program_mir_path = f"./target/{program_name}.nada.bin"

    # Store the program
    program_id = await store_program(
        client, user_id, cluster_id, program_name, program_mir_path
    )

    # Create and store secrets for two parties
    A = 3.2
    A_store_id = await store_secrets(
        client,
        cluster_id,
        program_id,
        party_id,
        party_names[0],
        A,
        "my_input_0",
        na_client.secret_rational,
    )

    B = 2.3
    B_store_id = await store_secrets(
        client,
        cluster_id,
        program_id,
        party_id,
        party_names[1],
        B,
        "my_input_1",
        na_client.secret_rational,
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
    result = await compute(
        client,
        cluster_id,
        compute_bindings,
        [A_store_id, B_store_id],
        computation_time_secrets,
        verbose=False,
    )
    output = na_client.float_from_rational(result["my_output_0"])
    print("✅  Compute complete")
    print("🖥️  The result is", output)
    return output


# Run the main function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
