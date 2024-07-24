"""Dot Product example script"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio

import nada_numpy.client as na_client
import numpy as np
import py_nillion_client as nillion
from cosmpy.aerial.client import LedgerClient
from cosmpy.aerial.wallet import LocalWallet
from cosmpy.crypto.keypairs import PrivateKey
from dotenv import load_dotenv
from nillion_python_helpers import (create_nillion_client,
                                    create_payments_config, get_quote,
                                    get_quote_and_pay, pay_with_quote)
from py_nillion_client import NodeKey, UserKey
from sklearn.linear_model import Ridge

from common.utils import compute, store_program, store_secret_array

home = os.getenv("HOME")

load_dotenv(f"{home}/.config/nillion/nillion-devnet.env")
# load_dotenv(f"/workspaces/ai/.nillion-testnet.env")


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
    program_name = "linear_regression_256"
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

    program_id = await store_program(
        client,
        payments_wallet,
        payments_client,
        user_id,
        cluster_id,
        program_name,
        program_mir_path,
    )

    ##### STORE SECRETS
    print("-----STORE SECRETS")

    # Input data
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    b = np.array([1.0, 2.0, 3.0])

    # Minmax normalization of data
    dmin = np.min([np.min(A), np.min(b)])
    dmax = np.max([np.max(A), np.max(b)])
    minmax = lambda x: ((x - dmin) / (dmax - dmin)) * 2 - 1
    LOG_SCALE = 8
    SCALE = 1 << LOG_SCALE
    # Minmax normalization of data
    A = minmax(A)
    b = minmax(b)

    clf = Ridge(alpha=0, fit_intercept=False)
    clf.fit(A, b)

    # Scale data into (-SCALE, SCALE) range
    A = np.round(A * SCALE).astype(np.int64)
    b = np.round(b * SCALE).astype(np.int64)

    print(A)
    print(b)
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

    # Create and store secrets for two parties
    store_id_B = await store_secret_array(
        client,
        payments_wallet,
        payments_client,
        cluster_id,
        program_id,
        b,
        "b",
        nillion.SecretInteger,
        1,
        permissions,
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

    result = await compute(
        client,
        payments_wallet,
        payments_client,
        program_id,
        cluster_id,
        compute_bindings,
        [store_id_A, store_id_B],
        computation_time_secrets,
        verbose=1,
    )

    w_0 = result["w_0"]
    w_1 = result["w_1"]
    w_2 = result["w_2"]
    b = result["b"]

    coeff = np.array([w_0, w_1, w_2]) / b

    print("--------------------")
    print("🔍  Expected Coefficients: ", clf.coef_)
    print("✅  Coefficients: ", coeff)
    return coeff


# Run the main function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
