"""General utils functions"""

import os
import time
from typing import Any, Callable, Dict, List, Type

import nada_algebra.client as na_client
import numpy as np
import py_nillion_client as nillion


def async_timer(file_path: os.PathLike) -> Callable:
    """
    Decorator function to measure and log the execution time of asynchronous functions.

    Args:
        file_path (os.PathLike): File to write performance metrics to.

    Returns:
        Callable: Wrapped function with timer.
    """

    def decorator(func: Callable) -> Callable:
        """
        Decorator function.

        Args:
            func (Callable): Function to decorate.

        Returns:
            Callable: Decorated function.
        """

        async def wrapper(*args, **kwargs) -> Any:
            """
            Returns function result and writes execution time to file.

            Returns:
                Any: Function result.
            """
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            with open(file_path, "a") as file:
                file.write(f"{elapsed_time:.6f},\n")
            return result

        return wrapper

    return decorator


async def store_program(
    client: nillion.NillionClient,
    user_id: str,
    cluster_id: str,
    program_name: str,
    program_mir_path: str,
    verbose: bool = True,
) -> str:
    """
    Asynchronous function to store a program on the nillion client.

    Args:
        client (nillion.NillionClient): Nillion client.
        user_id (str): User ID.
        cluster_id (str): Cluster ID.
        program_name (str): Program name.
        program_mir_path (str): Path to program MIR.
        verbose (bool, optional): Verbosity level. Defaults to True.

    Returns:
        str: Program ID.
    """
    action_id = await client.store_program(cluster_id, program_name, program_mir_path)
    program_id = f"{user_id}/{program_name}"
    if verbose:
        print("Stored program. action_id:", action_id)
        print("Stored program_id:", program_id)
    return program_id


async def store_secrets(
    client: nillion.NillionClient,
    cluster_id: str,
    program_id: str,
    party_id: str,
    party_name: str,
    secret_array: np.ndarray,
    prefix: str,
    nada_type: Any,
):
    """
    Asynchronous function to store secrets on the nillion client.

    Args:
        client (nillion.NillionClient): Nillion client.
        cluster_id (str): Cluster ID.
        program_id (str): Program ID.
        party_id (str): Party ID.
        party_name (str): Party name.
        secret_array (np.ndarray): Secret array.
        prefix (str): Secrets name.
        nada_type (Any): Nada type.

    Returns:
        str: Store ID.
    """
    stored_secret = nillion.Secrets(na_client.array(secret_array, prefix, nada_type))
    secret_bindings = nillion.ProgramBindings(program_id)
    secret_bindings.add_input_party(party_name, party_id)

    store_id = await client.store_secrets(
        cluster_id, secret_bindings, stored_secret, None
    )
    return store_id


async def compute(
    client: nillion.NillionClient,
    cluster_id: str,
    compute_bindings: nillion.ProgramBindings,
    store_ids: List[str],
    computation_time_secrets: nillion.Secrets,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Asynchronous function to perform computation on the nillion client.

    Args:
        client (nillion.NillionClient): Nillion client.
        cluster_id (str): Cluster ID.
        compute_bindings (nillion.ProgramBindings): Compute bindings.
        store_ids (List[str]): List of data store IDs.
        computation_time_secrets (nillion.Secrets): Computation time secrets.
        verbose (bool, optional): Verbosity level. Defaults to True.

    Returns:
        Dict[str, Any]: Result of computation.
    """
    compute_id = await client.compute(
        cluster_id,
        compute_bindings,
        store_ids,
        computation_time_secrets,
        nillion.PublicVariables({}),
    )

    if verbose:
        print(f"The computation was sent to the network. compute_id: {compute_id}")
    while True:
        compute_event = await client.next_compute_event()
        if isinstance(compute_event, nillion.ComputeFinishedEvent):
            if verbose:
                print(f"✅  Compute complete for compute_id {compute_event.uuid}")
                print(f"🖥️  The result is {compute_event.result.value}")
            return compute_event.result.value
