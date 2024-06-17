"""Contains key helpers"""

import py_nillion_client as nillion


def getUserKeyFromFile(userkey_filepath: str) -> nillion.UserKey:
    """
    Loads user key from file.

    Args:
        userkey_filepath (str): Path to user key file.

    Returns:
        nillion.UserKey: User key.
    """
    return nillion.UserKey.from_file(userkey_filepath)


def getNodeKeyFromFile(nodekey_filepath: str) -> nillion.NodeKey:
    """
    Loads node key from file.

    Args:
        nodekey_filepath (str): Path to node key file.

    Returns:
        nillion.NodeKey: Node key.
    """
    return nillion.NodeKey.from_file(nodekey_filepath)
