# When working with nilup, it creates libraries in version 0.1.0, which are slightly different from the ones available in the latest version.
# To install the latest version of the libraries with nada-algebra you can use the following script:
# Use: python install_version.py -v v2024-05-20-9c71763d1
# This script will install the specified version of the libraries and remove the previous versions.
# After running the script, you can run the tests again to check if the libraries are working correctly.
# If you want to install the default version, you can run the script without the -v option.
# Use: python install_version.py


import argparse
import subprocess
import sys

# Default version
DEFAULT_VERSION = "v2024-05-20-9c71763d1"

# ANSI escape codes for bold red text
BOLD_RED = "\033[1;31m"
GREEN = "\033[0;32m"
RESET = "\033[0m"


def usage():
    print(f"{GREEN} Usage: {sys.argv[0]} [-v version] {RESET}")
    sys.exit(1)


def run_command(command, success_message):
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"{GREEN}{success_message}{RESET}")
    except subprocess.CalledProcessError:
        print(f"{BOLD_RED}Error: Command failed - {command}{RESET}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Script to manage project dependencies and versions."
    )
    parser.add_argument(
        "-v", "--version", type=str, help="Specify the version to install"
    )
    args = parser.parse_args()

    version = args.version if args.version else DEFAULT_VERSION

    # Install dependencies with poetry
    run_command("poetry install", "Running: poetry install")

    # Uninstall specified Python packages
    run_command(
        "pip uninstall -y nada_dsl py_nillion_client",
        "Running: pip uninstall -y nada_dsl py_nillion_client",
    )

    # Install the specified version using nilup
    run_command(
        f"nilup install {version} --nada-dsl --python-client",
        f"Running: nilup install {version} --nada-dsl --python-client",
    )

    # Use the specified version
    run_command(f"nilup use {version}", f"Running: nilup use {version}")

    print(f"{GREEN}Done.{RESET}")


if __name__ == "__main__":
    main()
