# -------- __main__.py --------
# Lets the package be run as a module: python -m sltp --help

# -------- Imports --------
import sys

from sltp.cli import main


# -------- Entry Point --------
if __name__ == "__main__":
    sys.exit(main())
