"""Entry point: python -m paper2product"""
import os
import sys

def main():
    # Default to v2 server
    if "--legacy" in sys.argv:
        from .server import run
    else:
        from .server_v2 import run
    host = os.environ.get("P2P_HOST", "0.0.0.0")
    port = int(os.environ.get("P2P_PORT", "8000"))
    for arg in sys.argv[1:]:
        if arg.startswith("--port="):
            port = int(arg.split("=")[1])
        elif arg.startswith("--host="):
            host = arg.split("=")[1]
    run(host=host, port=port)

if __name__ == "__main__":
    main()
