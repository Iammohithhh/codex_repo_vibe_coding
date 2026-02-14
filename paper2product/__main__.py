"""Entry point: python -m paper2product"""
import sys

def main():
    # Default to v2 server
    if "--legacy" in sys.argv:
        from .server import run
    else:
        from .server_v2 import run
    port = 8000
    for arg in sys.argv[1:]:
        if arg.startswith("--port="):
            port = int(arg.split("=")[1])
    run(port=port)

if __name__ == "__main__":
    main()
