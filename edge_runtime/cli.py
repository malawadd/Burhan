"""Developer commands. Server binds loopback; the camera is explicitly started in UI."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="InspectFlow local edge runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Start the local dashboard")
    serve.add_argument("--port", type=int, default=8765)
    probe = sub.add_parser("probe-camera", help="Probe camera without saving media")
    probe.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.command == "probe-camera":
        from .sources import probe_camera
        print(json.dumps(probe_camera(args.index), indent=2))
    else:
        import uvicorn
        uvicorn.run("edge_runtime.app:app", host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
