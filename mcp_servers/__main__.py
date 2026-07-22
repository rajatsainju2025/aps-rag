"""Run all MCP servers. Use: python -m mcp_servers"""
import sys

if __name__ == "__main__":
    server = sys.argv[1] if len(sys.argv) > 1 else "controls"
    if server == "controls":
        from mcp_servers.controls import mcp
    elif server == "archiver":
        from mcp_servers.archiver import mcp
    else:
        print(f"Unknown server: {server}. Use 'controls' or 'archiver'.")
        sys.exit(1)
    mcp.run()
