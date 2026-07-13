"""Allow running with: python -m genesys_memory or genesys-memory CLI entry point."""
from __future__ import annotations

import asyncio
from genesys_memory.server import main as server_main


def main() -> None:
    """Entry point for console script."""
    asyncio.run(server_main())


if __name__ == "__main__":
    main()
