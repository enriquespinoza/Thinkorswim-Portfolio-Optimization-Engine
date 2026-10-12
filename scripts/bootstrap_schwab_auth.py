from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schwab.auth import client_from_manual_flow
from src.integrations.schwab.auth_bootstrap import load_schwab_auth_config


def main() -> None:
    config = load_schwab_auth_config(interactive=True)

    print("=" * 88)
    print("SCHWAB MANUAL OAUTH BOOTSTRAP")
    print("=" * 88)
    print("Callback URL:", config.callback_url)
    print("Token path:", config.token_path)
    print()
    print(
        "Complete Schwab login in the browser, then copy the ENTIRE redirected "
        "URL from the address bar and paste it into the terminal prompt."
    )
    print("Do not share the redirected URL or token contents.")
    print()

    client_from_manual_flow(
        api_key=config.app_key,
        app_secret=config.app_secret,
        callback_url=config.callback_url,
        token_path=str(config.token_path),
        asyncio=False,
        enforce_enums=True,
    )

    if not config.token_path.is_file():
        raise RuntimeError("Schwab token file was not created.")

    print()
    print("Schwab authentication bootstrap: OK")
    print("Token written ->", config.token_path)


if __name__ == "__main__":
    main()
