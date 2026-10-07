from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from dotenv import load_dotenv

from config.settings import PROJECT_ROOT

from src.integrations.schwab.read_only_client import (
    SchwabReadOnlyClient,
)


DEFAULT_MAX_TOKEN_AGE_SECONDS = 561_600.0


@dataclass(frozen=True)
class SchwabAuthConfig:
    """
    Local OAuth configuration for Schwab Trader API.

    Secrets are excluded from repr() so accidental
    logging of this dataclass does not expose them.
    """

    app_key: str = field(
        repr=False
    )

    app_secret: str = field(
        repr=False
    )

    callback_url: str

    token_path: Path

    interactive: bool = True

    max_token_age_seconds: float = (
        DEFAULT_MAX_TOKEN_AGE_SECONDS
    )


def _require_environment_value(
    environ: Mapping[str, str],
    name: str,
) -> str:
    value = str(
        environ.get(
            name,
            "",
        )
    ).strip()

    if not value:
        raise ValueError(
            f"Required environment variable "
            f"{name} is not configured."
        )

    return value


def validate_auth_config(
    config: SchwabAuthConfig,
) -> SchwabAuthConfig:
    if not isinstance(
        config,
        SchwabAuthConfig,
    ):
        raise TypeError(
            "config must be SchwabAuthConfig."
        )

    if not config.app_key.strip():
        raise ValueError(
            "Schwab app key cannot be empty."
        )

    if not config.app_secret.strip():
        raise ValueError(
            "Schwab app secret cannot be empty."
        )

    callback_url = (
        config.callback_url.strip()
    )

    parsed = urlparse(
        callback_url
    )

    if parsed.scheme.lower() != "https":
        raise ValueError(
            "Schwab callback URL must use HTTPS."
        )

    if not parsed.hostname:
        raise ValueError(
            "Schwab callback URL must contain "
            "a hostname."
        )

    token_path = Path(
        config.token_path
    )

    if token_path.name in {
        ".env",
        ".env.example",
    }:
        raise ValueError(
            "Schwab token path cannot point "
            "to an environment file."
        )

    if (
        token_path.suffix.lower()
        != ".json"
    ):
        raise ValueError(
            "Schwab token path must use "
            "a .json file."
        )

    try:
        max_token_age = float(
            config.max_token_age_seconds
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "max_token_age_seconds must "
            "be numeric."
        ) from exc

    if max_token_age <= 0:
        raise ValueError(
            "max_token_age_seconds must "
            "be positive."
        )

    return config


def load_schwab_auth_config(
    environ: Mapping[str, str] | None = None,
    project_root: Path = PROJECT_ROOT,
    interactive: bool = True,
) -> SchwabAuthConfig:
    """
    Load Schwab OAuth configuration.

    If environ is omitted, the project's .env file
    is loaded first.

    No authentication occurs here.
    No network requests occur here.
    No token file is created here.
    """
    project_root = Path(
        project_root
    ).resolve()

    if environ is None:
        dotenv_path = (
            project_root
            / ".env"
        )

        load_dotenv(
            dotenv_path=
                dotenv_path,

            override=False,
        )

        environ = os.environ

    app_key = (
        _require_environment_value(
            environ,
            "SCHWAB_APP_KEY",
        )
    )

    app_secret = (
        _require_environment_value(
            environ,
            "SCHWAB_APP_SECRET",
        )
    )

    callback_url = (
        _require_environment_value(
            environ,
            "SCHWAB_CALLBACK_URL",
        )
    )

    raw_token_path = (
        _require_environment_value(
            environ,
            "SCHWAB_TOKEN_PATH",
        )
    )

    token_path = Path(
        raw_token_path
    ).expanduser()

    if not token_path.is_absolute():
        token_path = (
            project_root
            / token_path
        )

    token_path = (
        token_path.resolve()
    )

    config = SchwabAuthConfig(
        app_key=
            app_key,

        app_secret=
            app_secret,

        callback_url=
            callback_url,

        token_path=
            token_path,

        interactive=
            bool(
                interactive
            ),

        max_token_age_seconds=
            DEFAULT_MAX_TOKEN_AGE_SECONDS,
    )

    return validate_auth_config(
        config
    )


def _load_easy_client() -> Callable[..., Any]:
    """
    Import schwab-py only when real authentication
    is requested.

    Keeping this import lazy allows the configuration
    and security tests to run without initiating any
    OAuth behavior.
    """
    try:
        from schwab.auth import (
            easy_client,
        )

    except ImportError as exc:
        raise RuntimeError(
            "schwab-py is not installed. "
            "Install the project dependency "
            "before attempting Schwab OAuth."
        ) from exc

    return easy_client


def create_authenticated_schwab_client(
    config: SchwabAuthConfig,
    auth_factory: (
        Callable[..., Any]
        | None
    ) = None,
):
    """
    Create the authenticated raw schwab-py client.

    This is the only boundary that invokes the
    OAuth client factory.

    The function itself does not make account,
    quote, or trading API requests.
    """
    validate_auth_config(
        config
    )

    token_path = Path(
        config.token_path
    )

    # schwab-py owns the token file itself.
    # We only ensure its parent directory exists.
    token_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if auth_factory is None:
        auth_factory = (
            _load_easy_client()
        )

    client = auth_factory(
        api_key=
            config.app_key,

        app_secret=
            config.app_secret,

        callback_url=
            config.callback_url,

        token_path=
            str(
                token_path
            ),

        asyncio=
            False,

        enforce_enums=
            True,

        max_token_age=
            float(
                config.max_token_age_seconds
            ),

        interactive=
            config.interactive,
    )

    if client is None:
        raise RuntimeError(
            "Schwab authentication factory "
            "returned no client."
        )

    return client


def create_read_only_schwab_client(
    config: SchwabAuthConfig,
    auth_factory: (
        Callable[..., Any]
        | None
    ) = None,
) -> SchwabReadOnlyClient:
    """
    Authenticate, then immediately place the raw
    Schwab client behind our read-only boundary.

    Application code should normally use this
    function rather than retaining the raw client.

    The returned wrapper exposes account and quote
    retrieval only. It exposes no order methods.
    """
    raw_client = (
        create_authenticated_schwab_client(
            config=
                config,

            auth_factory=
                auth_factory,
        )
    )

    return SchwabReadOnlyClient(
        raw_client
    )