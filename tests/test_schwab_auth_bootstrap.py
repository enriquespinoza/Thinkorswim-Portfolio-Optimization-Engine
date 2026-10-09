from pathlib import Path

import pytest

from src.integrations.schwab.auth_bootstrap import (
    DEFAULT_MAX_TOKEN_AGE_SECONDS,
    SchwabAuthConfig,
    create_authenticated_schwab_client,
    create_market_data_schwab_client,
    create_read_only_schwab_client,
    load_schwab_auth_config,
    validate_auth_config,
)

from src.integrations.schwab.market_data_client import (
    SchwabMarketDataClient,
)
from src.integrations.schwab.read_only_client import (
    SchwabReadOnlyClient,
)


def sample_environment():
    return {
        "SCHWAB_APP_KEY":
            "TEST-APP-KEY",

        "SCHWAB_APP_SECRET":
            "TEST-APP-SECRET",

        "SCHWAB_CALLBACK_URL":
            "https://127.0.0.1:8182",

        "SCHWAB_TOKEN_PATH":
            "tokens/schwab_token.json",
    }


def sample_config(
    tmp_path: Path,
) -> SchwabAuthConfig:
    return SchwabAuthConfig(
        app_key=
            "TEST-APP-KEY",

        app_secret=
            "TEST-APP-SECRET",

        callback_url=
            "https://127.0.0.1:8182",

        token_path=
            tmp_path
            / "tokens"
            / "schwab_token.json",

        interactive=
            True,

        max_token_age_seconds=
            DEFAULT_MAX_TOKEN_AGE_SECONDS,
    )


class FakeRawSchwabClient:
    pass


class RecordingAuthFactory:
    def __init__(
        self,
    ):
        self.calls = []

        self.client = (
            FakeRawSchwabClient()
        )

    def __call__(
        self,
        **kwargs,
    ):
        self.calls.append(
            kwargs
        )

        return self.client


def test_loads_auth_config_from_environment(
    tmp_path: Path,
):
    config = (
        load_schwab_auth_config(
            environ=
                sample_environment(),

            project_root=
                tmp_path,
        )
    )

    assert (
        config.app_key
        == "TEST-APP-KEY"
    )

    assert (
        config.app_secret
        == "TEST-APP-SECRET"
    )

    assert (
        config.callback_url
        == "https://127.0.0.1:8182"
    )

    assert (
        config.token_path
        == (
            tmp_path
            / "tokens"
            / "schwab_token.json"
        ).resolve()
    )


@pytest.mark.parametrize(
    "missing_variable",
    [
        "SCHWAB_APP_KEY",
        "SCHWAB_APP_SECRET",
        "SCHWAB_CALLBACK_URL",
        "SCHWAB_TOKEN_PATH",
    ],
)
def test_missing_required_environment_variable_rejected(
    tmp_path: Path,
    missing_variable: str,
):
    environ = (
        sample_environment()
    )

    del environ[
        missing_variable
    ]

    with pytest.raises(
        ValueError,
        match=
            missing_variable,
    ):
        load_schwab_auth_config(
            environ=
                environ,

            project_root=
                tmp_path,
        )


def test_callback_requires_https(
    tmp_path: Path,
):
    config = SchwabAuthConfig(
        app_key=
            "key",

        app_secret=
            "secret",

        callback_url=
            "http://127.0.0.1:8182",

        token_path=
            tmp_path
            / "token.json",
    )

    with pytest.raises(
        ValueError,
        match="HTTPS",
    ):
        validate_auth_config(
            config
        )


def test_token_path_requires_json(
    tmp_path: Path,
):
    config = SchwabAuthConfig(
        app_key=
            "key",

        app_secret=
            "secret",

        callback_url=
            "https://127.0.0.1:8182",

        token_path=
            tmp_path
            / "token.txt",
    )

    with pytest.raises(
        ValueError,
        match=r"\.json",
    ):
        validate_auth_config(
            config
        )


def test_relative_token_path_uses_project_root(
    tmp_path: Path,
):
    config = (
        load_schwab_auth_config(
            environ=
                sample_environment(),

            project_root=
                tmp_path,
        )
    )

    assert (
        config.token_path
        == (
            tmp_path
            / "tokens"
            / "schwab_token.json"
        ).resolve()
    )


def test_config_repr_hides_credentials(
    tmp_path: Path,
):
    config = (
        sample_config(
            tmp_path
        )
    )

    representation = repr(
        config
    )

    assert (
        "TEST-APP-KEY"
        not in representation
    )

    assert (
        "TEST-APP-SECRET"
        not in representation
    )


def test_auth_factory_receives_expected_configuration(
    tmp_path: Path,
):
    config = (
        sample_config(
            tmp_path
        )
    )

    factory = (
        RecordingAuthFactory()
    )

    result = (
        create_authenticated_schwab_client(
            config=
                config,

            auth_factory=
                factory,
        )
    )

    assert (
        result
        is factory.client
    )

    assert len(
        factory.calls
    ) == 1

    call = (
        factory.calls[0]
    )

    assert (
        call[
            "api_key"
        ]
        == "TEST-APP-KEY"
    )

    assert (
        call[
            "app_secret"
        ]
        == "TEST-APP-SECRET"
    )

    assert (
        call[
            "callback_url"
        ]
        == "https://127.0.0.1:8182"
    )

    assert (
        call[
            "token_path"
        ]
        == str(
            config.token_path
        )
    )

    assert (
        call[
            "asyncio"
        ]
        is False
    )

    assert (
        call[
            "enforce_enums"
        ]
        is True
    )

    assert (
        call[
            "interactive"
        ]
        is True
    )

    assert (
        call[
            "max_token_age"
        ]
        == DEFAULT_MAX_TOKEN_AGE_SECONDS
    )


def test_token_parent_directory_created(
    tmp_path: Path,
):
    config = (
        sample_config(
            tmp_path
        )
    )

    factory = (
        RecordingAuthFactory()
    )

    assert not (
        config.token_path.parent.exists()
    )

    create_authenticated_schwab_client(
        config=
            config,

        auth_factory=
            factory,
    )

    assert (
        config.token_path.parent.exists()
    )


def test_bootstrap_does_not_create_token_file_itself(
    tmp_path: Path,
):
    config = (
        sample_config(
            tmp_path
        )
    )

    factory = (
        RecordingAuthFactory()
    )

    create_authenticated_schwab_client(
        config=
            config,

        auth_factory=
            factory,
    )

    # Our code creates only the directory.
    # Real schwab-py owns token creation.
    assert not (
        config.token_path.exists()
    )


def test_none_auth_client_rejected(
    tmp_path: Path,
):
    config = (
        sample_config(
            tmp_path
        )
    )

    def broken_factory(
        **kwargs,
    ):
        return None

    with pytest.raises(
        RuntimeError,
        match="returned no client",
    ):
        create_authenticated_schwab_client(
            config=
                config,

            auth_factory=
                broken_factory,
        )


def test_read_only_wrapper_created(
    tmp_path: Path,
):
    config = (
        sample_config(
            tmp_path
        )
    )

    factory = (
        RecordingAuthFactory()
    )

    client = (
        create_read_only_schwab_client(
            config=
                config,

            auth_factory=
                factory,
        )
    )

    assert isinstance(
        client,
        SchwabReadOnlyClient,
    )


def test_read_only_wrapper_exposes_no_order_methods(
    tmp_path: Path,
):
    client = (
        create_read_only_schwab_client(
            config=
                sample_config(
                    tmp_path
                ),

            auth_factory=
                RecordingAuthFactory(),
        )
    )

    assert not hasattr(
        client,
        "place_order",
    )

    assert not hasattr(
        client,
        "replace_order",
    )

    assert not hasattr(
        client,
        "cancel_order",
    )

    assert not hasattr(
        client,
        "preview_order",
    )


def test_nonpositive_token_age_rejected(
    tmp_path: Path,
):
    config = SchwabAuthConfig(
        app_key=
            "key",

        app_secret=
            "secret",

        callback_url=
            "https://127.0.0.1:8182",

        token_path=
            tmp_path
            / "token.json",

        max_token_age_seconds=
            0,
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        validate_auth_config(
            config
        )


def test_market_data_wrapper_created(
    tmp_path: Path,
):
    config = (
        sample_config(
            tmp_path
        )
    )

    factory = (
        RecordingAuthFactory()
    )

    client = (
        create_market_data_schwab_client(
            config=
                config,

            auth_factory=
                factory,
        )
    )

    assert isinstance(
        client,
        SchwabMarketDataClient,
    )


def test_market_data_wrapper_exposes_no_account_or_order_methods(
    tmp_path: Path,
):
    client = (
        create_market_data_schwab_client(
            config=
                sample_config(
                    tmp_path
                ),

            auth_factory=
                RecordingAuthFactory(),
        )
    )

    forbidden = (
        "get_account",
        "get_account_numbers",
        "place_order",
        "replace_order",
        "cancel_order",
        "preview_order",
    )

    for method_name in forbidden:
        assert not hasattr(
            client,
            method_name,
        )
