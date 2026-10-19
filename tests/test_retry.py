"""Retry exhaustion, validation, and backoff regressions."""

from unittest.mock import AsyncMock, patch

import pytest

from nexuseval.retry import retry_with_exponential_backoff


@pytest.mark.asyncio
async def test_backoff_and_retry_count():
    request = AsyncMock(side_effect=[ValueError("one"), ValueError("two"), "OK"])
    wrapped = retry_with_exponential_backoff(max_retries=2, initial_delay=0.5)(request)
    with patch("nexuseval.retry.asyncio.sleep", new_callable=AsyncMock) as sleep:
        assert await wrapped() == "OK"
    assert request.await_count == 3
    assert [call.args[0] for call in sleep.await_args_list] == [0.5, 1.0]


@pytest.mark.asyncio
async def test_zero_retries_calls_once_and_reraises():
    request = AsyncMock(side_effect=ValueError("failure"))
    with pytest.raises(ValueError, match="failure"):
        await retry_with_exponential_backoff(max_retries=0)(request)()
    assert request.await_count == 1


@pytest.mark.parametrize("kwargs", [{"max_retries": -1}, {"max_retries": True}, {"initial_delay": -1}])
def test_invalid_retry_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        retry_with_exponential_backoff(**kwargs)
