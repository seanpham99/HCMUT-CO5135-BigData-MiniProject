from unittest.mock import patch

import pytest

from dags.etl_modules import request_governor


@pytest.mark.unit
class TestGovernedCallRetryClassification:
    @patch("dags.etl_modules.request_governor._acquire_source_slot")
    @patch("dags.etl_modules.request_governor.time.sleep")
    def test_retries_timeout_error_until_success(self, mock_sleep, _mock_acquire):
        attempts = {"count": 0}

        def flaky_request():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise TimeoutError("read timed out")
            return {"ok": True}

        result = request_governor.governed_call(
            "vci",
            flaky_request,
            retry_profile="conservative",
            operation="POST https://trading.vietcap.com.vn/data-mt/graphql",
        )

        assert result == {"ok": True}
        assert attempts["count"] == 3
        assert mock_sleep.call_count >= 2
