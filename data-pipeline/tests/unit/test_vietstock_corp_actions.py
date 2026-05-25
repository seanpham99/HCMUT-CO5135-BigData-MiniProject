from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from dags.etl_modules import vietstock_corp_actions as provider


@pytest.mark.unit
def test_extract_token_from_unquoted_input():
    html = (
        "<form id=__CHART_AjaxAntiForgeryForm>"
        "<input name=__RequestVerificationToken type=hidden value=token123>"
        "</form>"
    )
    assert provider._extract_token(html) == "token123"


@pytest.mark.unit
def test_parse_rate_to_percentage_supports_ratio_and_numeric():
    assert provider._parse_rate_to_percentage("5") == 5.0
    assert provider._parse_rate_to_percentage("100:5") == 5.0
    assert provider._parse_rate_to_percentage("20:7") == 35.0


@pytest.mark.unit
def test_build_dividends_frame_maps_cash_and_stock_channels():
    actions = pd.DataFrame(
        [
            {
                "channel_id": 13,
                "channel_name": "Cash dividend",
                "rate": "5",
                "gdkhq_date": date(2026, 4, 5),
                "ndkcc_date": None,
            },
            {
                "channel_id": 15,
                "channel_name": "Stock dividend",
                "rate": "10:3",
                "gdkhq_date": date(2026, 5, 5),
                "ndkcc_date": None,
            },
        ]
    )
    df = provider._build_dividends_frame(actions)

    assert len(df) == 2
    assert df.loc[0, "cash_dividend_percentage"] == 5.0
    assert df.loc[0, "stock_dividend_percentage"] == 0.0
    assert df.loc[1, "cash_dividend_percentage"] == 0.0
    assert df.loc[1, "stock_dividend_percentage"] == 30.0


@pytest.mark.unit
@patch("dags.etl_modules.vietstock_corp_actions.requests.Session")
def test_fetch_actions_normalizes_event_rows(mock_session_cls):
    session = MagicMock()
    mock_session_cls.return_value = session

    page_response = MagicMock()
    page_response.status_code = 200
    page_response.text = (
        "<input name=__RequestVerificationToken type=hidden value=token123>"
    )
    page_response.raise_for_status.return_value = None
    session.get.return_value = page_response

    event_type_response = MagicMock()
    event_type_response.status_code = 200
    event_type_response.text = (
        '[[{"EventTypeID":1,"Name":"Dividend","NameEn":"Dividend"}],'
        '[{"ChannelID":13,"Name":"Cash dividend","NameEn":"Cash dividend"}]]'
    )
    event_type_response.raise_for_status.return_value = None

    event_rows_payload = (
        '[[{"EventID":1,"Code":"VCI","CompanyName":"Vietcap","CatID":1,'
        '"GDKHQDate":"\\/Date(1775494800000)\\/","NDKCCDate":"\\/Date(1775581200000)\\/",'
        '"Time":"\\/Date(1778173200000)\\/","Note":"Cash payout 5%",'
        '"Title":"VCI dividend notice","Content":"<p>Dividend content</p>",'
        '"FileUrl":"https://example.com/file.pdf","RateTypeID":1,"Rate":"5",'
        '"VolumePublishing":null,"Row":1}],[[1]]]'
    )
    event_rows_response = MagicMock()
    event_rows_response.status_code = 200
    event_rows_response.text = event_rows_payload
    event_rows_response.raise_for_status.return_value = None

    session.post.side_effect = [
        event_type_response,  # /data/eventtypebyid
        event_rows_response,  # /data/eventstypedata page=1
    ]

    df = provider.fetch_vietstock_corporate_actions(
        "VCI",
        from_date="2020-01-01",
        to_date="2026-04-08",
        channel_ids=(13,),
        page_size=100,
    )

    assert len(df) == 1
    assert df.loc[0, "event_id"] == 1
    assert df.loc[0, "code"] == "VCI"
    assert str(df.loc[0, "gdkhq_date"]) == "2026-04-06"
    assert str(df.loc[0, "ndkcc_date"]) == "2026-04-07"
    assert str(df.loc[0, "event_date"]) == "2026-05-07"
    assert df.loc[0, "channel_name"] == "Cash dividend"


@pytest.mark.unit
@patch("dags.etl_modules.vietstock_corp_actions.requests.Session")
def test_fetch_actions_enriches_article_details_when_article_id_exists(mock_session_cls):
    session = MagicMock()
    mock_session_cls.return_value = session

    page_response = MagicMock()
    page_response.status_code = 200
    page_response.text = (
        "<input name=__RequestVerificationToken type=hidden value=token123>"
    )
    page_response.raise_for_status.return_value = None
    session.get.return_value = page_response

    event_type_response = MagicMock()
    event_type_response.status_code = 200
    event_type_response.text = (
        '[[{"EventTypeID":1,"Name":"Dividend","NameEn":"Dividend"}],'
        '[{"ChannelID":13,"Name":"Cash dividend","NameEn":"Cash dividend"}]]'
    )
    event_type_response.raise_for_status.return_value = None

    event_rows_response = MagicMock()
    event_rows_response.status_code = 200
    event_rows_response.text = (
        '[[{"EventID":1,"ArticleID":987,"Code":"VCI","CompanyName":"Vietcap","CatID":1,'
        '"GDKHQDate":"\\/Date(1775494800000)\\/","NDKCCDate":"\\/Date(1775581200000)\\/",'
        '"Time":"\\/Date(1778173200000)\\/","Title":"VCI dividend notice",'
        '"RateTypeID":1,"Rate":"5","Row":1}],[[1]]]'
    )
    event_rows_response.raise_for_status.return_value = None

    article_response = MagicMock()
    article_response.status_code = 200
    article_response.text = (
        '{"Title":"Article title","Head":"Article head","Content":"<p>Article body</p>",'
        '"PublishTime":"2026-04-02","TimeString":"02/04/2026"}'
    )
    article_response.raise_for_status.return_value = None

    session.post.side_effect = [
        event_type_response,  # /data/eventtypebyid
        event_rows_response,  # /data/eventstypedata
        article_response,  # /data/GetArticle
    ]

    df = provider.fetch_vietstock_corporate_actions(
        "VCI",
        from_date="2020-01-01",
        to_date="2026-04-08",
        channel_ids=(13,),
    )

    assert len(df) == 1
    assert df.loc[0, "article_id"] == 987
    assert df.loc[0, "article_title"] == "Article title"
    assert df.loc[0, "article_head"] == "Article head"
    assert df.loc[0, "article_content"] == "Article body"


@pytest.mark.unit
@patch("dags.etl_modules.vietstock_corp_actions.governed_call")
@patch("dags.etl_modules.vietstock_corp_actions.requests.Session")
def test_fetch_actions_uses_governed_request_wrapper(mock_session_cls, mock_governed_call):
    session = MagicMock()
    mock_session_cls.return_value = session

    page_response = MagicMock()
    page_response.text = "<input name=__RequestVerificationToken type=hidden value=token123>"

    event_type_response = MagicMock()
    event_type_response.text = (
        '[[{"EventTypeID":1,"Name":"Dividend","NameEn":"Dividend"}],'
        '[{"ChannelID":13,"Name":"Cash dividend","NameEn":"Cash dividend"}]]'
    )

    event_rows_response = MagicMock()
    event_rows_response.text = (
        '[[{"EventID":1,"Code":"VCI","CompanyName":"Vietcap","Row":1}],[[1]]]'
    )

    session.get.return_value = page_response
    session.post.side_effect = [event_type_response, event_rows_response]
    mock_governed_call.side_effect = lambda source, request_fn, **kwargs: request_fn()

    df = provider.fetch_vietstock_corporate_actions(
        "VCI",
        from_date="2020-01-01",
        to_date="2026-04-08",
        channel_ids=(13,),
    )

    assert len(df) == 1
    assert mock_governed_call.call_count >= 3
    assert all(call.args[0] == "vietstock" for call in mock_governed_call.call_args_list)
