from collections.abc import Mapping, Sequence


def report_failed_symbols(
    stage: str, failed_symbols: Sequence[Mapping[str, object]]
) -> None:
    if not failed_symbols:
        return
    print(f"{stage}: failed symbols ({len(failed_symbols)}):")
    for item in failed_symbols:
        symbol = str(item.get("symbol", "unknown"))
        error = str(item.get("error", "unknown error"))
        print(f"- {symbol}: {error}")
