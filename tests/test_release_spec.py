from scripts.build_data_release import RECORDS, describe_column


def test_release_includes_continuous_control_inputs() -> None:
    destinations = {record.destination for record in RECORDS}

    assert "data/core/resource_matches.parquet" in destinations
    assert "data/sensitivity/coal_price_quantiles.csv" in destinations


def test_resource_percentile_columns_receive_controlled_definitions() -> None:
    for percentile in (0, 1, 50, 90, 99):
        description, source = describe_column(f"resource_p{percentile}")

        assert f"P{percentile}" in description
        assert source == "controlled definition"
