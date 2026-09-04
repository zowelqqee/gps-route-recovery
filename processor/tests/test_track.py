from geotrace.coordinates import LocalFrame
from geotrace.pipeline import Track


def test_reanchor_break_exports_two_route_segments() -> None:
    """A correction must not be rendered as a cross-city route leg."""
    track = Track("imu")
    track.add(0.0, (0.0, 0.0))
    track.add(1.0, (10.0, 0.0))
    track.break_line()
    track.add(2.0, (1_000.0, 0.0))

    feature = track.to_geojson(LocalFrame(59.0, 30.0))

    assert feature["geometry"]["type"] == "MultiLineString"
    assert [len(segment) for segment in feature["geometry"]["coordinates"]] == [2, 1]
    assert feature["properties"]["segment_count"] == 2
