"""HTML report.

Builds a single self-contained ``report.html`` with:

  * a Folium/Leaflet map with toggleable layers -
      green  = original GPS,
      red    = corrupted GPS,
      blue   = reconstructed route,
      shaded = the 95% probability corridors,
      markers for rejected fixes and for the start/end of the outage;
  * a legend;
  * a position-error-vs-time chart (hand-drawn SVG, no plotting dependency);
  * the stop photos with their OCR.

The original GPS layer is labelled "reference route" only when the trip carries
a synthetic corruption manifest. During a real failure there is no ground truth
and the report says so instead of implying one.
"""

from __future__ import annotations

import base64
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import folium
import numpy as np

from geotrace.coordinates import LocalFrame
from geotrace.models import Trip
from geotrace.pipeline import ReconstructionResult
from geotrace.polygons import UncertaintySet

COLOR_ORIGINAL = "#1a9850"
COLOR_CORRUPTED = "#d73027"
COLOR_RECONSTRUCTED = "#2166ac"
COLOR_POLYGON = "#7b3294"
COLOR_REJECTED = "#e08214"
COLOR_PARKING = "#d01c8b"

MAX_EMBEDDED_PHOTO_BYTES = 1_800_000


@dataclass
class ReportInputs:
    trip: Trip
    result: ReconstructionResult
    metrics: dict[str, Any]
    original_latlon: Optional[np.ndarray] = None
    corrupted_latlon: Optional[np.ndarray] = None
    photo_blocks: list[dict[str, Any]] = None  # type: ignore[assignment]
    parking: Optional[dict[str, Any]] = None
    polygon_stride: int = 5


def _latlon_list(points: np.ndarray) -> list[list[float]]:
    return [[float(lat), float(lon)] for lat, lon in np.asarray(points).reshape(-1, 2)]


def build_map(inputs: ReportInputs) -> folium.Map:
    result = inputs.result
    frame = result.frame
    has_reference = bool(inputs.trip.reference_locations)

    centre = [frame.lat0, frame.lon0]
    fmap = folium.Map(
        location=centre,
        zoom_start=15,
        tiles=None,
        control_scale=True,
        # Otherwise the wheel zooms the map instead of scrolling the report.
        scrollWheelZoom=False,
    )
    folium.TileLayer(
        "OpenStreetMap", name="OpenStreetMap", control=True, max_zoom=19
    ).add_to(fmap)

    bounds: list[list[float]] = []

    # Corridors are drawn first so the route lines stay legible on top of them,
    # and thinned so the map does not turn into one solid purple smear.
    poly_group = folium.FeatureGroup(
        name=(
            f"{int(result.uncertainty[0].confidence * 100)}% position polygons"
            if result.uncertainty
            else "95% position polygons"
        ),
        show=True,
    ).add_to(fmap)
    for index, item in enumerate(result.uncertainty):
        if index % max(1, inputs.polygon_stride):
            continue
        for component in item.components:
            folium.GeoJson(
                component.to_geojson_feature(frame),
                style_function=lambda _f: {
                    "color": COLOR_POLYGON,
                    "weight": 1,
                    "fillColor": COLOR_POLYGON,
                    "fillOpacity": 0.12,
                    "opacity": 0.45,
                },
                tooltip=(
                    f"t = {item.t - inputs.trip.t0:.0f}s &middot; "
                    f"{component.component_id} &middot; p = {component.probability:.2f}"
                    + (f" &middot; {', '.join(component.street_names)}" if component.street_names else "")
                ),
            ).add_to(poly_group)


    original_name = (
        "Reference route (synthetic ground truth)"
        if has_reference
        else "Recorded GPS (no ground truth)"
    )
    if inputs.original_latlon is not None and len(inputs.original_latlon):
        pts = _latlon_list(inputs.original_latlon)
        bounds.extend(pts)
        folium.PolyLine(
            pts, color=COLOR_ORIGINAL, weight=4, opacity=0.85, name=original_name,
            tooltip=original_name,
        ).add_to(folium.FeatureGroup(name=original_name, show=True).add_to(fmap))

    if inputs.corrupted_latlon is not None and len(inputs.corrupted_latlon):
        pts = _latlon_list(inputs.corrupted_latlon)
        bounds.extend(pts)
        group = folium.FeatureGroup(name="Corrupted GPS", show=True).add_to(fmap)
        folium.PolyLine(
            pts, color=COLOR_CORRUPTED, weight=3, opacity=0.8, dash_array="6,6",
            tooltip="Corrupted GPS",
        ).add_to(group)

    track = result.primary
    if track.xy:
        pts = _latlon_list(frame.to_geo_array(track.array))
        bounds.extend(pts)
        group = folium.FeatureGroup(
            name="IMU route with stable GPS corrections",
            show=True,
        ).add_to(fmap)
        starts = track.segment_starts + [len(pts)]
        for i in range(len(starts) - 1):
            segment = pts[starts[i]:starts[i + 1]]
            if segment:
                folium.PolyLine(
                    segment, color=COLOR_RECONSTRUCTED, weight=4, opacity=0.9,
                    tooltip="IMU route; stable GPS corrects position but is never drawn as a route leg",
                ).add_to(group)

        road_end = pts[-1]
        folium.CircleMarker(road_end, radius=7, color=COLOR_RECONSTRUCTED, fill=True,
                            tooltip="IMU route endpoint").add_to(group)

    parking = result.parking_result
    if parking is not None:
        group = folium.FeatureGroup(name="ParkingTracker trajectory and endpoint", show=True).add_to(fmap)
        if parking.trajectory:
            parking_geo = _latlon_list(frame.to_geo_array(np.asarray(parking.trajectory)))
            folium.PolyLine(parking_geo, color=COLOR_PARKING, weight=4, opacity=0.9,
                            tooltip="ParkingTracker trajectory").add_to(group)
            bounds.extend(parking_geo)
        plat, plon = frame.to_geo(*parking.position)
        folium.Circle([plat, plon], parking.polygon_radius_m, color=COLOR_PARKING,
                      fill=True, fill_opacity=0.12, tooltip=f"Parking confidence: {parking.status}").add_to(group)
        folium.Marker([plat, plon], tooltip=f"ParkingTracker endpoint: {parking.status}",
                      icon=folium.Icon(color="pink", icon="flag")).add_to(group)

    rejected = result.diagnostics.get("rejected_fixes", [])
    if rejected:
        group = folium.FeatureGroup(name=f"Rejected GPS fixes ({len(rejected)})", show=True).add_to(fmap)
        for item in rejected:
            folium.CircleMarker(
                [item["latitude"], item["longitude"]],
                radius=4, color=COLOR_REJECTED, fill=True, fill_opacity=0.9,
                tooltip=(
                    f"t = {item['t']}s &middot; rejected: {', '.join(item['reasons']) or 'n/a'}"
                    f" &middot; state {item['state']}"
                ),
            ).add_to(group)

    shocks = result.diagnostics.get("imu_shocks", [])
    if shocks:
        group = folium.FeatureGroup(name=f"IMU mount disturbances ({len(shocks)})", show=True).add_to(fmap)
        for item in shocks:
            folium.Marker(
                [item["latitude"], item["longitude"]],
                tooltip=(
                    f"IMU disturbance: t = {item['start_s']:.1f}–{item['end_s']:.1f}s"
                    f" &middot; peak acceleration {item['peak_accel_ms2']:.1f} m/s²"
                    f" &middot; peak rotation {item['peak_gyro_rads']:.2f} rad/s"
                ),
                icon=folium.Icon(color="purple", icon="warning-sign"),
            ).add_to(group)

    if result.outage_windows:
        group = folium.FeatureGroup(name="Outage start / end", show=True).add_to(fmap)
        estimate_times = np.array(track.times) if track.times else np.zeros(0)
        estimate_geo = frame.to_geo_array(track.array) if track.xy else np.zeros((0, 2))
        for window in result.outage_windows:
            for label, t_rel in (("outage start", window["start_s"]), ("outage end", window["end_s"])):
                if estimate_times.size == 0:
                    continue
                i = int(np.argmin(np.abs(estimate_times - (inputs.trip.t0 + t_rel))))
                folium.Marker(
                    [float(estimate_geo[i][0]), float(estimate_geo[i][1])],
                    tooltip=f"{label}: t = {t_rel:.0f}s",
                    icon=folium.Icon(color="black" if "start" in label else "gray", icon="info-sign"),
                ).add_to(group)

    for block in inputs.photo_blocks or []:
        if block.get("latitude") is None:
            continue
        folium.Marker(
            [block["latitude"], block["longitude"]],
            tooltip=f"Photo {block['name']} (assumed position, not a verified address)",
            icon=folium.Icon(color="purple", icon="camera", prefix="glyphicon"),
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    if bounds:
        arr = np.array(bounds)
        fmap.fit_bounds([[arr[:, 0].min(), arr[:, 1].min()], [arr[:, 0].max(), arr[:, 1].max()]])
    return fmap


def error_chart_svg(
    series: dict[str, Any],
    outage_windows: Sequence[dict[str, float]],
    width: int = 900,
    height: int = 260,
) -> str:
    """Position error vs. time, drawn directly as SVG.

    Hand-rolled rather than pulled from a plotting library: it is a single line
    and two shaded bands, and it keeps the dependency list to what the
    specification asks for.
    """
    times = series.get("t") or []
    errors = series.get("error_m") or []
    if not times or not errors:
        return (
            '<p class="muted">No position-error chart: this trip has no reference '
            "track, so the error is undefined.</p>"
        )

    pad_l, pad_r, pad_t, pad_b = 58, 16, 14, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    t_max = max(times) or 1.0
    e_max = max(max(errors), 1.0) * 1.12

    def px(t: float) -> float:
        return pad_l + plot_w * (t / t_max)

    def py(e: float) -> float:
        return pad_t + plot_h * (1.0 - e / e_max)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Position error over time">',
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        'fill="#fbfbfd" stroke="#d8dae0"/>',
    ]

    for window in outage_windows:
        x0, x1 = px(window["start_s"]), px(min(window["end_s"], t_max))
        if x1 > x0:
            parts.append(
                f'<rect x="{x0:.1f}" y="{pad_t}" width="{x1 - x0:.1f}" height="{plot_h}" '
                'fill="#d73027" fill-opacity="0.10"/>'
            )
            parts.append(
                f'<text x="{x0 + 4:.1f}" y="{pad_t + 14}" font-size="11" fill="#a33">'
                "GPS not trusted</text>"
            )

    for i in range(5):
        value = e_max * i / 4
        y = py(value)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            'stroke="#e6e8ec"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" '
            f'fill="#666">{value:.0f}</text>'
        )
    for i in range(6):
        t = t_max * i / 5
        x = px(t)
        parts.append(
            f'<text x="{x:.1f}" y="{height - 12}" font-size="11" text-anchor="middle" '
            f'fill="#666">{t:.0f}s</text>'
        )

    points = " ".join(f"{px(t):.1f},{py(e):.1f}" for t, e in zip(times, errors))
    parts.append(
        f'<polyline points="{points}" fill="none" stroke="{COLOR_RECONSTRUCTED}" '
        'stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{pad_l - 44}" y="{pad_t + plot_h / 2:.1f}" font-size="11" fill="#666" '
        f'transform="rotate(-90 {pad_l - 44} {pad_t + plot_h / 2:.1f})" '
        'text-anchor="middle">error, m</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _photo_block_html(block: dict[str, Any]) -> str:
    lines = block.get("ocr_lines") or []
    ocr_html = (
        "".join(
            f'<li><span class="ocr-text">{html.escape(str(line["text"]))}</span>'
            f'<span class="ocr-conf">{float(line["confidence"]):.2f}</span></li>'
            for line in lines
        )
        or '<li class="muted">no text recognised</li>'
    )
    image_html = (
        f'<img src="{block["data_uri"]}" alt="stop photo {html.escape(block["name"])}"/>'
        if block.get("data_uri")
        else f'<div class="muted">image not embedded ({html.escape(block.get("path", ""))})</div>'
    )
    extracted = block.get("extracted") or {}
    chips = "".join(
        f'<span class="chip">{html.escape(k)}: {html.escape(", ".join(map(str, v)))}</span>'
        for k, v in extracted.items()
        if v
    )
    return f"""
    <div class="photo">
      {image_html}
      <div class="photo-meta">
        <h4>{html.escape(block["name"])}</h4>
        <div class="chips">{chips or '<span class="muted">nothing structured extracted</span>'}</div>
        <ul class="ocr">{ocr_html}</ul>
        <p class="muted">Assumed position only. A single photograph does not
        identify an address without a reference image database.</p>
      </div>
    </div>"""


def collect_photo_blocks(trip: Trip, diagnostics: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare the photo panels, embedding small images as data URIs."""
    blocks: list[dict[str, Any]] = []
    for photo, diag in zip(trip.photos, diagnostics):
        path = Path(photo.image_path) if photo.image_path else None
        data_uri = None
        if path and path.exists() and path.stat().st_size <= MAX_EMBEDDED_PHOTO_BYTES:
            suffix = path.suffix.lower().lstrip(".") or "jpeg"
            mime = "image/png" if suffix == "png" else "image/jpeg"
            data_uri = (
                f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
            )
        ocr = diag.get("ocr", {})
        blocks.append(
            {
                "name": path.name if path else "photo",
                "path": str(path) if path else "",
                "data_uri": data_uri,
                "ocr_lines": [
                    {"text": o.text, "confidence": o.confidence} for o in photo.ocr
                ],
                "extracted": {
                    "street": ocr.get("street_names", []),
                    "house": ocr.get("house_numbers", []),
                    "parking zone": ocr.get("parking_zones", []),
                },
                "latitude": photo.assumed_latitude,
                "longitude": photo.assumed_longitude,
            }
        )
    return blocks


def _metric(label: str, value: Any, unit: str = "") -> str:
    if value is None:
        shown = "n/a"
    elif isinstance(value, float):
        shown = f"{value:,.1f}{unit}"
    else:
        shown = f"{value}{unit}"
    return f'<div class="metric"><div class="metric-value">{shown}</div><div class="metric-label">{html.escape(label)}</div></div>'


def build_report(inputs: ReportInputs, output: str | Path) -> Path:
    """Render report.html."""
    trip = inputs.trip
    result = inputs.result
    metrics = inputs.metrics
    has_reference = bool(trip.reference_locations)

    fmap = build_map(inputs)
    map_html = fmap.get_root().render()

    error = metrics.get("position_error", {}) or {}
    polygons = metrics.get("polygons", {}) or {}
    branches = metrics.get("branches", {}) or {}
    gates = metrics.get("gps_gates", {}) or {}
    recovery = metrics.get("trust_recovery", {}) or {}

    chart = error_chart_svg(error.get("series", {}), result.outage_windows)

    faults_note = ""
    if trip.faults:
        applied = trip.faults.get("faults", [])
        rows = "".join(
            f"<tr><td>{html.escape(str(f.get('kind')))}</td>"
            f"<td>{f.get('start_s')}</td>"
            f"<td>{f.get('duration_s') if f.get('duration_s') is not None else '&infin;'}</td>"
            f"<td><code>{html.escape(json.dumps(f.get('params', {}), ensure_ascii=False))}</code></td>"
            f"<td>{f.get('affected_samples')}</td></tr>"
            for f in applied
        )
        faults_note = f"""
        <h3>Injected faults</h3>
        <div class="tablewrap"><table><thead><tr><th>kind</th><th>start, s</th><th>duration, s</th>
        <th>parameters</th><th>fixes affected</th></tr></thead><tbody>{rows}</tbody></table></div>"""

    truth_banner = (
        '<div class="banner ok">This trip was corrupted synthetically, so the green '
        "line is a genuine reference track and the error numbers below are real "
        "measurements against it.</div>"
        if has_reference
        else '<div class="banner warn">This is a real recording. There is no ground '
        "truth for the period when GPS failed, so position error is undefined and "
        "is reported as n/a. The blue line is a probabilistic estimate.</div>"
    )

    parking_html = ""
    dual = result.parking_result
    if dual is not None:
        parking_html = f"""
        <h3>ParkingTracker terminal result</h3>
        <div class="tablewrap"><table><thead><tr><th>source</th><th>status</th><th>confidence</th>
        <th>terminal GPS</th><th>polygon, m</th><th>reverse</th><th>reason</th></tr></thead><tbody><tr>
        <td><code>parking_tracker</code></td><td>{html.escape(dual.status)}</td><td>{dual.confidence:.2f}</td>
        <td>{dual.terminal_cluster_count}</td><td>{dual.polygon_radius_m:.1f}</td>
        <td>{"yes" if dual.has_reverse_motion else "no"}</td><td>{html.escape(dual.reason)}</td>
        </tr></tbody></table></div>"""
    if inputs.parking:
        p = inputs.parking
        parking_html += f"""
        <h3>Parking zone</h3>
        <p><strong>{html.escape(str(p.get('selected_zone')))}</strong> &mdash;
        p = {p.get('probability', 0):.3f}, runner-up {p.get('second_probability', 0):.3f},
        decision <code>{html.escape(str(p.get('decision')))}</code></p>
        <p class="muted">Zone source: {html.escape(str(p.get('zone_source', 'unspecified')))}</p>"""

    photos_html = "".join(_photo_block_html(b) for b in (inputs.photo_blocks or []))
    if photos_html:
        photos_html = f"<h3>Stop photos and OCR</h3><div class=\"photos\">{photos_html}</div>"

    outage_rows = "".join(
        f"<tr><td>{w['start_s']:.1f}</td><td>{w['end_s']:.1f}</td>"
        f"<td>{w['end_s'] - w['start_s']:.1f}</td><td>{html.escape(str(w.get('state', '')))}</td></tr>"
        for w in result.outage_windows
    ) or '<tr><td colspan="4" class="muted">no outage detected</td></tr>'

    baselines = metrics.get("baselines", {}) or {}
    baseline_rows = "".join(
        f"<tr><td><code>{html.escape(name)}</code></td>"
        f"<td>{_fmt(stats.get('mean_m'))}</td><td>{_fmt(stats.get('median_m'))}</td>"
        f"<td>{_fmt(stats.get('p95_m'))}</td><td>{_fmt(stats.get('max_m'))}</td>"
        f"<td>{_fmt(stats.get('error_at_outage_end_m'))}</td></tr>"
        for name, stats in baselines.items()
    )
    primary_row = (
        f"<tr class=\"primary\"><td><code>{html.escape(result.algorithm)}</code> (selected)</td>"
        f"<td>{_fmt(error.get('mean_m'))}</td><td>{_fmt(error.get('median_m'))}</td>"
        f"<td>{_fmt(error.get('p95_m'))}</td><td>{_fmt(error.get('max_m'))}</td>"
        f"<td>{_fmt(error.get('error_at_outage_end_m'))}</td></tr>"
    )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>GeoTrace report - {html.escape(trip.metadata.trip_id)}</title>
<style>
 :root {{ color-scheme: light; }}
 body {{ margin:0; font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
        background:#f4f5f7; color:#1d2129; }}
 .wrap {{ max-width:1180px; margin:0 auto; padding:24px 20px 64px; }}
 h1 {{ font-size:22px; margin:0 0 4px; }}
 h3 {{ font-size:16px; margin:28px 0 10px; }}
 .sub {{ color:#666; margin:0 0 18px; }}
 .card {{ background:#fff; border:1px solid #e3e5e9; border-radius:10px; padding:18px 20px; margin-bottom:18px; }}
 .banner {{ border-radius:8px; padding:11px 14px; margin-bottom:18px; font-size:13px; }}
 .banner.ok {{ background:#eaf4ec; border:1px solid #bcd9c3; color:#215c31; }}
 .banner.warn {{ background:#fdf3e3; border:1px solid #e8ceA0; color:#7a4f10; }}
 .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
 .metric {{ background:#fafbfc; border:1px solid #e6e8ec; border-radius:8px; padding:12px 14px; }}
 .metric-value {{ font-size:20px; font-weight:600; }}
 .metric-label {{ font-size:12px; color:#666; margin-top:2px; }}
 .tablewrap {{ overflow-x:auto; -webkit-overflow-scrolling:touch; }}
 table {{ border-collapse:collapse; width:100%; font-size:13px; min-width:520px; }}
 th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid #eceef1; }}
 th {{ color:#555; font-weight:600; background:#fafbfc; }}
 tr.primary td {{ background:#eef4fb; font-weight:600; }}
 code {{ background:#f1f2f4; border-radius:4px; padding:1px 5px; font-size:12px; }}
 .legend {{ display:flex; flex-wrap:wrap; gap:16px; margin:12px 0 4px; font-size:13px; }}
 .legend span {{ display:flex; align-items:center; gap:7px; }}
 .swatch {{ width:22px; height:4px; border-radius:2px; display:inline-block; }}
 .dot {{ width:11px; height:11px; border-radius:50%; display:inline-block; }}
 iframe {{ width:100%; height:640px; border:1px solid #e3e5e9; border-radius:8px; background:#fff; }}
 .muted {{ color:#8a8f98; }}
 .photos {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
 .photo {{ border:1px solid #e6e8ec; border-radius:8px; overflow:hidden; background:#fafbfc; }}
 .photo img {{ width:100%; display:block; max-height:280px; object-fit:cover; }}
 .photo-meta {{ padding:12px 14px; }}
 .photo-meta h4 {{ margin:0 0 8px; font-size:14px; }}
 .chips {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; }}
 .chip {{ background:#e9edf3; border-radius:12px; padding:2px 9px; font-size:12px; }}
 ul.ocr {{ list-style:none; margin:0; padding:0; font-size:13px; }}
 ul.ocr li {{ display:flex; justify-content:space-between; gap:12px; padding:2px 0; border-bottom:1px dotted #e6e8ec; }}
 .ocr-conf {{ color:#8a8f98; font-variant-numeric:tabular-nums; }}
</style></head><body><div class="wrap">
<h1>GeoTrace route recovery report</h1>
<p class="sub">Trip <code>{html.escape(trip.metadata.trip_id)}</code> &middot;
 {trip.duration_s:.0f} s &middot; {len(trip.locations)} GPS fixes &middot;
 {len(trip.motions)} motion samples &middot; algorithm
 <code>{html.escape(result.algorithm)}</code> &middot; seed {result.diagnostics.get('seed')}</p>
{truth_banner}

<div class="card">
  <div class="legend">
    <span><i class="swatch" style="background:{COLOR_ORIGINAL}"></i>
      {"Reference route (synthetic ground truth)" if has_reference else "Recorded GPS"}</span>
    <span><i class="swatch" style="background:{COLOR_CORRUPTED}"></i> Corrupted GPS</span>
    <span><i class="swatch" style="background:{COLOR_RECONSTRUCTED}"></i> Reconstructed route (estimate)</span>
    <span><i class="swatch" style="background:{COLOR_POLYGON};height:11px;opacity:.35"></i> 95% position polygons</span>
    <span><i class="dot" style="background:{COLOR_REJECTED}"></i> Rejected GPS fix</span>
  </div>
  <p class="muted" style="margin:6px 0 12px">Layers can be switched on and off in the control at the top right of the map.</p>
  <iframe srcdoc="{html.escape(map_html, quote=True)}" title="route map"></iframe>
</div>

<div class="card">
  <h3 style="margin-top:0">Headline numbers</h3>
  <div class="metrics">
    {_metric("mean error", error.get("mean_m"), " m")}
    {_metric("median error", error.get("median_m"), " m")}
    {_metric("95th percentile", error.get("p95_m"), " m")}
    {_metric("max error", error.get("max_m"), " m")}
    {_metric("error at outage end", error.get("error_at_outage_end_m"), " m")}
    {_metric("polygon coverage", polygons.get("coverage_95"))}
    {_metric("mean polygon area", polygons.get("mean_area_m2"), " m2")}
    {_metric("top-1 branch accuracy", branches.get("top1_accuracy"))}
    {_metric("top-3 branch recall", branches.get("top3_recall"))}
    {_metric("max branches held", branches.get("max_branch_count"))}
    {_metric("rejected good fixes", gates.get("rejected_good_fraction"))}
    {_metric("accepted false fixes", gates.get("accepted_false_fraction"))}
    {_metric("trust recovery", recovery.get("mean_recovery_s"), " s")}
  </div>
</div>

<div class="card">
  <h3 style="margin-top:0">Position error over time</h3>
  {chart}
</div>

<div class="card">
  <h3 style="margin-top:0">Algorithm comparison</h3>
  <div class="tablewrap"><table><thead><tr><th>algorithm</th><th>mean, m</th><th>median, m</th>
  <th>p95, m</th><th>max, m</th><th>at outage end, m</th></tr></thead>
  <tbody>{primary_row}{baseline_rows}</tbody></table></div>

  <h3>Detected GPS outages</h3>
  <div class="tablewrap"><table><thead><tr><th>start, s</th><th>end, s</th><th>duration, s</th><th>final state</th></tr></thead>
  <tbody>{outage_rows}</tbody></table></div>
  {faults_note}
  {parking_html}
</div>

<div class="card">{photos_html or '<p class="muted">No stop photos in this trip.</p>'}</div>

<p class="muted">The reconstructed route is a probabilistic estimate produced from
inertial data constrained by an OpenStreetMap road graph. It is not a measurement
of where the car was. Where the belief covers more than one road, the report shows
several polygons with their own probabilities rather than a single averaged point.</p>
</div></body></html>"""

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return '<span class="muted">n/a</span>'
    if isinstance(value, float):
        return f"{value:,.1f}"
    return str(value)


def track_geojson(
    result: ReconstructionResult, name: str, properties: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    track = result.tracks[name]
    return {"type": "FeatureCollection", "features": [track.to_geojson(result.frame, properties)]}


def locations_geojson(
    locations: Iterable[Any], frame: LocalFrame, name: str, properties: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """LineString of a raw GPS track, plus one point feature per fix."""
    samples = [s for s in locations if s.is_usable]
    coords = [[float(s.longitude), float(s.latitude)] for s in samples]
    props = {"name": name, "point_count": len(coords)}
    if properties:
        props.update(properties)
    features: list[dict[str, Any]] = [
        {"type": "Feature", "properties": props,
         "geometry": {"type": "LineString", "coordinates": coords}}
    ]
    for sample in samples:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "monotonic_time": sample.monotonic_time,
                    "horizontal_accuracy": sample.horizontal_accuracy,
                    "speed": sample.speed,
                    "course": sample.course,
                    "synthetic": sample.synthetic,
                },
                "geometry": {"type": "Point", "coordinates": [sample.longitude, sample.latitude]},
            }
        )
    return {"type": "FeatureCollection", "features": features}
