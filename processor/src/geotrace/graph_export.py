"""Build-time converter: OSMnx GraphML -> compact binary road graph for iOS.

Parsing a 25 MB XML GraphML takes seconds and hundreds of megabytes of peak
memory. Neither is acceptable at the start of every trip on a phone, so the
graph is converted once, offline, into a flat little-endian binary that the
Swift runtime memory-maps and uses directly.

Layout (all little-endian, sections in this order, each 8-byte aligned):

    header      96 bytes, see HEADER_STRUCT
    nodeLat     f64 * nodeCount
    nodeLon     f64 * nodeCount
    edges       32 bytes * edgeCount, see EDGE_STRUCT
    pointLat    f64 * pointCount
    pointLon    f64 * pointCount
    adjOffsets  u32 * (nodeCount + 1)      CSR: outgoing edges per node
    adjEdges    u32 * edgeCount
    cellOffsets u32 * (gridCols * gridRows + 1)   CSR: edges per grid cell
    cellEdges   u32 * (last cellOffset)

Coordinates stay in WGS84. The runtime projects them into the trip's own local
frame at load time, which is what keeps the Swift and Python geometry identical:
both start from exactly the same latitudes and longitudes.

The road class is stored as an index into ROAD_CLASSES so the Swift side can
look up the same route prior and default speed limit without carrying strings.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from geotrace.road_graph import DEFAULT_SPEED_KMH, _first, _parse_speed, load_graph

MAGIC = b"GTRGRAPH"
FORMAT_VERSION = 1

HEADER_STRUCT = struct.Struct("<8sIIIIIIIIdddddd")
"""magic, version, flags, nodeCount, edgeCount, pointCount, gridCols, gridRows,
reserved, minLat, minLon, maxLat, maxLon, cellLat, cellLon"""

EDGE_STRUCT = struct.Struct("<IIIIffiBBH")
"""uNode, vNode, pointOffset, pointCount, lengthM, speedLimitMS, reverseEdge,
oneway, roadClass, pad"""

ROAD_CLASSES = (
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service", "road", "unknown",
)
ROAD_CLASS_INDEX = {name: index for index, name in enumerate(ROAD_CLASSES)}

DEFAULT_CELL_SIZE_M = 250.0
"""Grid cell edge. Small enough that a nearest-edge query touches a handful of
cells, large enough that the index stays a few hundred kilobytes."""


class GraphExportError(RuntimeError):
    pass


def _road_class_index(highway: Any) -> int:
    name = str(_first(highway) or "unknown")
    return ROAD_CLASS_INDEX.get(name, ROAD_CLASS_INDEX["unknown"])


def _pad_to_eight(blob: bytearray) -> None:
    while len(blob) % 8:
        blob.append(0)


def export_graph(
    graph: nx.MultiDiGraph,
    output: str | Path,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
) -> dict[str, Any]:
    """Write ``graph`` to the binary format and return its metadata."""
    nodes = [n for n, d in graph.nodes(data=True) if "x" in d and "y" in d]
    if not nodes:
        raise GraphExportError("graph has no nodes carrying x/y coordinates")
    node_index = {node: i for i, node in enumerate(nodes)}
    node_lat = [float(graph.nodes[n]["y"]) for n in nodes]
    node_lon = [float(graph.nodes[n]["x"]) for n in nodes]

    min_lat, max_lat = min(node_lat), max(node_lat)
    min_lon, max_lon = min(node_lon), max(node_lon)

    edge_records: list[dict[str, Any]] = []
    point_lat: list[float] = []
    point_lon: list[float] = []

    for u, v, _key, data in graph.edges(keys=True, data=True):
        if u not in node_index or v not in node_index:
            continue
        geometry = data.get("geometry")
        if geometry is not None and hasattr(geometry, "coords"):
            coords = [(float(lat), float(lon)) for lon, lat in geometry.coords]
        else:
            coords = [
                (node_lat[node_index[u]], node_lon[node_index[u]]),
                (node_lat[node_index[v]], node_lon[node_index[v]]),
            ]
        if len(coords) < 2:
            continue

        offset = len(point_lat)
        for lat, lon in coords:
            point_lat.append(lat)
            point_lon.append(lon)
            min_lat, max_lat = min(min_lat, lat), max(max_lat, lat)
            min_lon, max_lon = min(min_lon, lon), max(max_lon, lon)

        edge_records.append(
            {
                "u": node_index[u],
                "v": node_index[v],
                "offset": offset,
                "count": len(coords),
                # Length is recomputed by the runtime in the local frame; the
                # stored value is OSM's own and is only a fallback.
                "length": float(data.get("length") or 0.0),
                "speed": _parse_speed(data.get("maxspeed"), data.get("highway")),
                "oneway": 1 if bool(data.get("oneway", False)) else 0,
                "road_class": _road_class_index(data.get("highway")),
            }
        )

    if not edge_records:
        raise GraphExportError("graph produced no usable edges")

    # Reverse twin, used to forbid U-turns. Same rule as RoadNetwork._build.
    by_pair: dict[tuple[int, int], int] = {}
    for index, record in enumerate(edge_records):
        by_pair.setdefault((record["u"], record["v"]), index)
    for record in edge_records:
        record["reverse"] = by_pair.get((record["v"], record["u"]), -1)

    # CSR adjacency: outgoing edges per node, in edge-index order so the Swift
    # successor list matches RoadNetwork.out_edges exactly.
    adjacency: list[list[int]] = [[] for _ in nodes]
    for index, record in enumerate(edge_records):
        adjacency[record["u"]].append(index)
    adj_offsets: list[int] = [0]
    adj_edges: list[int] = []
    for bucket in adjacency:
        adj_edges.extend(bucket)
        adj_offsets.append(len(adj_edges))

    # Uniform lat/lon grid. Cell size is specified in metres and converted at
    # the graph's mean latitude.
    mean_lat = 0.5 * (min_lat + max_lat)
    metres_per_deg_lat = 111_132.0
    metres_per_deg_lon = max(1.0, 111_320.0 * math.cos(math.radians(mean_lat)))
    cell_lat = cell_size_m / metres_per_deg_lat
    cell_lon = cell_size_m / metres_per_deg_lon
    grid_rows = max(1, int(math.ceil((max_lat - min_lat) / cell_lat)) + 1)
    grid_cols = max(1, int(math.ceil((max_lon - min_lon) / cell_lon)) + 1)

    buckets: list[list[int]] = [[] for _ in range(grid_rows * grid_cols)]
    for index, record in enumerate(edge_records):
        seen: set[int] = set()
        start, count = record["offset"], record["count"]
        for i in range(start, start + count):
            row = min(grid_rows - 1, max(0, int((point_lat[i] - min_lat) / cell_lat)))
            col = min(grid_cols - 1, max(0, int((point_lon[i] - min_lon) / cell_lon)))
            # Also register the segment's midpoint cells so a long straight edge
            # is not missing from the cells it passes through.
            cells = {row * grid_cols + col}
            if i + 1 < start + count:
                mid_lat = 0.5 * (point_lat[i] + point_lat[i + 1])
                mid_lon = 0.5 * (point_lon[i] + point_lon[i + 1])
                mrow = min(grid_rows - 1, max(0, int((mid_lat - min_lat) / cell_lat)))
                mcol = min(grid_cols - 1, max(0, int((mid_lon - min_lon) / cell_lon)))
                cells.add(mrow * grid_cols + mcol)
            for cell in cells:
                if cell not in seen:
                    seen.add(cell)
                    buckets[cell].append(index)

    cell_offsets: list[int] = [0]
    cell_edges: list[int] = []
    for bucket in buckets:
        cell_edges.extend(bucket)
        cell_offsets.append(len(cell_edges))

    blob = bytearray()
    blob += HEADER_STRUCT.pack(
        MAGIC, FORMAT_VERSION, 0,
        len(nodes), len(edge_records), len(point_lat),
        grid_cols, grid_rows, 0,
        min_lat, min_lon, max_lat, max_lon, cell_lat, cell_lon,
    )
    _pad_to_eight(blob)
    blob += struct.pack(f"<{len(node_lat)}d", *node_lat)
    blob += struct.pack(f"<{len(node_lon)}d", *node_lon)
    for record in edge_records:
        blob += EDGE_STRUCT.pack(
            record["u"], record["v"], record["offset"], record["count"],
            record["length"], record["speed"], record["reverse"],
            record["oneway"], record["road_class"], 0,
        )
    _pad_to_eight(blob)
    blob += struct.pack(f"<{len(point_lat)}d", *point_lat)
    blob += struct.pack(f"<{len(point_lon)}d", *point_lon)
    blob += struct.pack(f"<{len(adj_offsets)}I", *adj_offsets)
    blob += struct.pack(f"<{len(adj_edges)}I", *adj_edges)
    _pad_to_eight(blob)
    blob += struct.pack(f"<{len(cell_offsets)}I", *cell_offsets)
    blob += struct.pack(f"<{len(cell_edges)}I", *cell_edges)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(blob))

    metadata = {
        "format": "GTRGRAPH",
        "format_version": FORMAT_VERSION,
        "nodes": len(nodes),
        "edges": len(edge_records),
        "polyline_points": len(point_lat),
        "bounds": {
            "min_latitude": min_lat, "min_longitude": min_lon,
            "max_latitude": max_lat, "max_longitude": max_lon,
        },
        "grid": {
            "cols": grid_cols, "rows": grid_rows,
            "cell_size_m": cell_size_m,
            "cell_latitude_deg": cell_lat, "cell_longitude_deg": cell_lon,
            "entries": len(cell_edges),
        },
        "road_classes": list(ROAD_CLASSES),
        "bytes": len(blob),
        "sha256": hashlib.sha256(bytes(blob)).hexdigest(),
    }
    out.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def export_graphml(
    graphml: str | Path, output: str | Path, cell_size_m: float = DEFAULT_CELL_SIZE_M
) -> dict[str, Any]:
    return export_graph(load_graph(graphml), output, cell_size_m)
