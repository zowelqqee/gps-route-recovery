"""Resolving raw OSM `type=restriction` relations into graph edge ids."""

from __future__ import annotations

import json

import networkx as nx
import pytest

from geotrace.coordinates import LocalFrame
from geotrace.road_graph import RoadNetwork, extract_turn_restrictions

from conftest import edge_named

ORIGIN_LAT = 59.9311
ORIGIN_LON = 30.3609


def _relation(
    kind: str,
    from_way: int,
    to_way: int,
    via: list[dict],
    tags_extra: dict | None = None,
    relation_id: int = 1,
) -> dict:
    tags = {"type": "restriction", "restriction": kind}
    tags.update(tags_extra or {})
    return {
        "type": "relation",
        "id": relation_id,
        "tags": tags,
        "members": [
            {"type": "way", "ref": from_way, "role": "from"},
            *via,
            {"type": "way", "ref": to_way, "role": "to"},
        ],
    }


@pytest.fixture
def junction_graph() -> nx.MultiDiGraph:
    """A one real junction (B) with a through way split around it.

    Way 100 ("Main") is one OSM way that OSMnx split into two graph edges at
    the junction node B: A->B and B->C. Way 200 ("Side") leaves B towards D.
    This is the case that makes `from`/`to` resolution ambiguous unless the
    via node is used to disambiguate which of a way's several edges is meant.
    """
    g = nx.MultiDiGraph()
    g.add_node("A", x=30.000, y=59.900)
    g.add_node("B", x=30.001, y=59.900)
    g.add_node("C", x=30.002, y=59.900)
    g.add_node("D", x=30.001, y=59.901)
    g.add_edge("A", "B", 0, osmid=100)
    g.add_edge("B", "C", 0, osmid=100)
    g.add_edge("B", "D", 0, osmid=200)
    g.add_edge("D", "B", 0, osmid=200)
    return g


def test_no_turn_resolves_via_a_node(junction_graph: nx.MultiDiGraph) -> None:
    relation = _relation("no_left_turn", from_way=100, to_way=200, via=[{"type": "node", "ref": "B", "role": "via"}])
    resolved = extract_turn_restrictions(junction_graph, [relation])
    assert len(resolved) == 1
    r = resolved[0]
    assert r["kind"] == "no"
    assert tuple(r["from_edge"]) == ("A", "B", 0)
    assert tuple(r["to_edge"]) == ("B", "D", 0)
    assert r["via_edges"] == []


def test_only_turn_resolves_via_a_node(junction_graph: nx.MultiDiGraph) -> None:
    relation = _relation("only_straight_on", from_way=100, to_way=100, via=[{"type": "node", "ref": "B", "role": "via"}])
    resolved = extract_turn_restrictions(junction_graph, [relation])
    assert len(resolved) == 1
    assert resolved[0]["kind"] == "only"
    assert tuple(resolved[0]["from_edge"]) == ("A", "B", 0)
    assert tuple(resolved[0]["to_edge"]) == ("B", "C", 0)


def test_the_via_node_disambiguates_a_split_way(junction_graph: nx.MultiDiGraph) -> None:
    """Way 100 has two edges; only the one ending at the via node may be `from`."""
    relation = _relation("no_u_turn", from_way=100, to_way=100, via=[{"type": "node", "ref": "B", "role": "via"}])
    resolved = extract_turn_restrictions(junction_graph, [relation])
    assert tuple(resolved[0]["from_edge"]) == ("A", "B", 0)
    assert tuple(resolved[0]["to_edge"]) == ("B", "C", 0)


def test_except_motorcar_is_dropped(junction_graph: nx.MultiDiGraph) -> None:
    """A restriction that explicitly does not apply to cars is not our problem."""
    relation = _relation(
        "no_left_turn", from_way=100, to_way=200,
        via=[{"type": "node", "ref": "B", "role": "via"}],
        tags_extra={"except": "motorcar"},
    )
    assert extract_turn_restrictions(junction_graph, [relation]) == []


def test_except_psv_still_applies_to_a_car(junction_graph: nx.MultiDiGraph) -> None:
    relation = _relation(
        "no_left_turn", from_way=100, to_way=200,
        via=[{"type": "node", "ref": "B", "role": "via"}],
        tags_extra={"except": "psv"},
    )
    assert len(extract_turn_restrictions(junction_graph, [relation])) == 1


def test_a_relation_for_a_way_outside_the_graph_is_dropped(junction_graph: nx.MultiDiGraph) -> None:
    """Common after clipping: the `to` way was cut out of the local graph."""
    relation = _relation("no_left_turn", from_way=100, to_way=999999, via=[{"type": "node", "ref": "B", "role": "via"}])
    assert extract_turn_restrictions(junction_graph, [relation]) == []


def test_a_non_restriction_relation_is_ignored(junction_graph: nx.MultiDiGraph) -> None:
    relation = {
        "type": "relation", "id": 2, "tags": {"type": "route"},
        "members": [{"type": "way", "ref": 100, "role": ""}],
    }
    assert extract_turn_restrictions(junction_graph, [relation]) == []


@pytest.fixture
def chained_graph() -> nx.MultiDiGraph:
    """A -> B -> C -> D, three separate ways meeting at real junctions B, C."""
    g = nx.MultiDiGraph()
    g.add_node("A", x=30.000, y=59.900)
    g.add_node("B", x=30.001, y=59.900)
    g.add_node("C", x=30.002, y=59.900)
    g.add_node("D", x=30.003, y=59.900)
    g.add_edge("A", "B", 0, osmid=100)
    g.add_edge("B", "C", 0, osmid=200)
    g.add_edge("C", "D", 0, osmid=300)
    return g


def test_a_chained_via_way_restriction_resolves_the_whole_path(chained_graph: nx.MultiDiGraph) -> None:
    relation = _relation(
        "no_left_turn", from_way=100, to_way=300,
        via=[{"type": "way", "ref": 200, "role": "via"}],
    )
    resolved = extract_turn_restrictions(chained_graph, [relation])
    assert len(resolved) == 1
    r = resolved[0]
    assert tuple(r["from_edge"]) == ("A", "B", 0)
    assert [tuple(e) for e in r["via_edges"]] == [("B", "C", 0)]
    assert tuple(r["to_edge"]) == ("C", "D", 0)


def test_restrictions_survive_a_json_round_trip_into_a_road_network(
    junction_graph: nx.MultiDiGraph,
) -> None:
    """What `load_graph` does: deserialise the cached JSON back onto the graph
    before `RoadNetwork` reads `graph.graph['turn_restrictions']`."""
    relation = _relation("no_left_turn", from_way=100, to_way=200, via=[{"type": "node", "ref": "B", "role": "via"}])
    restrictions = extract_turn_restrictions(junction_graph, [relation])
    junction_graph.graph["turn_restrictions_json"] = json.dumps(restrictions)

    # Round-trip through JSON exactly as the GraphML cache does.
    junction_graph.graph["turn_restrictions"] = json.loads(
        junction_graph.graph["turn_restrictions_json"]
    )
    network = RoadNetwork(junction_graph, LocalFrame(ORIGIN_LAT, ORIGIN_LON))
    assert len(network.turn_restrictions) == 1
    from_index = network.edge_index[("A", "B", 0)]
    to_index = network.edge_index[("B", "D", 0)]
    allowed = network.allowed_successors(from_index, history=(from_index,))
    assert to_index not in allowed


# ------------------------------------------- distance-budgeted reachability


def test_reachable_within_distance_is_bounded_by_the_budget(fork_network) -> None:
    """The stem is 500 m; from its start a 100 m budget cannot leave it."""
    stem = edge_named(fork_network, "Stem", (0.0, 0.0))
    reach = fork_network.reachable_within_distance(stem, s=0.0, budget_m=100.0)
    assert set(reach) == {stem}
    assert reach[stem] == (0.0, stem)


def test_reachable_within_distance_counts_only_the_edge_still_ahead(fork_network) -> None:
    """Standing 490 m along a 500 m edge, the junction is 10 m away, not 500."""
    stem = edge_named(fork_network, "Stem", (0.0, 0.0))
    branch_a = edge_named(fork_network, "Branch A", (500.0, 0.0))
    branch_b = edge_named(fork_network, "Branch B", (500.0, 0.0))
    reach = fork_network.reachable_within_distance(stem, s=490.0, budget_m=50.0)
    assert {branch_a, branch_b} <= set(reach)


def test_reachable_within_distance_labels_the_first_hop_at_a_fork(fork_network) -> None:
    """Every edge is tagged with which way out of the junction reaches it,
    so a caller can sort particle weight into branches in one pass."""
    stem = edge_named(fork_network, "Stem", (0.0, 0.0))
    branch_a = edge_named(fork_network, "Branch A", (500.0, 0.0))
    branch_b = edge_named(fork_network, "Branch B", (500.0, 0.0))
    reach = fork_network.reachable_within_distance(stem, s=499.0, budget_m=5000.0)
    assert reach[branch_a][1] == branch_a
    assert reach[branch_b][1] == branch_b
    for edge, (_distance, first_hop) in reach.items():
        if edge != stem:
            assert first_hop in (branch_a, branch_b)


def test_reachable_within_distance_honours_a_one_way_street(oneway_network) -> None:
    """A displayed route must not be walked the wrong way down a one-way."""
    one_way = edge_named(oneway_network, "One way east", (300.0, 0.0))
    reach = oneway_network.reachable_within_distance(one_way, s=0.0, budget_m=5000.0)
    stem = edge_named(oneway_network, "Stem", (0.0, 0.0))
    assert stem not in reach, "the one-way street cannot lead back into the stem"


def test_reachable_within_distance_always_includes_its_own_edge(fork_network) -> None:
    stem = edge_named(fork_network, "Stem", (0.0, 0.0))
    reach = fork_network.reachable_within_distance(stem, s=0.0, budget_m=0.0)
    assert set(reach) == {stem}


def test_route_between_returns_the_connected_road_geometry(fork_network) -> None:
    """A post-outage route is topology, not an interpolated straight line."""
    route = fork_network.route_between((100.0, 0.0), (1000.0, 500.0))
    assert route is not None
    assert route.length_m == pytest.approx(400.0 + 2**0.5 * 500.0, abs=2.0)
    assert route.coords[0] == pytest.approx((100.0, 0.0), abs=1.0)
    assert route.coords[-1] == pytest.approx((1000.0, 500.0), abs=1.0)


def test_route_between_never_reverses_a_one_way_street(oneway_network) -> None:
    assert oneway_network.route_between((850.0, 0.0), (350.0, 0.0)) is None
