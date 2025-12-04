from numba import njit


@njit
def edge(node1: int, node2: int) -> tuple[int, int]:
    """Return a sorted edge tuple."""
    return (node1, node2) if node1 < node2 else (node2, node1)


def edges_from_adj(adj: dict[int, set[int]]) -> set[tuple[int, int]]:
    """Convert adjacency list to a set of edges.

    Parameters:
    -----------
    adj : dict[int, set[int]]
        Adjacency list representation of a graph.

    Returns:
    --------
    set[tuple[int, int]]
        Set of edges represented as tuples (node1, node2).
    """

    edges = set()
    for node, neighbors in adj.items():
        for neighbor in neighbors:
            if node < neighbor:
                edges.add((node, neighbor))

    return edges


@njit
def detect_t1_events(
    edges1: set[tuple[int, int]], edges2: set[tuple[int, int]]
) -> list[tuple[int, int, int, int]]:
    """
    Detect T1 events between two sets of edges.

    Parameters:
    -----------
    edges1 : set[tuple[int, int]]
        Set of edges at time point 1.
    edges2 : set[tuple[int, int]]
        Set of edges at time point 2.
    """
    lost_edges = edges1 - edges2
    gained_edges = edges2 - edges1
    common_edges = edges1 & edges2

    t1_events = []

    # The order of A and C / B and D does not matter, since the labels are
    # interchangeable. It is sufficient to check one ordering.
    for A, C in gained_edges:
        for B, D in lost_edges:
            if (
                edge(A, B) in common_edges
                and edge(A, D) in common_edges
                and edge(B, C) in common_edges
                and edge(C, D) in common_edges
            ):
                t1_events.append((A, B, C, D))

    return t1_events
