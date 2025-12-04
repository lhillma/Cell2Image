from cell2image.t1_detection import detect_t1_events, edges_from_adj


def test_edges_from_adj():
    adj = {
        1: {2, 4},
        2: {1, 3, 4},
        3: {2, 4},
        4: {1, 2, 3},
    }
    edges = edges_from_adj(adj)
    expected_edges = {(1, 2), (1, 4), (2, 3), (2, 4), (3, 4)}
    assert edges == expected_edges


def test_simple_t1():
    adj1 = {
        1: {2, 4},
        2: {1, 3, 4},
        3: {2, 4},
        4: {1, 2, 3},
    }
    adj2 = {
        1: {2, 3, 4},
        2: {1, 3},
        3: {1, 2, 4},
        4: {1, 3},
    }
    t1_events = detect_t1_events(adj1, adj2)
    assert len(t1_events) == 1
    assert t1_events[0] == (1, 2, 3, 4)


def test_simple_t1_with_extra():
    adj1 = {1: {2, 4}, 2: {1, 3, 4}, 3: {2, 4}, 4: {1, 2, 3}, 5: {1, 3}}
    adj2 = {
        1: {2, 3, 4},
        2: {1, 3},
        3: {1, 2, 4},
        4: {1, 3},
        5: {1, 2},
    }
    t1_events = detect_t1_events(adj1, adj2)
    assert len(t1_events) == 1
    assert t1_events[0] == (1, 2, 3, 4)
