from cell2image.t1_detection import detect_t1_events


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
