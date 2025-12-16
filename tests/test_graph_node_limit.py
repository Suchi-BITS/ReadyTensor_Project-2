from agents.supervisor import Graph, Node



class DummyNode(Node):
    def __init__(self, name):
        super().__init__(name)

    def run(self, state):
        count = state.get("count", 0)
        return {"count": count + 1}


def test_graph_aborts_when_node_limit_exceeded():
    """
    Graph execution should abort when node execution
    exceeds the MAX_NODES limit.
    """
    graph = Graph()

    # Add more nodes than MAX_NODES (10)
    for i in range(15):
        graph.add_node(DummyNode(f"node_{i}"))

    initial_state = {"count": 0}
    result = graph.run(initial_state)

    assert result.get("error") is True
    assert "node limit exceeded" in result.get(
        "error_message", ""
    ).lower()
