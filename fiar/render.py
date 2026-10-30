"""
Graphviz rendering of an extracted search tree (Figure 1D).

The root box shows the board's FEN; circles are the model's own moves and
boxes the opponent's replies, alternating with depth. Nodes from the
'trees_annotated' column may carry tags ('r,c', 'illegal') or
('r,c', 'final', 'win' | 'loss' | 'tie'), which are coloured.
"""

import re

import graphviz

FILL = {'win': '#00FF41', 'loss': '#FF073A', 'tie': '#FFD700'}


def _head(head):
    """Return (label, status) for a node head."""
    if isinstance(head, tuple):
        if len(head) >= 2 and head[1] == 'illegal':
            return str(head[0]), 'illegal'
        if len(head) >= 3 and head[1] == 'final':
            return str(head[0]), head[2]
        return str(head[0]), None
    return str(head), None


def tree_graph(trees, board_state=None, node_size=0.85, font_size=30, line_width=4) -> graphviz.Digraph:
    dot = graphviz.Digraph(
        graph_attr={'rankdir': 'TB', 'ranksep': '0.35', 'nodesep': '0.5', 'splines': 'polyline',
                    'bgcolor': '#FFFFFF', 'size': '16,10!', 'ratio': 'fill', 'margin': '0.3', 'pad': '0.3'},
        node_attr={'fontname': 'Arial', 'fixedsize': 'true', 'width': str(node_size),
                   'height': str(node_size), 'fontsize': str(font_size), 'penwidth': str(line_width)},
        edge_attr={'color': '#000000', 'penwidth': str(line_width), 'dir': 'none'},
    )
    fen = re.search(r'FEN:\s*[^\n]+', board_state or '')
    dot.node('root', fen.group(0) if fen else 'START', shape='box', fixedsize='false',
             fontname='Arial-Bold', fontsize=str(font_size + 1), width=str(node_size * 3.5))

    counter = iter(range(10**9))

    def add(node, parent, depth):
        if isinstance(node, str):
            node = [node]
        label, status = _head(node[0])
        node_id = f'n{next(counter)}'
        style = {'shape': 'circle' if depth % 2 else 'box'}
        edge = {}
        if status == 'illegal':
            style.update(style='filled,dashed', fillcolor='#666666', fontcolor='white',
                         penwidth=str(line_width * 0.75))
            edge.update(style='dashed', color='#666666')
        elif status in FILL:
            style.update(style='filled', fillcolor=FILL[status], penwidth='0',
                         fontcolor='white' if status == 'loss' else 'black')
            edge.update(color=FILL[status], penwidth=str(line_width * 1.5))
        dot.node(node_id, label, **style)
        dot.edge(parent, node_id, **edge)
        for child in node[1:]:
            add(child, node_id, depth + 1)

    for node in trees:
        add(node, 'root', 1)
    return dot


def render_tree(trees, board_state=None, filename=None, **style) -> graphviz.Digraph:
    """Build the tree graph and, if filename is given, write filename.pdf."""
    dot = tree_graph(trees, board_state, **style)
    if filename:
        dot.render(filename, format='pdf', cleanup=True)
    return dot
