"""
Value backup over an extracted search tree (paper Eqs. 2-3).

Each node stores 'board_self', the board after its move, encoded from the
perspective of the player to move next. Values are therefore computed in
negamax form: a node's value is from the next mover's view, and its parent
sees it negated. With discount gamma, an internal node's value is
    (1 - gamma) * h(s) + gamma * max_children(-V(child)),
so gamma = 1 is full minimax and gamma = 0 evaluates each node by h alone.
"""

from .board import EMPTY, apply_move


def annotate_tree_with_boards(tree: list, root_board) -> None:
    """Attach 'board_self' to every node in place. Moves onto occupied squares get None."""
    def annotate(node, parent_board):
        r, c = node['move']
        if parent_board[r, c] != EMPTY:
            node['board_self'] = None
            node['children'] = []
            return
        node['board_self'] = apply_move(parent_board, r, c)
        for child in node['children']:
            annotate(child, node['board_self'])

    for node in tree:
        annotate(node, root_board)


def root_action_values(tree: list, value_fn, weights, C=1.0, gamma=1.0) -> dict:
    """Return {move: value} for each depth-1 move, from the root player's view."""
    def node_value(node):
        board = node.get('board_self')
        if board is None:
            return 0.0
        h = value_fn(board, weights, C)
        if not node['children']:
            return h
        backed_up = max(-node_value(child) for child in node['children'])
        return (1.0 - gamma) * h + gamma * backed_up

    return {node['move']: -node_value(node) for node in tree}
