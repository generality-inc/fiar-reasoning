"""
minimax.py
==========
Two-pass pipeline for evaluating a reasoning tree:

  Pass 1 — annotate_tree_with_boards()
      Walk the tree recursively, replaying moves from the root board state
      to reconstruct the board state at every node. Stores the board in-place
      as node['board_self']. This is done ONCE and cached; board states are
      independent of the model weights.

  Pass 2 — minimax_values()
      Given weight vector w and scaling constant C, evaluate every leaf
      with compute_value(), then propagate up via negamax:
          node_value = max(child_values_from_this_node's_perspective)
                     = max(-child_node_values)   [because child is from
                                                   the opponent's perspective]

Negamax correctness
--------------------
  board_self at depth d is always encoded from the perspective of whoever
  is to move at depth d.  compute_value() therefore always returns a number
  where higher = better for that player.

  When propagating from child to parent: the parent's SELF is the child's OPP,
  so the parent sees the child's value NEGATED.

  At each internal node:
      node['value'] = max( -child['value']  for child in children )

  At the root, node['value'] for each root child = the value of that move
  FROM THE ROOT PLAYER'S PERSPECTIVE. This is exactly what you need for
  choice modelling.

Terminal detection
-------------------
  If the board at a node already contains a four-in-a-row for SELF (+1),
  that means the move that *led to this node* was a winning move — but
  the board is now from the NEXT player's perspective, so it looks like
  OPP won. Handle this by checking before flipping, i.e. right after
  apply_move we check if SELF just won (four +1 pieces in a row on the
  pre-flip board).

  Concretely in annotate_tree_with_boards, we store the board AFTER
  apply_move (i.e. from the next player's perspective). Then in
  minimax_values we check: does the *incoming* player (the one who just
  moved) have four in a row? That player is represented as OPP (= -1)
  on the stored board (since we already flipped). So:
      terminal win for the player who just moved
          <-> four-in-a-row of -1 on node['board_self']
          <-> value = -T  from the current player's perspective
                      (current player = SELF = the one to move next = the loser)

  Without explicit terminal check: four-in-a-row boards are valued via w_four * phi_four
"""

import numpy as np
from preprocess import apply_move, ROWS, COLS, EMPTY
from features import compute_value

# Terminal value is now a fitted parameter passed into minimax_values().


# ---------------------------------------------------------------------------
# Pass 1: annotate tree nodes with board states
# ---------------------------------------------------------------------------

def annotate_tree_with_boards(tree: list, root_board: np.ndarray) -> None:
    """
    Walk the parsed tree and attach 'board_self' to every node IN-PLACE.

    The board stored at a node is the state AFTER that node's move has been
    played, encoded from the perspective of the NEXT player to move.
    (i.e. apply_move already flips perspective.)

    Parameters
    ----------
    tree : list of node dicts
        Each node is {'move': (r,c), 'children': [...]}
        Output: adds 'board_self' key to every node.
    root_board : np.ndarray, shape (4,9)
        Board BEFORE any move in the tree, in self/opp encoding for the
        player whose turn it is at the root.

    Notes
    -----
    - We skip nodes whose move leads to an already-occupied square
      (can happen if the LLM tree is slightly inconsistent). Such nodes
      get board_self=None and are treated as leaves with undefined value
      (excluded from minimax).
    - Modifies tree in-place; safe to call multiple times (idempotent).
    """
    def _annotate(node: dict, parent_board: np.ndarray) -> None:
        r, c = node['move']

        # Guard: skip if square already occupied (malformed tree node)
        if parent_board[r, c] != EMPTY:
            node['board_self'] = None   # sentinel for invalid node
            node['children'] = []       # don't recurse further
            return

        # Apply move: place SELF piece, flip to next player's perspective
        child_board = apply_move(parent_board, r, c)
        node['board_self'] = child_board

        # Recurse into children — they see child_board as their parent board
        for child in node['children']:
            _annotate(child, child_board)

    for root_node in tree:
        _annotate(root_node, root_board)


# ---------------------------------------------------------------------------
# Pass 2: minimax (negamax) value computation
# ---------------------------------------------------------------------------

def minimax_values(tree: list,
                   weights: np.ndarray,
                   C: float = 1.0,
                   gamma: float = 1.0) -> dict:
    """
    Compute the minimax value for each root action in the tree.

    Assumes annotate_tree_with_boards() has already been called.

    Parameters
    ----------
    tree : list of node dicts  (annotated with 'board_self')
    weights : np.ndarray, shape (5,)
        [w_centre, w_connected2, w_unconnected2, w_three, w_four]
    C : float
        Active scaling constant.
    gamma : float in [0, 1]
        Discount factor for backed-up values at internal nodes.
        At each internal node the value is a blend:
            (1 - gamma) * heuristic(board) + gamma * max(-child_values)
        gamma=1.0 → standard minimax (current default behaviour)
        gamma=0.0 → every node evaluated by its own heuristic (myopic)

    Returns
    -------
    root_action_values : dict  {(row, col): float}
        For each root-level move, its minimax value from the root player's
        perspective. Higher = better for the player to move at the root.

    Note on terminals
    -----------------
    There is no explicit terminal check. A four-in-a-row position is handled
    naturally by compute_value via w_four * phi_four. The optimizer will push
    w_four large enough that winning positions dominate the softmax, which is
    the correct approach when there is no feature dropout or pruning (as in
    the paper). With dropout/pruning, a separate terminal override T would be
    needed because w_four could be zeroed out before T is applied; without
    them, w_four and T are redundant and only w_four is needed.
    """
    def _node_value(node: dict) -> float:
        """
        Recursively compute the value of a node from the perspective of
        the player whose turn it is AT THIS NODE.
        """
        board = node.get('board_self')

        # Invalid/skipped node — return neutral value
        if board is None:
            return 0.0

        # Leaf node: evaluate heuristically.
        if not node['children']:
            return compute_value(board, weights, C)

        # Internal node: blend heuristic with backed-up minimax value.
        # When gamma=1 this reduces to standard negamax.
        # When gamma=0 every node is evaluated by its own heuristic (myopic).
        heuristic   = compute_value(board, weights, C)
        child_vals  = [-_node_value(child) for child in node['children']]
        minimax_val = max(child_vals)
        return (1.0 - gamma) * heuristic + gamma * minimax_val

    # _node_value returns the value from the perspective of whoever's board
    # is stored at the node (i.e. the player AFTER the move was made = opponent).
    # The root player wants the NEGATIVE of that — same negamax rule applied
    # one level above the tree.
    root_action_values = {}
    for root_node in tree:
        move = root_node['move']
        root_action_values[move] = -_node_value(root_node)

    return root_action_values


# ---------------------------------------------------------------------------
# Convenience: full pipeline for one df row
# ---------------------------------------------------------------------------

def get_root_action_values(tree: list,
                            root_board: np.ndarray,
                            weights: np.ndarray,
                            C: float = 1.0,
                            gamma: float = 1.0,
                            already_annotated: bool = False) -> dict:
    """
    Full pipeline: annotate (if needed) then compute minimax values.

    Parameters
    ----------
    tree : list of parsed node dicts
    root_board : np.ndarray  (board before any move in the tree)
    weights : np.ndarray, shape (5,)
    C : float
    already_annotated : bool
        If True, skip Pass 1 (useful when weights change but boards don't).

    Returns
    -------
    dict  {(row, col): float}
    """
    if not already_annotated:
        annotate_tree_with_boards(tree, root_board)
    return minimax_values(tree, weights, C, gamma)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from preprocess import parse_board_state, parse_tree, board_to_str

    board_state_str = "FEN: 2WBBB3/4W4/5W3/9\n\nCurrent player: Black"
    _, player, root_board = parse_board_state(board_state_str)

    weights = np.array([1.0, 2.0, 1.5, 5.0, 100.0])
    C = 2.0

    # ------------------------------------------------------------------
    # Test 1: depth-1 tree
    # ------------------------------------------------------------------
    print("=== Test 1: Depth-1 tree ===")
    print(board_to_str(root_board, player))
    raw_tree = [[0, 0], [0, 1], [1, 3]]
    tree = parse_tree(raw_tree)
    annotate_tree_with_boards(tree, root_board)
    vals = minimax_values(tree, weights, C)
    print(f"Root action values:")
    for move, v in vals.items():
        print(f"  move {move}: {v:.4f}")

    # ------------------------------------------------------------------
    # Test 2: depth-2 tree — verify negamax sign flip
    # ------------------------------------------------------------------
    print("\n=== Test 2: Depth-2 tree (verify negamax) ===")
    # (0,0) is Black's move; then (0,1) is White's response -> leaf
    raw_tree2 = [[0, 0, [0, 1]], [1, 3]]
    tree2 = parse_tree(raw_tree2)
    annotate_tree_with_boards(tree2, root_board)
    vals2 = minimax_values(tree2, weights, C)
    print(f"Root action values: {vals2}")

    # Manual verification:
    # board_after_00: flipped once (White's perspective)
    # board_after_01: flipped twice (Black's perspective)
    # _node_value(child (0,1))     = compute_value(board_after_01) = leaf_val
    # _node_value(node  (0,0))     = max(-leaf_val)  = -leaf_val    [negamax inside]
    # root_action_values[(0,0)]    = -_node_value    = leaf_val     [root negation]
    board_after_00 = apply_move(root_board, 0, 0)
    board_after_01 = apply_move(board_after_00, 0, 1)
    leaf_val = compute_value(board_after_01, weights, C)
    expected_00 = leaf_val   # two negations cancel: root is back to Black's view
    print(f"\nManual check move (0,0):")
    print(f"  leaf (0,1) value = {leaf_val:.4f}")
    print(f"  expected (0,0)   = {expected_00:.4f}  (two negations cancel)")
    print(f"  minimax returned = {vals2[(0,0)]:.4f}")
    assert abs(vals2[(0,0)] - expected_00) < 1e-9, "Negamax mismatch!"
    print("\u2713 Negamax correctly propagated")

    # ------------------------------------------------------------------
    # Test 3: winning move is valued highly via w_four
    # ------------------------------------------------------------------
    print("\n=== Test 3: Winning move valued highly via w_four ===")
    board_win = np.zeros((ROWS, COLS), dtype=np.int8)
    board_win[0, 0:3] = 1    # SELF has 3 pieces; (0,3) completes four-in-a-row
    raw_tree3 = [[0, 3], [1, 4]]
    tree3 = parse_tree(raw_tree3)
    annotate_tree_with_boards(tree3, board_win)
    vals3 = minimax_values(tree3, weights, C)
    print(f"Values (winning move at (0,3)): {vals3}")
    # After (0,3) is played, the board (from OPP's perspective) has SELF's
    # four-in-a-row as OPP's pieces — but compute_value returns a value from
    # the current player's view, so the winning board returns a very negative
    # value (OPP just lost), and the root negates it to a large positive.
    assert vals3[(0, 3)] > vals3[(1, 4)], "Winning move should be valued higher"
    print(f"\u2713 Winning move (0,3) has higher value than neutral move (1,4)")

    # ------------------------------------------------------------------
    # Test 4: opponent's winning response makes move less attractive
    # ------------------------------------------------------------------
    print("\n=== Test 4: Opponent winning response propagates correctly ===")
    board_opp = np.zeros((ROWS, COLS), dtype=np.int8)
    board_opp[0, 0:3] = -1   # OPP has 3-in-a-row; can win at (0,3)
    # Self plays (1,4), then OPP wins at (0,3)
    raw_tree4 = [[1, 4, ['0,3']]]
    tree4 = parse_tree(raw_tree4)
    annotate_tree_with_boards(tree4, board_opp)
    vals4 = minimax_values(tree4, weights, C)
    print(f"Value of (1,4) when OPP wins after: {vals4}")
    # After (1,4), OPP plays (0,3) which gives OPP four-in-a-row.
    # That board (from the next player's view) has a very large negative
    # value — propagated back, (1,4) should be strongly negative.
    assert vals4[(1, 4)] < 0, "Move that lets OPP win should have negative value"
    print(f"\u2713 Move leading to opponent win is correctly negative")

    # ------------------------------------------------------------------
    # Test 5: idempotency
    # ------------------------------------------------------------------
    print("\n=== Test 5: Idempotency of annotation ===")
    annotate_tree_with_boards(tree2, root_board)
    vals2b = minimax_values(tree2, weights, C)
    for move in vals2:
        assert abs(vals2[move] - vals2b[move]) < 1e-9
    print("\u2713 Double annotation gives identical results")

    print("\n=== All tests passed ===")
