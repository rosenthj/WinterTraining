"""Game-level validation of PGN input, ahead of feature extraction.

This replaces the separate filtering pass that ChessDataGen/filter_pgn.py used to run
over the shard PGNs. Folding the checks in here means each shard is read once, which
matters now that conversion runs per shard on the cluster rather than over merged
450k-game files.

Two distinct concerns:

  * **Structural** -- the movetext or the result cannot be trusted at all, so the game
    is dropped outright. These are not merely a data-quality question: an unfinished
    "*" result raises out of ``utils.string_to_result_class`` and an illegal move trips
    the assertion in ``game.ItGame.make_move``, either of which aborts the whole
    conversion run rather than skipping one game. Data generation runs on the
    preemptable ``standby`` QOS, so a shard killed mid-write and left with a partial
    final game is a live scenario, not a hypothetical one. Structural checks therefore
    always apply, regardless of ``config.drop_abnormal``.

  * **Result trust** -- the movetext is fine but the recorded result was not produced
    by chess, i.e. a time forfeit. See ``forfeit_final_board``.
"""

import chess

import count

# Terminations that mean the result was decided over the board, or by an adjudication
# rule we asked for. fastchess stamps [Termination "normal"] on ordinary games too, so
# whenever PGN_COMMENTS=true the tag is present and authoritative. Anything outside this
# set -- "time forfeit", "illegal move", "stalled connection", "abandoned" -- means the
# result was decided by something other than the position.
GOOD_TERMINATIONS = frozenset({"normal", "adjudication"})

VALID_RESULTS = frozenset({"1-0", "0-1", "1/2-1/2"})


def forfeit_final_board(g, board):
    """Return ``board`` if the game's decisive result was not produced by chess, else None.

    A time forfeit is recorded by writing the forfeiting side's last move and then
    scoring the game against them, so the game stops at a position that is neither mate,
    stalemate, insufficient material, a repetition, nor a 50-move draw. The recorded
    result is then unrelated to the position -- measured across desk_v312 the forfeiting
    side was materially *ahead* about as often as behind, and 116 games in the first 6000
    ended with the result exactly inverted relative to the Syzygy verdict.

    Detection uses the [Termination] tag where it is decisive and falls back to the
    positional test otherwise, so one rule covers both the tagged PGNs and the untagged
    desk_v303..v317 generated with PGN_COMMENTS=false.

    The positional fallback is exact rather than heuristic only while RESIGN_ADJ is
    empty in datagen/datagen_config.sh: with no resign adjudication configured there is
    no way to end a game decisively except by mate, so a decisive non-terminal position
    can only be a forfeit. Enabling RESIGN_ADJ would make legitimate adjudicated wins
    look identical, which is why an explicit "adjudication" tag short-circuits the test.
    """
    if g.headers.get("Result") == "1/2-1/2":
        return None
    termination = g.headers.get("Termination")
    if termination is not None:
        termination = termination.lower()
        if termination not in GOOD_TERMINATIONS:
            return board
        if termination == "adjudication":
            return None
    # Repetition is tested at twofold, the permissive reading, so that a genuine
    # repetition is never mistaken for a forfeit.
    if (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material()
            or board.is_repetition(2) or board.halfmove_clock >= 100):
        return None
    return board


def check_game(g):
    """Validate one game. Returns ``(reason, forfeit_board)``.

    ``reason`` names a failed structural check and means the game must be dropped, or is
    None when the game is usable. ``forfeit_board`` is the final position when the result
    is untrustworthy and must be overridden from the tablebase, else None; it is only
    meaningful when ``reason`` is None.

    The final board is computed once here because both halves need it.
    """
    result = g.headers.get("Result", "*")
    if result not in VALID_RESULTS:
        # python-chess defaults a missing Result to "*", which is what a shard truncated
        # before its final game's header block produces.
        return "unfinished", None
    if g.errors:
        # Moves python-chess could not parse were skipped, so the mainline no longer
        # matches the game that was played.
        return "unparsable", None

    board = g.end().board()

    # A terminal final position states the result outright, and it outranks both the
    # Result header and the Termination tag: the moves replayed cleanly, so the mate or
    # draw is real whatever the runner recorded.
    if board.is_checkmate():
        true_result = "0-1" if board.turn == chess.WHITE else "1-0"
    elif board.is_stalemate() or board.is_insufficient_material():
        true_result = "1/2-1/2"
    else:
        true_result = None

    if true_result is not None:
        if result != true_result:
            # This is the forfeit the positional test cannot see. A side that flags on
            # the very move delivering mate leaves a genuine checkmate behind, so the
            # position looks entirely normal and only the result is wrong -- desk_v312
            # has four such games in its first 2993, every one of them a White mate
            # scored 0-1. Correcting is strictly better than dropping: the result is
            # certain and the game is otherwise a full-length, normally played one.
            g.headers["Result"] = true_result
            count.relabelled_terminal += 1
        return None, None

    return None, forfeit_final_board(g, board)
