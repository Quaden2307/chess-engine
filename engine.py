"""
Classical chess engine: iterative-deepening negamax + alpha-beta, quiescence,
move ordering (MVV-LVA, killers, history), transposition table, and tapered
piece-square evaluation. No neural network. Plays ~1600-1800 strength.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional

import chess
import chess.polyglot


# ---- Material values (midgame / endgame, centipawns) ----

MG_VALUES = {chess.PAWN: 82, chess.KNIGHT: 337, chess.BISHOP: 365,
             chess.ROOK: 477, chess.QUEEN: 1025, chess.KING: 0}
EG_VALUES = {chess.PAWN: 94, chess.KNIGHT: 281, chess.BISHOP: 297,
             chess.ROOK: 512, chess.QUEEN: 936, chess.KING: 0}

PHASE_WEIGHTS = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1,
                 chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0}
TOTAL_PHASE = 24


# ---- Piece-square tables (white's perspective, A1=0 ... H8=63) ----
# Classical Tomasz Michniewski-style tables, adapted.

PST_MG = {
    chess.PAWN: [
          0,   0,   0,   0,   0,   0,   0,   0,
          5,  10,  10, -20, -20,  10,  10,   5,
          5,  -5, -10,   0,   0, -10,  -5,   5,
          0,   0,   0,  20,  20,   0,   0,   0,
          5,   5,  10,  25,  25,  10,   5,   5,
         10,  10,  20,  30,  30,  20,  10,  10,
         50,  50,  50,  50,  50,  50,  50,  50,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    chess.KNIGHT: [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20,   0,   5,   5,   0, -20, -40,
        -30,   5,  10,  15,  15,  10,   5, -30,
        -30,   0,  15,  20,  20,  15,   0, -30,
        -30,   5,  15,  20,  20,  15,   5, -30,
        -30,   0,  10,  15,  15,  10,   0, -30,
        -40, -20,   0,   0,   0,   0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ],
    chess.BISHOP: [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   5,   0,   0,   0,   0,   5, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,   0,  10,  10,  10,  10,   0, -10,
        -10,   5,   5,  10,  10,   5,   5, -10,
        -10,   0,   5,  10,  10,   5,   0, -10,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    chess.ROOK: [
          0,   0,   0,   5,   5,   0,   0,   0,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
          5,  10,  10,  10,  10,  10,  10,   5,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    chess.QUEEN: [
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   5,   0,   0,   0,   0, -10,
        -10,   5,   5,   5,   5,   5,   0, -10,
          0,   0,   5,   5,   5,   5,   0,  -5,
         -5,   0,   5,   5,   5,   5,   0,  -5,
        -10,   0,   5,   5,   5,   5,   0, -10,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20,
    ],
    chess.KING: [
         20,  30,  10,   0,   0,  10,  30,  20,
         20,  20,   0,   0,   0,   0,  20,  20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
    ],
}

PST_EG = {
    chess.PAWN: [
          0,   0,   0,   0,   0,   0,   0,   0,
         10,  10,  10,  10,  10,  10,  10,  10,
         10,  10,  20,  20,  20,  20,  10,  10,
         20,  20,  30,  30,  30,  30,  20,  20,
         30,  30,  40,  40,  40,  40,  30,  30,
         50,  50,  50,  50,  50,  50,  50,  50,
         80,  80,  80,  80,  80,  80,  80,  80,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    chess.KNIGHT: PST_MG[chess.KNIGHT],
    chess.BISHOP: PST_MG[chess.BISHOP],
    chess.ROOK: PST_MG[chess.ROOK],
    chess.QUEEN: PST_MG[chess.QUEEN],
    chess.KING: [
        -50, -30, -30, -30, -30, -30, -30, -50,
        -30, -30,   0,   0,   0,   0, -30, -30,
        -30, -10,  20,  30,  30,  20, -10, -30,
        -30, -10,  30,  40,  40,  30, -10, -30,
        -30, -10,  30,  40,  40,  30, -10, -30,
        -30, -10,  20,  30,  30,  20, -10, -30,
        -30, -20, -10,   0,   0, -10, -20, -30,
        -50, -40, -30, -20, -20, -30, -40, -50,
    ],
}


def _mirror_sq(sq: int) -> int:
    return sq ^ 56  # flip rank for black-perspective PST lookup


# ---- Evaluation ----

def _phase(board: chess.Board) -> int:
    p = 0
    for piece_type, w in PHASE_WEIGHTS.items():
        if w == 0:
            continue
        p += w * (chess.popcount(board.pieces_mask(piece_type, chess.WHITE)) +
                  chess.popcount(board.pieces_mask(piece_type, chess.BLACK)))
    return min(p, TOTAL_PHASE)


def evaluate(board: chess.Board) -> int:
    """Static evaluation in centipawns from side-to-move perspective."""
    if board.is_checkmate():
        return -MATE
    if (board.is_stalemate() or board.is_insufficient_material() or
            board.is_seventyfive_moves() or board.is_fivefold_repetition()):
        return 0

    mg = 0
    eg = 0

    for sq, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        idx = sq if piece.color == chess.WHITE else _mirror_sq(sq)
        mg += sign * (MG_VALUES[piece.piece_type] + PST_MG[piece.piece_type][idx])
        eg += sign * (EG_VALUES[piece.piece_type] + PST_EG[piece.piece_type][idx])

    # Bishop pair
    if chess.popcount(board.pieces_mask(chess.BISHOP, chess.WHITE)) >= 2:
        mg += 30
        eg += 50
    if chess.popcount(board.pieces_mask(chess.BISHOP, chess.BLACK)) >= 2:
        mg -= 30
        eg -= 50

    # Pawn structure: doubled and isolated
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        pawns = board.pieces(chess.PAWN, color)
        files = [chess.square_file(sq) for sq in pawns]
        file_counts = [0] * 8
        for f in files:
            file_counts[f] += 1
        for n in file_counts:
            if n > 1:
                mg -= sign * 10 * (n - 1)
                eg -= sign * 20 * (n - 1)
        for f, n in enumerate(file_counts):
            if n == 0:
                continue
            left = file_counts[f - 1] if f > 0 else 0
            right = file_counts[f + 1] if f < 7 else 0
            if left == 0 and right == 0:
                mg -= sign * 12 * n
                eg -= sign * 20 * n

    # King safety: penalize exposed king in midgame (pawn shield)
    ph = _phase(board)
    if ph > 8:  # only matters when significant material remains
        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            king_sq = board.king(color)
            if king_sq is None:
                continue
            kf = chess.square_file(king_sq)
            kr = chess.square_rank(king_sq)
            shield = 0
            pawn_rank = kr + 1 if color == chess.WHITE else kr - 1
            if 0 <= pawn_rank <= 7:
                for df in (-1, 0, 1):
                    f = kf + df
                    if 0 <= f <= 7:
                        sq = chess.square(f, pawn_rank)
                        p = board.piece_at(sq)
                        if p and p.piece_type == chess.PAWN and p.color == color:
                            shield += 1
            mg += sign * (shield * 10 - 20)

    score = (mg * ph + eg * (TOTAL_PHASE - ph)) // TOTAL_PHASE

    # Small tempo bonus
    score += 10 if board.turn == chess.WHITE else -10

    return score if board.turn == chess.WHITE else -score


# ---- Search ----

INF = 10_000_000
MATE = 100_000  # mate score; ply-adjusted in search


class _TimeUp(Exception):
    pass


@dataclass
class _TTEntry:
    depth: int
    score: int
    flag: int  # 0=exact, 1=lower-bound, 2=upper-bound
    move: Optional[chess.Move]


class Engine:
    def __init__(self, tt_max: int = 1_000_000):
        self.tt: dict[int, _TTEntry] = {}
        self.tt_max = tt_max
        self.killers: dict[int, list[chess.Move]] = {}
        self.history: dict[tuple[int, int], int] = {}
        self.nodes = 0
        self.start = 0.0
        self.time_limit = 0.0
        self.root_scores: dict[chess.Move, int] = {}

    def _check_time(self):
        if (self.nodes & 4095) == 0:
            if time.time() - self.start > self.time_limit:
                raise _TimeUp()

    def _order_moves(self, board: chess.Board, moves, tt_move, ply: int):
        killers = self.killers.get(ply, [])
        scored = []
        for m in moves:
            if tt_move is not None and m == tt_move:
                s = 10_000_000
            elif board.is_capture(m):
                victim = board.piece_at(m.to_square)
                attacker = board.piece_at(m.from_square)
                v = MG_VALUES[victim.piece_type] if victim is not None else MG_VALUES[chess.PAWN]
                a = MG_VALUES[attacker.piece_type] if attacker is not None else 0
                s = 1_000_000 + v * 10 - a
                if m.promotion:
                    s += MG_VALUES[m.promotion]
            elif m.promotion:
                s = 900_000 + MG_VALUES[m.promotion]
            elif m in killers:
                s = 800_000 - killers.index(m)
            else:
                s = self.history.get((m.from_square, m.to_square), 0)
            scored.append((s, m))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored]

    def _store_tt(self, key: int, entry: _TTEntry):
        if len(self.tt) >= self.tt_max:
            # Cheap eviction: drop a random ~10% slice
            for k in random.sample(list(self.tt.keys()), self.tt_max // 10):
                del self.tt[k]
        self.tt[key] = entry

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> tuple[int, Optional[chess.Move]]:
        self.nodes += 1
        self._check_time()

        # Draw checks (not at root; root handled by caller)
        if ply > 0:
            if (board.is_repetition(2) or board.is_fivefold_repetition() or
                    board.is_seventyfive_moves() or board.is_insufficient_material()):
                return 0, None
            if board.halfmove_clock >= 100:
                return 0, None

        in_check = board.is_check()

        # Check extension
        if in_check and depth < 32:
            depth += 1

        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply), None

        # TT lookup
        key = chess.polyglot.zobrist_hash(board)
        entry = self.tt.get(key)
        tt_move: Optional[chess.Move] = None
        if entry is not None:
            tt_move = entry.move
            if entry.depth >= depth and ply > 0:
                if entry.flag == 0:
                    return entry.score, entry.move
                if entry.flag == 1 and entry.score >= beta:
                    return entry.score, entry.move
                if entry.flag == 2 and entry.score <= alpha:
                    return entry.score, entry.move

        moves = list(board.legal_moves)
        if not moves:
            return (-MATE + ply, None) if in_check else (0, None)

        moves = self._order_moves(board, moves, tt_move, ply)
        best_score = -INF
        best_move: Optional[chess.Move] = None
        original_alpha = alpha

        for move in moves:
            board.push(move)
            score, _ = self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            score = -score
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                # Beta cutoff: record killer & history for quiet moves
                if not board.is_capture(move) and move.promotion is None:
                    klist = self.killers.setdefault(ply, [])
                    if move not in klist:
                        klist.insert(0, move)
                        if len(klist) > 2:
                            klist.pop()
                    h_key = (move.from_square, move.to_square)
                    self.history[h_key] = self.history.get(h_key, 0) + depth * depth
                break

        flag = 0
        if best_score <= original_alpha:
            flag = 2
        elif best_score >= beta:
            flag = 1
        self._store_tt(key, _TTEntry(depth, best_score, flag, best_move))
        return best_score, best_move

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        self._check_time()

        if board.is_checkmate():
            return -MATE + ply
        if (board.is_stalemate() or board.is_insufficient_material() or
                board.is_fivefold_repetition() or board.is_seventyfive_moves()):
            return 0

        stand = evaluate(board)
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand

        # Generate captures + queen promotions
        moves = []
        for m in board.legal_moves:
            if board.is_capture(m) or m.promotion == chess.QUEEN:
                moves.append(m)
        moves = self._order_moves(board, moves, None, ply)

        for m in moves:
            board.push(m)
            score = -self._quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _search_root(self, board: chess.Board, depth: int) -> dict[chess.Move, int]:
        """Search every root move with a full window; returns move->score."""
        moves = list(board.legal_moves)
        if self.root_scores:
            moves.sort(key=lambda m: -self.root_scores.get(m, -INF))
        else:
            moves = self._order_moves(board, moves, None, 0)

        scores: dict[chess.Move, int] = {}
        for move in moves:
            board.push(move)
            score, _ = self._negamax(board, depth - 1, -INF, INF, 1)
            score = -score
            board.pop()
            scores[move] = score
        return scores

    def search(self, board: chess.Board, time_limit: float = 1.5, max_depth: int = 64
               ) -> tuple[Optional[chess.Move], int, int]:
        """Iterative deepening within a time budget. Returns (move, score_cp, depth).

        Operates on an internal copy (including move stack, so repetition checks
        see the real game history). A mid-iteration time-out unwinds any pushes
        we didn't pop, so the engine stays consistent across requests.
        """
        self.killers.clear()
        self.history.clear()
        self.nodes = 0
        self.start = time.time()
        self.time_limit = time_limit
        self.root_scores = {}

        best_move: Optional[chess.Move] = None
        best_score = 0
        completed_depth = 0

        work_board = board.copy()  # includes move_stack for is_repetition()
        base_stack_len = len(work_board.move_stack)

        for depth in range(1, max_depth + 1):
            try:
                scores = self._search_root(work_board, depth)
            except _TimeUp:
                while len(work_board.move_stack) > base_stack_len:
                    work_board.pop()
                break
            self.root_scores = scores
            if scores:
                best_move, best_score = max(scores.items(), key=lambda kv: kv[1])
            completed_depth = depth
            if abs(best_score) > MATE - 1000:
                break
            if time.time() - self.start > self.time_limit * 0.45:
                break

        return best_move, best_score, completed_depth


# ---- Opening books ----
# Keys are EPD (piece-placement + turn + castling + ep), so they ignore clocks.
# Two layers:
#   1. _OPENING_BOOK — the default "varied" book: each position maps to a list of
#      plausible moves and the engine picks one at random for variety.
#   2. OPENINGS — named opening repertoires the user can pick from the UI.
#      Each one is a set of move-lines compiled into an EPD -> single-UCI dict,
#      so the engine plays deterministically toward that opening until either the
#      user deviates or we run out of book.

_OPENING_BOOK: dict[str, list[str]] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": [
        "e2e4", "d2d4", "c2c4", "g1f3", "e2e4", "d2d4",
    ],
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3": [
        "e7e5", "c7c5", "e7e6", "c7c6", "d7d5", "g8f6",
    ],
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3": [
        "g8f6", "d7d5", "e7e6",
    ],
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3": [
        "e7e5", "g8f6", "c7c5", "e7e6",
    ],
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq -": [
        "g8f6", "d7d5", "c7c5",
    ],
}


def _compile_lines(lines: list[list[str]]) -> dict[str, str]:
    """Walk each move sequence on a fresh board and record EPD -> next-UCI.

    If two lines share a prefix, later lines may overwrite earlier branchings —
    we accept the last one (good enough for a curated repertoire). If a UCI in
    a line is illegal in the current position, we stop walking that line.
    """
    book: dict[str, str] = {}
    for line in lines:
        b = chess.Board()
        for uci in line:
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                break
            if move not in b.legal_moves:
                break
            book[b.epd()] = uci
            b.push(move)
    return book


# Each opening: a display name plus a list of canonical move sequences.
# The sequences cover both colors so the bot stays in the opening whether it's
# playing white or black, as long as the human cooperates by playing the
# "expected" reply (otherwise we fall through to the search).
_OPENING_DEFS: dict[str, dict] = {
    "italian": {
        "name": "Italian Game",
        "lines": [
            # 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.c3 Nf6 5.d3 d6 — Giuoco Pianissimo
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6", "d2d3", "d7d6"],
            # 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 — Two Knights
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5"],
        ],
    },
    "ruy_lopez": {
        "name": "Ruy Lopez",
        "lines": [
            # 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
             "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6"],
        ],
    },
    "sicilian": {
        "name": "Sicilian Defense",
        "lines": [
            # 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6 — Najdorf
            ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6",
             "b1c3", "a7a6", "c1e3", "e7e5"],
            # 1.e4 c5 2.Nf3 Nc6 3.d4 cxd4 4.Nxd4 — Open Sicilian alt
            ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"],
        ],
    },
    "french": {
        "name": "French Defense",
        "lines": [
            # 1.e4 e6 2.d4 d5 3.Nc3 Nf6 4.Bg5 — Classical
            ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "c1g5", "f8e7", "e4e5", "f6d7"],
        ],
    },
    "caro_kann": {
        "name": "Caro-Kann",
        "lines": [
            # 1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Bf5 — Classical
            ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5",
             "e4g3", "f5g6", "h2h4"],
        ],
    },
    "queens_gambit": {
        "name": "Queen's Gambit",
        "lines": [
            # 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 — QGD Classical
            ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3", "e8g8"],
            # 1.d4 d5 2.c4 dxc4 3.Nf3 — QGA
            ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3"],
        ],
    },
    "kings_indian": {
        "name": "King's Indian Defense",
        "lines": [
            # 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3 O-O 6.Be2 e5
            ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6",
             "g1f3", "e8g8", "f1e2", "e7e5"],
        ],
    },
    "english": {
        "name": "English Opening",
        "lines": [
            # 1.c4 e5 2.Nc3 Nf6 3.g3 d5 — Reversed Sicilian
            ["c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5", "c4d5", "f6d5", "f1g2"],
            # 1.c4 c5 — Symmetrical
            ["c2c4", "c7c5", "g1f3", "g8f6", "b1c3", "b8c6"],
        ],
    },
}

# Compile each opening's lines into the EPD -> UCI lookup the engine consults.
OPENINGS: dict[str, dict] = {
    key: {"name": defn["name"], "book": _compile_lines(defn["lines"])}
    for key, defn in _OPENING_DEFS.items()
}


def available_openings() -> list[dict]:
    """Public helper for the API to list named openings."""
    return [{"id": key, "name": defn["name"]} for key, defn in OPENINGS.items()]


# ---- Public API ----

_engine = Engine()


def choose_move(board: chess.Board, time_limit: float = 1.5,
                opening: Optional[str] = None
                ) -> tuple[Optional[chess.Move], int, int]:
    """Pick the engine's move. Returns (move, score_cp_stm, depth).

    If `opening` is the id of a known named opening (see OPENINGS), the engine
    consults that opening's book first. The user dropping out of book (or the
    book running out) silently falls through to the default varied book and
    then to the search.
    """
    if board.is_game_over():
        return None, 0, 0

    epd = board.epd()
    legal_uci = {m.uci() for m in board.legal_moves}

    # Named opening repertoire (deterministic next move while in book).
    if opening and opening in OPENINGS:
        book_uci = OPENINGS[opening]["book"].get(epd)
        if book_uci is not None and book_uci in legal_uci:
            return chess.Move.from_uci(book_uci), 0, 0

    # Default varied book (random pick for variety on early moves).
    book_moves = _OPENING_BOOK.get(epd)
    if book_moves:
        choices = [u for u in book_moves if u in legal_uci]
        if choices:
            return chess.Move.from_uci(random.choice(choices)), 0, 0

    move, score, depth = _engine.search(board, time_limit=time_limit)
    if move is None:
        legals = list(board.legal_moves)
        if legals:
            return legals[0], 0, 0
        return None, 0, 0
    return move, score, depth


def top_moves(board: chess.Board, n: int = 3, time_limit: float = 1.0
              ) -> list[tuple[chess.Move, int]]:
    """Return top-N moves sorted best-first with their scores (cp, side-to-move POV)."""
    if board.is_game_over():
        return []
    _engine.search(board, time_limit=time_limit)
    items = sorted(_engine.root_scores.items(), key=lambda kv: -kv[1])
    return items[:n]


def evaluate_search(board: chess.Board, time_limit: float = 0.5) -> tuple[int, int]:
    """Search-based evaluation. Returns (score_cp_stm, depth)."""
    if board.is_checkmate():
        return (-MATE, 0)
    if (board.is_stalemate() or board.is_insufficient_material() or
            board.is_fivefold_repetition() or board.is_seventyfive_moves()):
        return (0, 0)
    _, score, depth = _engine.search(board, time_limit=time_limit)
    return score, depth


def mate_distance_from_score(score: int) -> Optional[int]:
    """If `score` represents a forced mate (from side-to-move POV), return signed mate distance in plies."""
    if score > MATE - 1000:
        return MATE - score
    if score < -(MATE - 1000):
        return -(MATE + score)
    return None
