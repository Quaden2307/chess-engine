"""
Vercel serverless entry. Delegates to the classical engine in engine.py.
"""
import os
import sys

# Make the parent directory importable so we can pull in engine.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from flask import Flask, jsonify, request
from flask_cors import CORS

import engine

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# Vercel serverless has tight time budgets and shared CPU; keep the engine
# snappy so cold-start + search stays well under the 10s Hobby cap.
TIME_BEST_MOVE = float(os.environ.get('CHESS_TIME_MOVE', '0.8'))
TIME_SUGGEST = float(os.environ.get('CHESS_TIME_SUGGEST', '0.4'))
TIME_EVAL = float(os.environ.get('CHESS_TIME_EVAL', '0.2'))

SCORE_CAP_CP = 5000


def _cp(cp: int) -> float:
    clamped = max(-SCORE_CAP_CP, min(SCORE_CAP_CP, cp))
    return round(clamped / 100.0, 2)


def _white_pov(score_stm: int, board: chess.Board) -> int:
    return score_stm if board.turn == chess.WHITE else -score_stm


def _mate_in_moves_white_pov(score_stm: int, board: chess.Board):
    plies = engine.mate_distance_from_score(score_stm)
    if plies is None:
        return None
    moves = (abs(plies) + 1) // 2
    side_to_mate = board.turn if plies > 0 else (not board.turn)
    return moves if side_to_mate == chess.WHITE else -moves


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'engine': 'classical-alphabeta'})


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        fen = (request.json or {}).get('fen')
        if not fen:
            return jsonify({'error': 'FEN string required'}), 400
        board = chess.Board(fen)
        score_stm, _ = engine.evaluate_search(board, time_limit=TIME_EVAL)
        resp = {'evaluation': _cp(_white_pov(score_stm, board)), 'fen': fen}
        mate = _mate_in_moves_white_pov(score_stm, board)
        if mate is not None:
            resp['mate_in'] = mate
        return jsonify(resp)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/best-move', methods=['POST'])
def best_move():
    try:
        fen = (request.json or {}).get('fen')
        if not fen:
            return jsonify({'error': 'FEN string required'}), 400
        board = chess.Board(fen)
        if board.is_game_over():
            return jsonify({'error': 'Game is over'}), 400
        move, score, _ = engine.choose_move(board, time_limit=TIME_BEST_MOVE)
        if move is None:
            return jsonify({'error': 'No legal moves available'}), 400
        return jsonify({'move': move.uci(), 'san': board.san(move), 'evaluation': _cp(score)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/suggested-moves', methods=['POST'])
def suggested_moves():
    try:
        data = request.json or {}
        fen = data.get('fen')
        top_n = int(data.get('top_n', 3))
        if not fen:
            return jsonify({'error': 'FEN string required'}), 400
        board = chess.Board(fen)
        if board.is_game_over():
            return jsonify({'moves': [], 'current_evaluation': 0.0})
        results = engine.top_moves(board, n=top_n, time_limit=TIME_SUGGEST)
        moves = [{'move': m.uci(), 'san': board.san(m), 'score': _cp(s)} for m, s in results]
        if results:
            best_score = results[0][1]
        else:
            best_score, _ = engine.evaluate_search(board, time_limit=TIME_EVAL)
        return jsonify({'moves': moves, 'current_evaluation': _cp(_white_pov(best_score, board))})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/make-ai-move', methods=['POST'])
def make_ai_move():
    try:
        data = request.json or {}
        fen = data.get('fen')
        opening = data.get('opening') or None
        if not fen:
            return jsonify({'error': 'FEN string required'}), 400
        board = chess.Board(fen)
        if board.is_game_over():
            return jsonify({'error': 'Game is over', 'result': board.result()}), 400
        move, _, _ = engine.choose_move(board, time_limit=TIME_BEST_MOVE, opening=opening)
        if move is None:
            return jsonify({'error': 'No legal moves available'}), 400
        san = board.san(move)
        board.push(move)
        return jsonify({
            'move': move.uci(),
            'san': san,
            'new_fen': board.fen(),
            'is_game_over': board.is_game_over(),
            'result': board.result() if board.is_game_over() else None,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/openings', methods=['GET'])
def openings():
    return jsonify({'openings': engine.available_openings()})
