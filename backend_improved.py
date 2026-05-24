"""
Flask backend for the chess UI.

Move generation and evaluation are delegated to the classical search engine in
`engine.py` (iterative-deepening alpha-beta with quiescence). The previous
neural-net evaluator was removed: at 1-ply lookahead it played poorly and
looped on repetition.
"""
import os

import chess
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import engine

app = Flask(__name__, static_folder='chess-frontend/build', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)


# Time budgets (seconds) for the search per endpoint.
TIME_BEST_MOVE = float(os.environ.get('CHESS_TIME_MOVE', '1.5'))
TIME_SUGGEST = float(os.environ.get('CHESS_TIME_SUGGEST', '0.8'))
TIME_EVAL = float(os.environ.get('CHESS_TIME_EVAL', '0.4'))

# Cap displayed per-move scores so the UI doesn't render +999.xx for mate lines.
SCORE_DISPLAY_CAP_CP = 5000


def _cp_to_pawns(cp: int) -> float:
    """Convert centipawns to pawns, clamped for UI sanity."""
    clamped = max(-SCORE_DISPLAY_CAP_CP, min(SCORE_DISPLAY_CAP_CP, cp))
    return round(clamped / 100.0, 2)


def _white_pov(score_stm: int, board: chess.Board) -> int:
    return score_stm if board.turn == chess.WHITE else -score_stm


def _mate_in_moves_white_pov(score_stm: int, board: chess.Board):
    """Signed mate distance in full moves from white's perspective, or None."""
    plies = engine.mate_distance_from_score(score_stm)
    if plies is None:
        return None
    moves = (abs(plies) + 1) // 2
    side_to_mate = board.turn if plies > 0 else (not board.turn)
    return moves if side_to_mate == chess.WHITE else -moves


# ---- Endpoints ----

@app.route('/api/health', methods=['GET'])
def health():
    import sys
    return jsonify({
        'status': 'healthy',
        'engine': 'classical-alphabeta',
        'python_version': sys.version,
    })


@app.route('/api/evaluate', methods=['POST'])
def evaluate_endpoint():
    try:
        fen = (request.json or {}).get('fen')
        if not fen:
            return jsonify({'error': 'FEN string required'}), 400
        board = chess.Board(fen)
        score_stm, _ = engine.evaluate_search(board, time_limit=TIME_EVAL)
        white_pov = _white_pov(score_stm, board)
        response = {
            'evaluation': _cp_to_pawns(white_pov),
            'fen': fen,
        }
        mate = _mate_in_moves_white_pov(score_stm, board)
        if mate is not None:
            response['mate_in'] = mate
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/best-move', methods=['POST'])
def best_move_endpoint():
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
        san = board.san(move)
        return jsonify({
            'move': move.uci(),
            'san': san,
            'evaluation': _cp_to_pawns(score),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/suggested-moves', methods=['POST'])
def suggested_moves_endpoint():
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

        moves_payload = []
        for mv, sc in results:
            moves_payload.append({
                'move': mv.uci(),
                'san': board.san(mv),
                'score': _cp_to_pawns(sc),
            })

        # Current eval is the score of the best move (already a search result).
        if results:
            best_score = results[0][1]
        else:
            best_score, _ = engine.evaluate_search(board, time_limit=TIME_EVAL)
        white_pov = _white_pov(best_score, board)
        return jsonify({
            'moves': moves_payload,
            'current_evaluation': _cp_to_pawns(white_pov),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/make-ai-move', methods=['POST'])
def make_ai_move_endpoint():
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
def openings_endpoint():
    return jsonify({'openings': engine.available_openings()})


# ---- Frontend static serving (single-service Docker deployment) ----

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Chess backend listening on http://0.0.0.0:{port} (engine: classical alpha-beta)")
    app.run(host='0.0.0.0', debug=False, port=port)
