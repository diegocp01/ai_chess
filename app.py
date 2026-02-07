import atexit
import chess
import random
import signal
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from engine import get_ai_move, get_api_call_counts

app = Flask(__name__, static_folder="static")
CORS(app)

game = {
    "board": chess.Board(),
    "game_over_logged": False,
}
shutdown_summary_printed = False


def _print_api_call_summary(event_name):
    counts = get_api_call_counts()
    mini_calls = counts.get("gpt-5-mini", 0)
    orchestrator_calls = counts.get("gpt-5.2", 0)
    total_calls = mini_calls + orchestrator_calls
    print(
        f"[API SUMMARY - {event_name}] "
        f"gpt-5-mini={mini_calls}, gpt-5.2={orchestrator_calls}, total={total_calls}"
    )


def _log_game_over_summary_once():
    if game.get("game_over_logged"):
        return
    _print_api_call_summary("game over")
    game["game_over_logged"] = True


def _print_shutdown_summary_once(event_name):
    global shutdown_summary_printed
    if shutdown_summary_printed:
        return
    _print_api_call_summary(event_name)
    shutdown_summary_printed = True


def _handle_shutdown_signal(signum, _frame):
    _print_shutdown_summary_once(f"signal {signum}")
    raise SystemExit(0)


atexit.register(_print_shutdown_summary_once, "process exit")
signal.signal(signal.SIGINT, _handle_shutdown_signal)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)


def _parse_square(square_name, field_name):
    if not isinstance(square_name, str):
        raise ValueError(f"Missing or invalid '{field_name}' square.")
    try:
        return chess.parse_square(square_name)
    except ValueError as exc:
        raise ValueError(f"Invalid square for '{field_name}': {square_name}") from exc


def _parse_promotion(promotion_symbol):
    if promotion_symbol in (None, ""):
        return None
    if not isinstance(promotion_symbol, str) or promotion_symbol.lower() not in {"q", "r", "b", "n"}:
        raise ValueError("Invalid promotion piece. Use one of: q, r, b, n.")
    return chess.Piece.from_symbol(promotion_symbol.lower()).piece_type


def _find_matching_legal_move(board, from_sq, to_sq, promotion_piece_type=None):
    move = chess.Move(from_sq, to_sq, promotion=promotion_piece_type)
    if move in board.legal_moves:
        return move

    for legal_move in board.legal_moves:
        if legal_move.from_square == from_sq and legal_move.to_square == to_sq:
            return legal_move
    return None


def _fallback_ai_move(board, reason):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, {"reasoning": reason, "proposals": []}

    move = random.choice(legal_moves)
    return move, {
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
        "promotion": chess.piece_symbol(move.promotion).lower() if move.promotion else None,
        "reasoning": reason,
        "proposals": [],
    }


def _normalize_proposals(raw_proposals):
    return raw_proposals if isinstance(raw_proposals, list) else []


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/new", methods=["POST"])
def new_game():
    game["board"] = chess.Board()
    game["game_over_logged"] = False
    return jsonify({"fen": game["board"].fen()})


@app.route("/api/move", methods=["POST"])
def human_move():
    """Handle human (White) move, then get AI (Black) response."""
    data = request.get_json(silent=True) or {}

    board = game["board"]

    try:
        from_sq = _parse_square(data.get("from"), "from")
        to_sq = _parse_square(data.get("to"), "to")
        promotion = _parse_promotion(data.get("promotion"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    move = _find_matching_legal_move(board, from_sq, to_sq, promotion)
    if move is None:
        return jsonify({"error": "Illegal move"}), 400

    board.push(move)

    if board.is_game_over():
        _log_game_over_summary_once()
        return jsonify({
            "fen": board.fen(),
            "game_over": True,
            "result": board.result(),
        })

    # AI turn
    try:
        ai_result = get_ai_move(board.fen())
    except Exception as exc:
        print(f"AI move generation error: {exc}")
        ai_result = {}

    if not isinstance(ai_result, dict):
        ai_result = {}

    try:
        ai_from = _parse_square(ai_result.get("from"), "from")
        ai_to = _parse_square(ai_result.get("to"), "to")
        ai_promo = _parse_promotion(ai_result.get("promotion"))
        ai_move = _find_matching_legal_move(board, ai_from, ai_to, ai_promo)
    except ValueError:
        ai_move = None

    if ai_move is None:
        ai_move, fallback = _fallback_ai_move(board, "fallback - invalid AI move from orchestrator")
        ai_result = (
            {**fallback, "proposals": _normalize_proposals(ai_result.get("proposals"))}
            if isinstance(ai_result, dict)
            else fallback
        )
        if ai_move is None:
            _log_game_over_summary_once()
            return jsonify({
                "fen": board.fen(),
                "game_over": True,
                "result": board.result(),
            })

    board.push(ai_move)
    if board.is_game_over():
        _log_game_over_summary_once()
    proposals = _normalize_proposals(ai_result.get("proposals"))

    return jsonify({
        "fen": board.fen(),
        "ai_move": {
            "from": chess.square_name(ai_move.from_square),
            "to": chess.square_name(ai_move.to_square),
            "promotion": chess.piece_symbol(ai_move.promotion).lower() if ai_move.promotion else None,
            "reasoning": ai_result.get("reasoning", ""),
            "chosen_piece_reason": ai_result.get("chosen_piece_reason", ""),
            "proposals": proposals,
            "pieces_polled": ai_result.get("pieces_polled", 0),
            "pieces_volunteered": ai_result.get("pieces_volunteered", 0),
        },
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
    })


@app.route("/api/state", methods=["GET"])
def get_state():
    board = game["board"]
    return jsonify({
        "fen": board.fen(),
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
