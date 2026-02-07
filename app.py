import atexit
import chess
import os
import random
import signal
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from engine import get_ai_move, get_api_call_counts

app = Flask(__name__, static_folder="static")
CORS(app)

game = {
    "board": chess.Board(),
    "move_stats": [],
    "captured_by_white": [],
    "captured_by_black": [],
    "game_over_logged": False,
    "results_saved": False,
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


def _save_game_results_once():
    if game.get("results_saved"):
        return
    _save_game_results()
    game["results_saved"] = True


def _finalize_game_once():
    _log_game_over_summary_once()
    _save_game_results_once()


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


def _save_game_results():
    board = game["board"]
    stats = game["move_stats"]
    now = datetime.now()

    move_count = len(board.move_stack)
    white_moves = (move_count + 1) // 2
    black_moves = move_count // 2
    result = board.result()

    os.makedirs("results", exist_ok=True)
    filename = f"results/{now.strftime('%d%m%y')}.md"

    lines = []
    lines.append(f"# Game - {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## Summary\n")
    lines.append(f"- **Date**: {now.strftime('%B %d, %Y')}")
    lines.append(f"- **Time**: {now.strftime('%I:%M %p')}")
    lines.append(f"- **Result**: {result}")
    lines.append(f"- **White moves**: {white_moves}")
    lines.append(f"- **Black moves**: {black_moves}\n")

    total_pieces_polled = sum(int(stat.get("pieces_polled", 0)) for stat in stats)
    total_pieces_volunteered = sum(int(stat.get("pieces_volunteered", 0)) for stat in stats)
    overall_volunteer_pct = (
        (total_pieces_volunteered / total_pieces_polled * 100) if total_pieces_polled > 0 else 0
    )

    lines.append(f"- **Total pieces with legal moves (all AI turns)**: {total_pieces_polled}")
    lines.append(f"- **Total volunteered pieces (all AI turns)**: {total_pieces_volunteered}")
    lines.append(f"- **Overall volunteered rate**: {overall_volunteer_pct:.1f}%\n")

    # Per-move table
    lines.append("## Per-Move AI Stats\n")
    lines.append(
        "| Black Move | Pieces w/ Moves | Volunteered | % Volunteered | Orchestrator Used | Chose Highest Prob | Mini Avg (turn) | Orchestrator Time |"
    )
    lines.append(
        "|------------|-----------------|-------------|---------------|-------------------|--------------------|-----------------|-------------------|"
    )

    chose_highest_count_all = 0
    chose_highest_count_model = 0
    model_decision_count = 0
    orchestrator_attempt_count = 0
    orchestrator_error_count = 0
    all_mini_times = []
    mini_turn_totals = []
    all_orchestrator_times = []

    for i, stat in enumerate(stats):
        polled = stat["pieces_polled"]
        vol = stat["pieces_volunteered"]
        pct = f"{vol / polled * 100:.1f}%" if polled > 0 else "N/A"
        orchestrator_used = bool(stat.get("orchestrator_used", False))
        orchestrator_used_text = "Yes" if orchestrator_used else "No"
        orchestrator_error = bool(stat.get("orchestrator_error", False))
        if orchestrator_used or orchestrator_error:
            orchestrator_attempt_count += 1
        if orchestrator_error:
            orchestrator_error_count += 1

        chose_highest_raw = stat.get("chose_highest_prob")
        if isinstance(chose_highest_raw, bool):
            chose_text = "Yes" if chose_highest_raw else "No"
            if chose_highest_raw:
                chose_highest_count_all += 1
        else:
            chose_text = "N/A"

        if orchestrator_used and isinstance(chose_highest_raw, bool):
            model_decision_count += 1
            if chose_highest_raw:
                chose_highest_count_model += 1

        mini_times = [float(t) for t in stat.get("mini_times", []) if isinstance(t, (int, float)) and t > 0]
        mini_turn_total = sum(mini_times)
        mini_turn_totals.append(mini_turn_total)
        mini_avg_turn_text = f"{(mini_turn_total / len(mini_times)):.2f}s" if mini_times else "N/A"
        all_mini_times.extend(mini_times)

        orch_time = stat.get("orchestrator_time", 0)
        if isinstance(orch_time, (int, float)) and orch_time > 0:
            orch_time = float(orch_time)
            all_orchestrator_times.append(orch_time)
            orch_time_text = f"{orch_time:.2f}s"
        else:
            orch_time_text = "N/A"

        lines.append(
            f"| {i + 1} | {polled} | {vol} | {pct} | {orchestrator_used_text} | {chose_text} | {mini_avg_turn_text} | {orch_time_text} |"
        )

    total_ai_moves = len(stats)
    lines.append("")

    # Orchestrator accuracy
    if total_ai_moves > 0 and model_decision_count > 0:
        lines.append("## Orchestrator Accuracy\n")
        pct = chose_highest_count_model / model_decision_count * 100
        lines.append(
            f"- On model decisions only: **{chose_highest_count_model}/{model_decision_count}** "
            f"({pct:.1f}%) chose the highest probability move."
        )
        lines.append(
            f"- Across all AI turns: **{chose_highest_count_all}/{total_ai_moves}** "
            f"({(chose_highest_count_all / total_ai_moves * 100):.1f}%).\n"
        )
    elif total_ai_moves > 0:
        lines.append("## Orchestrator Accuracy\n")
        lines.append("- No valid orchestrator model decisions were recorded this game.\n")

    if orchestrator_attempt_count > 0:
        lines.append(
            f"- Orchestrator unavailable/errors: **{orchestrator_error_count}/{orchestrator_attempt_count}** "
            f"({(orchestrator_error_count / orchestrator_attempt_count * 100):.1f}%).\n"
        )
    elif total_ai_moves > 0:
        lines.append("- Orchestrator unavailable/errors: **0/0** (N/A).\n")

    # API response times
    lines.append("## API Response Times\n")
    lines.append("| Metric | gpt-5-mini | gpt-5.2 |")
    lines.append("|--------|-----------|---------|")

    if all_mini_times:
        avg_mini = sum(all_mini_times) / len(all_mini_times)
        avg_mini_per_turn = sum(mini_turn_totals) / max(total_ai_moves, 1)
        total_mini = sum(mini_turn_totals)
        min_mini = min(all_mini_times)
        max_mini = max(all_mini_times)
    else:
        avg_mini = avg_mini_per_turn = total_mini = min_mini = max_mini = 0

    if all_orchestrator_times:
        avg_orch = sum(all_orchestrator_times) / len(all_orchestrator_times)
        avg_orch_per_turn = sum(all_orchestrator_times) / max(total_ai_moves, 1)
        total_orch = sum(all_orchestrator_times)
        min_orch = min(all_orchestrator_times)
        max_orch = max(all_orchestrator_times)
    else:
        avg_orch = avg_orch_per_turn = total_orch = min_orch = max_orch = 0

    lines.append(f"| Avg per response | {avg_mini:.2f}s | {avg_orch:.2f}s |")
    lines.append(f"| Avg per turn | {avg_mini_per_turn:.2f}s | {avg_orch_per_turn:.2f}s |")
    lines.append(f"| Total game time | {total_mini:.2f}s | {total_orch:.2f}s |")
    lines.append(f"| Calls counted | {len(all_mini_times)} | {len(all_orchestrator_times)} |")
    lines.append(f"| Min | {min_mini:.2f}s | {min_orch:.2f}s |")
    lines.append(f"| Max | {max_mini:.2f}s | {max_orch:.2f}s |")
    lines.append("\n---\n")

    with open(filename, "a") as f:
        f.write("\n".join(lines))

    print(f"[RESULTS] Game saved to {filename}")


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


def _record_capture(board, move):
    if not board.is_capture(move):
        return

    if board.is_en_passant(move):
        capture_square = move.to_square - 8 if board.turn == chess.WHITE else move.to_square + 8
    else:
        capture_square = move.to_square

    captured_piece = board.piece_at(capture_square)
    if captured_piece is None:
        return

    captured_symbol = captured_piece.symbol()
    if board.turn == chess.WHITE:
        game["captured_by_white"].append(captured_symbol)
    else:
        game["captured_by_black"].append(captured_symbol)


def _append_move_stats(ai_result):
    if not isinstance(ai_result, dict):
        return

    pieces_polled = ai_result.get("pieces_polled", 0)
    if not isinstance(pieces_polled, (int, float)):
        pieces_polled = 0
    pieces_polled = max(int(pieces_polled), 0)

    pieces_volunteered = ai_result.get("pieces_volunteered", 0)
    if not isinstance(pieces_volunteered, (int, float)):
        pieces_volunteered = 0
    pieces_volunteered = max(int(pieces_volunteered), 0)

    mini_times = [
        float(t)
        for t in ai_result.get("mini_times", [])
        if isinstance(t, (int, float)) and t > 0
    ]

    orchestrator_time = ai_result.get("orchestrator_time", 0)
    if not isinstance(orchestrator_time, (int, float)) or orchestrator_time < 0:
        orchestrator_time = 0
    orchestrator_time = float(orchestrator_time)

    chose_highest_prob = ai_result.get("chose_highest_prob")
    if not isinstance(chose_highest_prob, bool):
        chose_highest_prob = None

    orchestrator_used = bool(ai_result.get("orchestrator_used", False))
    orchestrator_error = bool(ai_result.get("orchestrator_error", False))

    game["move_stats"].append({
        "pieces_polled": pieces_polled,
        "pieces_volunteered": pieces_volunteered,
        "mini_times": mini_times,
        "orchestrator_time": orchestrator_time,
        "chose_highest_prob": chose_highest_prob,
        "orchestrator_used": orchestrator_used,
        "orchestrator_error": orchestrator_error,
    })


def _normalize_proposals(raw_proposals):
    return raw_proposals if isinstance(raw_proposals, list) else []


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/new", methods=["POST"])
def new_game():
    game["board"] = chess.Board()
    game["move_stats"] = []
    game["captured_by_white"] = []
    game["captured_by_black"] = []
    game["game_over_logged"] = False
    game["results_saved"] = False
    return jsonify({
        "fen": game["board"].fen(),
        "captured_by_white": game["captured_by_white"],
        "captured_by_black": game["captured_by_black"],
    })


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

    _record_capture(board, move)
    board.push(move)

    if board.is_game_over():
        _finalize_game_once()
        return jsonify({
            "fen": board.fen(),
            "game_over": True,
            "result": board.result(),
            "captured_by_white": game["captured_by_white"],
            "captured_by_black": game["captured_by_black"],
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
            _finalize_game_once()
            return jsonify({
                "fen": board.fen(),
                "game_over": True,
                "result": board.result(),
                "captured_by_white": game["captured_by_white"],
                "captured_by_black": game["captured_by_black"],
            })

    _record_capture(board, ai_move)
    board.push(ai_move)
    _append_move_stats(ai_result)
    if board.is_game_over():
        _finalize_game_once()
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
            "chose_highest_prob": ai_result.get("chose_highest_prob"),
            "orchestrator_time": ai_result.get("orchestrator_time", 0),
            "orchestrator_used": ai_result.get("orchestrator_used", False),
            "orchestrator_error": ai_result.get("orchestrator_error", False),
        },
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "captured_by_white": game["captured_by_white"],
        "captured_by_black": game["captured_by_black"],
    })


@app.route("/api/state", methods=["GET"])
def get_state():
    board = game["board"]
    return jsonify({
        "fen": board.fen(),
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "captured_by_white": game["captured_by_white"],
        "captured_by_black": game["captured_by_black"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
