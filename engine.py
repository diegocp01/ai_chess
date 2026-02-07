import chess
import json
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

try:
    client = OpenAI()
except Exception as exc:
    print(f"OpenAI client initialization error: {exc}")
    client = None

PIECE_NAMES = {
    chess.PAWN: "Pawn",
    chess.KNIGHT: "Knight",
    chess.BISHOP: "Bishop",
    chess.ROOK: "Rook",
    chess.QUEEN: "Queen",
    chess.KING: "King",
}

API_CALL_COUNTS = {
    "gpt-5-mini": 0,
    "gpt-5.2": 0,
}
_api_call_counts_lock = threading.Lock()


class PiecePollResponse(BaseModel):
    should_move: bool
    target_square: str
    reason: str


class OrchestratorDecision(BaseModel):
    piece: str
    from_square: str
    to_square: str


def _increment_api_call(model_name):
    with _api_call_counts_lock:
        API_CALL_COUNTS[model_name] = API_CALL_COUNTS.get(model_name, 0) + 1


def get_api_call_counts():
    with _api_call_counts_lock:
        return dict(API_CALL_COUNTS)


def _strip_markdown_fences(text):
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _find_legal_move(board, from_square, to_square, promotion_symbol=None):
    if not isinstance(from_square, str) or not isinstance(to_square, str):
        return None

    try:
        from_sq = chess.parse_square(from_square)
        to_sq = chess.parse_square(to_square)
    except ValueError:
        return None

    promotion_piece_type = None
    if promotion_symbol not in (None, ""):
        if not isinstance(promotion_symbol, str) or promotion_symbol.lower() not in {"q", "r", "b", "n"}:
            return None
        promotion_piece_type = chess.Piece.from_symbol(promotion_symbol.lower()).piece_type

    move = chess.Move(from_sq, to_sq, promotion=promotion_piece_type)
    if move in board.legal_moves:
        return move

    for legal_move in board.legal_moves:
        if legal_move.from_square == from_sq and legal_move.to_square == to_sq:
            return legal_move
    return None


def _random_legal_move(board):
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves) if legal_moves else None


def _best_scored_fallback(scored_proposals, reasoning):
    best = max(scored_proposals, key=lambda x: x["probability_score"])
    return {
        "piece": best["piece"],
        "from": best["from_square"],
        "to": best["target_square"],
        "reasoning": reasoning,
    }


def get_black_pieces_with_moves(board):
    """Get all Black pieces that have legal moves and per-move capture risk."""
    legal_moves_by_square = {}
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece is None or piece.color != chess.BLACK:
            continue
        legal_moves_by_square.setdefault(move.from_square, []).append(move)

    pieces = []
    for square in sorted(legal_moves_by_square.keys()):
        piece = board.piece_at(square)
        if piece is None or piece.color != chess.BLACK:
            continue

        moves = legal_moves_by_square[square]
        if not moves:
            continue

        legal_moves = []
        legal_moves_with_risk = []
        for move in moves:
            uci_move = move.uci()
            legal_moves.append(uci_move)
            legal_moves_with_risk.append({
                "uci": uci_move,
                "target_square": chess.square_name(move.to_square),
                "lose_piece_probability": _move_capture_risk_probability(board, move),
            })

        legal_moves_with_risk.sort(key=lambda item: item["lose_piece_probability"])
        pieces.append({
            "piece": PIECE_NAMES[piece.piece_type],
            "square": chess.square_name(square),
            "legal_moves": legal_moves,
            "legal_moves_with_risk": legal_moves_with_risk,
        })
    return pieces


def poll_piece(piece_info, fen):
    """Ask gpt-5-mini whether this piece should move.
    Returns {"proposal": ... or None, "api_time": float}.
    """
    if client is None:
        return {"proposal": None, "api_time": 0}

    prompt = (
        f"You are a {piece_info['piece']} at {piece_info['square']}. "
        f"Board state (FEN): {fen}. "
        f"Your legal moves: {piece_info['legal_moves']}. "
        f"Your legal moves with estimated probability (0-1) of being captured by White on the next turn: "
        f"{piece_info['legal_moves_with_risk']}. "
        f"Should you move this turn? If yes, pick one target square and give a 1-liner why."
    )
    try:
        _increment_api_call("gpt-5-mini")
        start = time.time()
        response = client.responses.parse(
            model="gpt-5-mini",
            input=[{"role": "user", "content": prompt}],
            text_format=PiecePollResponse,
        )
        api_time = time.time() - start

        data = response.output_parsed
        if data is None or not data.should_move:
            return {"proposal": None, "api_time": api_time}

        target_square = data.target_square.strip().lower()
        if len(target_square) in (4, 5) and target_square[:2] == piece_info["square"]:
            target_square = target_square[2:4]

        if len(target_square) != 2:
            return {"proposal": None, "api_time": api_time}

        legal_targets = {uci_move[2:4] for uci_move in piece_info["legal_moves"]}
        if target_square not in legal_targets:
            return {"proposal": None, "api_time": api_time}

        risk_by_target = {}
        for move_data in piece_info.get("legal_moves_with_risk", []):
            target = move_data.get("target_square")
            risk = move_data.get("lose_piece_probability")
            if isinstance(target, str) and isinstance(risk, (int, float)):
                if target not in risk_by_target:
                    risk_by_target[target] = risk
                else:
                    risk_by_target[target] = min(risk_by_target[target], risk)

        return {
            "proposal": {
                "piece": piece_info["piece"],
                "from_square": piece_info["square"],
                "target_square": target_square,
                "reason": data.reason,
                "legal_moves": piece_info["legal_moves"],
                "lose_piece_probability": risk_by_target.get(target_square),
            },
            "api_time": api_time,
        }
    except Exception as exc:
        print(f"Error polling {piece_info['piece']} at {piece_info['square']}: {exc}")
    return {"proposal": None, "api_time": 0}


def poll_all_pieces(board):
    """Poll all Black pieces in parallel using threads.
    Returns (pieces_polled, proposals, mini_times).
    """
    pieces = get_black_pieces_with_moves(board)
    fen = board.fen()
    proposals = []
    mini_times = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(poll_piece, p, fen) for p in pieces]
        for f in futures:
            result = f.result()
            if result is not None:
                api_time = result.get("api_time", 0)
                if isinstance(api_time, (int, float)) and api_time > 0:
                    mini_times.append(float(api_time))
                if result["proposal"] is not None:
                    proposals.append(result["proposal"])
    return len(pieces), proposals, mini_times


# --- Step 2: Heuristic probability scoring ---

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3.25,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

CENTER_SQUARES = {chess.E4, chess.D4, chess.E5, chess.D5}
EXTENDED_CENTER = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}


def _score_move_for_color(board, move, mover_color):
    """Heuristic score for a move by `mover_color`."""
    opponent_color = not mover_color
    score = 0.0

    captured = board.piece_at(move.to_square)
    if captured and captured.color == opponent_color:
        score += PIECE_VALUES.get(captured.piece_type, 0) * 2

    board.push(move)
    if board.is_check():
        score += 3
    if board.is_checkmate():
        score += 100

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == opponent_color and board.is_attacked_by(mover_color, sq):
            score += PIECE_VALUES.get(piece.piece_type, 0) * 0.3
    board.pop()

    if move.to_square in CENTER_SQUARES:
        score += 1.5
    elif move.to_square in EXTENDED_CENTER:
        score += 0.5

    if board.is_attacked_by(opponent_color, move.to_square):
        moving_piece = board.piece_at(move.from_square)
        if moving_piece and moving_piece.color == mover_color:
            score -= PIECE_VALUES.get(moving_piece.piece_type, 0) * 0.8

    return max(score, 0.1)


def _move_probability_distribution(board):
    """Probability distribution over current side-to-move legal replies."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []

    mover_color = board.turn
    scored_moves = []
    total_score = 0.0
    for move in legal_moves:
        move_score = _score_move_for_color(board, move, mover_color)
        scored_moves.append((move, move_score))
        total_score += move_score

    if total_score <= 0:
        uniform = 1.0 / len(scored_moves)
        return [(move, uniform) for move, _ in scored_moves]

    return [(move, score / total_score) for move, score in scored_moves]


def _move_capture_risk_probability(board, move):
    """
    Probability that the moved Black piece is captured by White on the next ply.
    Uses a normalized distribution over White legal replies.
    """
    board.push(move)
    moved_square = move.to_square
    moved_piece = board.piece_at(moved_square)
    distribution = _move_probability_distribution(board)

    if moved_piece is None or not distribution:
        board.pop()
        return 0.0

    risk_probability = 0.0
    for reply_move, reply_probability in distribution:
        board.push(reply_move)
        if board.piece_at(moved_square) != moved_piece:
            risk_probability += reply_probability
        board.pop()

    board.pop()
    return round(min(max(risk_probability, 0.0), 1.0), 3)


def score_move(board, move):
    """Heuristic score for a proposed move."""
    moving_piece = board.piece_at(move.from_square)
    mover_color = moving_piece.color if moving_piece else board.turn
    return _score_move_for_color(board, move, mover_color)


def calculate_probabilities(board, proposals):
    """Score each proposal and normalize to probabilities."""
    scored = []
    for proposal in proposals:
        move = _find_legal_move(
            board,
            proposal.get("from_square"),
            proposal.get("target_square"),
            proposal.get("promotion"),
        )
        if move is None:
            continue

        scored_proposal = dict(proposal)
        scored_proposal["from_square"] = chess.square_name(move.from_square)
        scored_proposal["target_square"] = chess.square_name(move.to_square)
        scored_proposal["probability_score"] = round(score_move(board, move), 2)
        scored_proposal["uci"] = move.uci()
        scored.append(scored_proposal)

    total = sum(sp["probability_score"] for sp in scored)
    if total > 0:
        for sp in scored:
            sp["probability_score"] = round(sp["probability_score"] / total, 3)
    return scored


# --- Step 3: Orchestrator ---

def _is_highest_prob(scored_proposals, from_sq, to_sq):
    """Check if the chosen move matches the highest probability proposal."""
    if not scored_proposals:
        return True
    highest = max(scored_proposals, key=lambda x: x["probability_score"])
    return highest["from_square"] == from_sq and highest["target_square"] == to_sq


def orchestrator_decide(board, scored_proposals):
    """Ask gpt-5.2 to pick the best move from proposals."""
    if not scored_proposals:
        return None
    if client is None:
        result = _best_scored_fallback(scored_proposals, "fallback - OpenAI client not configured")
        result["orchestrator_time"] = 0
        result["chose_highest_prob"] = None
        result["orchestrator_used"] = False
        result["orchestrator_error"] = True
        return result

    proposals_text = json.dumps(scored_proposals, indent=2)
    prompt = (
        f"You are a chess grandmaster playing Black. "
        f"Board (FEN): {board.fen()}. "
        f"Your pieces propose these moves:\n{proposals_text}\n\n"
        f"Pick the best move. The MAIN GOAL is to win the game."
    )
    try:
        _increment_api_call("gpt-5.2")
        start = time.time()
        response = client.responses.parse(
            model="gpt-5.2",
            input=[{"role": "user", "content": prompt}],
            text_format=OrchestratorDecision,
        )
        orchestrator_time = time.time() - start

        decision = response.output_parsed
        if decision is not None:
            return {
                "piece": decision.piece,
                "from": decision.from_square,
                "to": decision.to_square,
                "orchestrator_time": orchestrator_time,
                "chose_highest_prob": _is_highest_prob(
                    scored_proposals, decision.from_square, decision.to_square
                ),
                "orchestrator_used": True,
                "orchestrator_error": False,
            }
    except Exception as exc:
        print(f"Orchestrator error: {exc}")

    result = _best_scored_fallback(scored_proposals, "fallback - highest heuristic score")
    result["orchestrator_time"] = 0
    result["chose_highest_prob"] = None
    result["orchestrator_used"] = False
    result["orchestrator_error"] = True
    return result


# --- Main entry point ---

def _find_chosen_reason(scored, move):
    """Find the reason from the proposal matching the chosen move."""
    from_name = chess.square_name(move.from_square)
    to_name = chess.square_name(move.to_square)
    for p in scored:
        if p.get("from_square") == from_name and p.get("target_square") == to_name:
            return p.get("reason", "")
    return ""


def get_ai_move(fen):
    """Full AI pipeline: poll -> score -> orchestrate -> validate."""
    board = chess.Board(fen)

    # Step 1: Poll pieces
    pieces_polled, proposals, mini_times = poll_all_pieces(board)
    pieces_volunteered = len(proposals)

    pipeline = {
        "pieces_polled": pieces_polled,
        "pieces_volunteered": pieces_volunteered,
        "mini_times": mini_times,
    }

    if not proposals:
        move = _random_legal_move(board)
        if move is None:
            return {}
        return {
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
            "reasoning": "no piece volunteered - random fallback",
            "chosen_piece_reason": "",
            "proposals": [],
            "orchestrator_time": 0,
            "chose_highest_prob": None,
            "orchestrator_used": False,
            "orchestrator_error": False,
            **pipeline,
        }

    # Step 2: Calculate probabilities
    scored = calculate_probabilities(board, proposals)
    if not scored:
        move = _random_legal_move(board)
        if move is None:
            return {}
        return {
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
            "reasoning": "no valid proposals - random fallback",
            "chosen_piece_reason": "",
            "proposals": [],
            "orchestrator_time": 0,
            "chose_highest_prob": None,
            "orchestrator_used": False,
            "orchestrator_error": False,
            **pipeline,
        }

    # Step 3: Orchestrator decides
    decision = orchestrator_decide(board, scored)

    # Step 4: Validate
    if isinstance(decision, dict):
        move = _find_legal_move(
            board,
            decision.get("from"),
            decision.get("to"),
            decision.get("promotion"),
        )
        if move is not None:
            return {
                "from": chess.square_name(move.from_square),
                "to": chess.square_name(move.to_square),
                "promotion": chess.piece_symbol(move.promotion).lower() if move.promotion else None,
                "reasoning": decision.get("reasoning", ""),
                "chosen_piece_reason": _find_chosen_reason(scored, move),
                "proposals": scored,
                "orchestrator_time": decision.get("orchestrator_time", 0),
                "chose_highest_prob": decision.get("chose_highest_prob"),
                "orchestrator_used": bool(decision.get("orchestrator_used", False)),
                "orchestrator_error": bool(decision.get("orchestrator_error", False)),
                **pipeline,
            }

    # Final fallback
    best = max(scored, key=lambda x: x["probability_score"])
    move = _find_legal_move(
        board,
        best.get("from_square"),
        best.get("target_square"),
        best.get("promotion"),
    )
    if move is None:
        move = _random_legal_move(board)
        if move is None:
            return {}
        reasoning = "orchestrator failed - random legal fallback"
        chose_highest_prob = None
    else:
        reasoning = "orchestrator failed - best heuristic pick"
        if isinstance(decision, dict):
            chose_highest_prob = decision.get("chose_highest_prob")
        else:
            chose_highest_prob = None

    return {
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
        "promotion": chess.piece_symbol(move.promotion).lower() if move.promotion else None,
        "reasoning": reasoning,
        "chosen_piece_reason": _find_chosen_reason(scored, move) if scored else "",
        "proposals": scored,
        "orchestrator_time": decision.get("orchestrator_time", 0) if isinstance(decision, dict) else 0,
        "chose_highest_prob": chose_highest_prob,
        "orchestrator_used": bool(decision.get("orchestrator_used", False)) if isinstance(decision, dict) else False,
        "orchestrator_error": bool(decision.get("orchestrator_error", False)) if isinstance(decision, dict) else False,
        **pipeline,
    }
