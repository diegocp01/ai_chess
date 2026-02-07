# ♟️ LLM Chess — Every Piece Has a Voice

An interactive chess app where the AI (Black) is chosen by a multi-agent LLM pipeline:

1. Each Black piece with legal moves is polled with **`gpt-5-mini`**.
2. Proposed moves are heuristically scored in Python.
3. An orchestrator model (**`gpt-5.2`**) picks the final move.

You play as White in the browser.

## Stack

- Frontend: HTML/CSS/JS + `chess.js` + `chessboard.js`
- Backend: Flask + Flask-CORS
- Chess rules/board state: `python-chess`
- LLM calls: OpenAI Python SDK

## Requirements

- Python 3.10+ (recommended)
- OpenAI API key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_key_here
```

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open: `http://localhost:5000`

## API Endpoints

- `POST /api/new`: reset game, returns current FEN
- `POST /api/move`: submit White move (`from`, `to`, optional `promotion`)
- `GET /api/state`: current board/game-over state

## Runtime Behavior Notes

- Backend validates user moves and AI moves against legal chess moves.
- If model output is invalid/unavailable, AI falls back to a legal move.
- Game state is stored in-memory in a single global board (good for local demo, not multi-user production).
