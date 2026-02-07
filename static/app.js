/* global Chessboard, Chess, $ */

var game = new Chess();
var board = null;
var thinking = false;
var selectedSquare = null;
var selectedTargets = [];

function updateStatus() {
  var status = document.getElementById("status");
  if (game.game_over()) {
    if (game.in_checkmate()) {
      status.textContent = game.turn() === "w"
        ? "Checkmate - Black wins!"
        : "Checkmate - White wins!";
    } else if (game.in_draw()) {
      status.textContent = "Draw!";
    } else {
      status.textContent = "Game over - " + (game.in_stalemate() ? "Stalemate" : "Draw");
    }
    return;
  }

  if (thinking) {
    status.textContent = "AI is thinking...";
  } else {
    status.textContent = game.turn() === "w" ? "Your turn (White)" : "AI's turn (Black)";
  }
}

function clearHighlights() {
  $("#board .selected-square").removeClass("selected-square");
  $("#board .target-square").removeClass("target-square");
}

function clearSelection() {
  selectedSquare = null;
  selectedTargets = [];
  clearHighlights();
}

function highlightSelection() {
  clearHighlights();
  if (!selectedSquare) return;

  $("#board .square-" + selectedSquare).addClass("selected-square");
  selectedTargets.forEach(function (target) {
    $("#board .square-" + target).addClass("target-square");
  });
}

function squareFromElement(el) {
  var dataSquare = el.getAttribute("data-square");
  if (dataSquare) return dataSquare;

  var classes = (el.className || "").split(/\s+/);
  for (var i = 0; i < classes.length; i += 1) {
    if (/^square-[a-h][1-8]$/.test(classes[i])) {
      return classes[i].slice(7);
    }
  }
  return null;
}

function submitMove(source, target) {
  var move = game.move({ from: source, to: target, promotion: "q" });
  if (move === null) {
    clearSelection();
    updateStatus();
    return;
  }

  // Show the user's move on the board immediately
  board.position(game.fen());

  // Undo in chess.js - server is authoritative
  game.undo();
  clearSelection();

  thinking = true;
  updateStatus();

  $.ajax({
    url: "/api/move",
    method: "POST",
    contentType: "application/json",
    data: JSON.stringify({ from: source, to: target, promotion: "q" }),
    success: function (data) {
      thinking = false;
      clearSelection();
      game.load(data.fen);
      board.position(data.fen);

      if (data.ai_move) {
        document.getElementById("pieces-polled").textContent =
          data.ai_move.pieces_polled;
        document.getElementById("pieces-volunteered").textContent =
          data.ai_move.pieces_volunteered;

        var chosenPiece = data.ai_move.from
          ? (data.ai_move.reasoning ? data.ai_move.reasoning + " " : "") +
            "(" + data.ai_move.from + " -> " + data.ai_move.to + ")"
          : "";
        var chosenReason = data.ai_move.chosen_piece_reason || "No reasoning provided.";
        document.getElementById("chosen-reason-text").textContent =
          chosenPiece ? chosenReason : "No reasoning provided.";

        var proposals = data.ai_move.proposals || [];
        var details = document.getElementById("proposals-details");
        var list = document.getElementById("proposals-list");
        if (proposals.length > 0) {
          details.style.display = "block";
          list.innerHTML = "";
          proposals.forEach(function (p) {
            var li = document.createElement("li");
            li.textContent =
              p.piece + " " + p.from_square + " -> " + p.target_square +
              " (score: " + p.probability_score + ") - " + p.reason;
            list.appendChild(li);
          });
        } else {
          details.style.display = "none";
        }
      }

      if (data.game_over) {
        var resultText = "";
        if (data.result === "1-0") resultText = "White wins!";
        else if (data.result === "0-1") resultText = "Black wins!";
        else resultText = "Draw!";
        document.getElementById("status").textContent = "Game over - " + resultText;
      } else {
        updateStatus();
      }
    },
    error: function (xhr) {
      thinking = false;
      clearSelection();
      var errMsg = "Error";
      try { errMsg = JSON.parse(xhr.responseText).error; } catch (e) {}
      alert("Move failed: " + errMsg);
      board.position(game.fen());
      updateStatus();
    }
  });
}

function selectPiece(square) {
  selectedSquare = square;
  selectedTargets = game.moves({ square: square, verbose: true }).map(function (m) {
    return m.to;
  });
  highlightSelection();
}

function onSquareClick(square) {
  if (!square || thinking || game.game_over() || game.turn() !== "w") return;

  var piece = game.get(square);

  if (!selectedSquare) {
    if (piece && piece.color === "w") {
      selectPiece(square);
    }
    return;
  }

  if (square === selectedSquare) {
    clearSelection();
    return;
  }

  if (selectedTargets.indexOf(square) !== -1) {
    submitMove(selectedSquare, square);
    return;
  }

  if (piece && piece.color === "w") {
    selectPiece(square);
    return;
  }

  clearSelection();
}

function bindBoardClicks() {
  $("#board").on("click", ".square-55d63", function () {
    var square = squareFromElement(this);
    onSquareClick(square);
  });
}

var config = {
  draggable: false,
  position: "start",
  pieceTheme: "https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png"
};

board = Chessboard("board", config);
bindBoardClicks();

$(window).on("resize", function () {
  board.resize();
});

document.getElementById("newGameBtn").addEventListener("click", function () {
  $.post("/api/new", function () {
    game.reset();
    board.position("start");
    thinking = false;
    clearSelection();
    document.getElementById("pieces-polled").textContent = "-";
    document.getElementById("pieces-volunteered").textContent = "-";
    document.getElementById("chosen-reason-text").textContent = "Waiting for AI move...";
    document.getElementById("proposals-details").style.display = "none";
    updateStatus();
  });
});

updateStatus();
