import React, { useState, useEffect, useCallback } from 'react';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';
import type { Square } from 'react-chessboard/dist/chessboard/types';
import axios from 'axios';
import './App.css';

interface Move {
  move: string;
  score: number;
  san: string;
}

interface Opening {
  id: string;
  name: string;
}

function App() {
  const [game, setGame] = useState(new Chess());
  const [evaluation, setEvaluation] = useState<number>(0);
  const [mateIn, setMateIn] = useState<number | null>(null); // null or number of moves to mate
  const [suggestedMoves, setSuggestedMoves] = useState<Move[]>([]);
  const [moveHistory, setMoveHistory] = useState<string[]>([]);
  const [isPlayerWhite, setIsPlayerWhite] = useState(true);
  const [isAiThinking, setIsAiThinking] = useState(false);
  const [gameMode, setGameMode] = useState<'pvp' | 'pva'>('pva');
  const [customArrows, setCustomArrows] = useState<Array<[Square, Square]>>([]);
  const [showArrows, setShowArrows] = useState(true);
  const [moveIndex, setMoveIndex] = useState(-1); // -1 means current position
  const [gameHistory, setGameHistory] = useState<string[]>([]); // Store all FEN positions
  const [capturedPieces, setCapturedPieces] = useState<{ white: string[], black: string[] }>({ white: [], black: [] });
  const [openings, setOpenings] = useState<Opening[]>([]);
  const [selectedOpening, setSelectedOpening] = useState<string>('');  // '' means default/varied

  // Use relative URL in production (same domain), localhost in development
  const API_URL = process.env.REACT_APP_API_URL || (
    process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:5001/api'
  );

  // Calculate captured pieces
  const calculateCapturedPieces = useCallback((currentGame: Chess) => {
    const pieceValues: { [key: string]: string } = {
      'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛',
      'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕'
    };

    const startingPieces: { [key: string]: number } = {
      'p': 8, 'n': 2, 'b': 2, 'r': 2, 'q': 1,
      'P': 8, 'N': 2, 'B': 2, 'R': 2, 'Q': 1
    };

    const currentPieces: { [key: string]: number } = {
      'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0,
      'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0
    };

    // Count current pieces on board
    const board = currentGame.board();
    board.forEach(row => {
      row.forEach(square => {
        if (square) {
          const piece = square.type;
          const color = square.color;
          const key = color === 'w' ? piece.toUpperCase() : piece.toLowerCase();
          currentPieces[key]++;
        }
      });
    });

    // Calculate captured pieces
    const whiteCaptured: string[] = [];
    const blackCaptured: string[] = [];

    // White pieces captured by black
    (['P', 'N', 'B', 'R', 'Q'] as const).forEach(piece => {
      const captured = startingPieces[piece] - currentPieces[piece];
      for (let i = 0; i < captured; i++) {
        whiteCaptured.push(pieceValues[piece]);
      }
    });

    // Black pieces captured by white
    (['p', 'n', 'b', 'r', 'q'] as const).forEach(piece => {
      const captured = startingPieces[piece] - currentPieces[piece];
      for (let i = 0; i < captured; i++) {
        blackCaptured.push(pieceValues[piece]);
      }
    });

    setCapturedPieces({ white: whiteCaptured, black: blackCaptured });
  }, []);

  // Fetch evaluation for current position
  const fetchEvaluation = useCallback(async (fen: string) => {
    try {
      const response = await axios.post(`${API_URL}/evaluate`, { fen });
      setEvaluation(response.data.evaluation);
      setMateIn(response.data.mate_in || null);
    } catch (error) {
      console.error('Error fetching evaluation:', error);
    }
  }, [API_URL]);

  // Fetch suggested moves
  const fetchSuggestedMoves = useCallback(async (fen: string) => {
    try {
      const response = await axios.post(`${API_URL}/suggested-moves`, {
        fen,
        top_n: 3
      });
      setSuggestedMoves(response.data.moves);

      // Create arrows for top 3 moves (only if showArrows is true)
      if (showArrows) {
        const arrows: Array<[Square, Square]> = response.data.moves.map((move: Move) => {
          const from = move.move.substring(0, 2) as Square;
          const to = move.move.substring(2, 4) as Square;
          return [from, to];
        });
        setCustomArrows(arrows);
      } else {
        setCustomArrows([]);
      }
    } catch (error) {
      console.error('Error fetching suggested moves:', error);
    }
  }, [API_URL, showArrows]);

  // Update evaluation and suggestions when position changes.
  // Skip suggestions while it's the AI's turn — those would be the engine's
  // own candidate moves and would visually telegraph what it's about to play.
  useEffect(() => {
    const fen = game.fen();
    fetchEvaluation(fen);
    calculateCapturedPieces(game);

    const isWhiteTurn = game.turn() === 'w';
    const isAiTurn = gameMode === 'pva' &&
      ((isPlayerWhite && !isWhiteTurn) || (!isPlayerWhite && isWhiteTurn));

    if (!game.isGameOver() && !isAiTurn) {
      fetchSuggestedMoves(fen);
    } else {
      setSuggestedMoves([]);
      setCustomArrows([]);
    }
  }, [game, gameMode, isPlayerWhite, fetchEvaluation, fetchSuggestedMoves, calculateCapturedPieces]);

  // Fetch the list of named openings once on mount.
  useEffect(() => {
    axios.get(`${API_URL}/openings`)
      .then(res => setOpenings(res.data.openings || []))
      .catch(err => console.error('Error fetching openings:', err));
  }, [API_URL]);

  // Make AI move
  const makeAiMove = useCallback(async (currentFen: string) => {
    if (isAiThinking) return;

    console.log('AI making move for position:', currentFen, 'opening:', selectedOpening || '(default)');
    setIsAiThinking(true);
    try {
      const response = await axios.post(`${API_URL}/make-ai-move`, {
        fen: currentFen,
        opening: selectedOpening || undefined,
      }, {
        headers: {
          'Content-Type': 'application/json'
        },
        timeout: 10000
      });

      console.log('AI response:', response.data);
      const newGame = new Chess(response.data.new_fen);
      setGame(newGame);
      setMoveHistory(prev => [...prev, response.data.san]);
      setGameHistory(prev => [...prev, response.data.new_fen]);

      if (response.data.is_game_over) {
        setTimeout(() => {
          alert(`Game Over! Result: ${response.data.result}`);
        }, 100);
      }
    } catch (error: any) {
      console.error('Error making AI move:', error);
      console.error('Error details:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status
      });
      alert(`Failed to get AI move: ${error.message}`);
    } finally {
      setIsAiThinking(false);
    }
  }, [API_URL, isAiThinking, selectedOpening]);

  // Check if it's AI's turn in PvA mode
  useEffect(() => {
    // Only allow AI to move if we're at the current position (not viewing history)
    if (gameMode === 'pva' && !game.isGameOver() && !isAiThinking && moveIndex === -1) {
      const isWhiteTurn = game.turn() === 'w';
      const isAiTurn = (isPlayerWhite && !isWhiteTurn) || (!isPlayerWhite && isWhiteTurn);

      console.log('AI Turn Check:', {
        gameMode,
        isWhiteTurn,
        isPlayerWhite,
        isAiTurn,
        isAiThinking,
        moveIndex,
        currentFen: game.fen()
      });

      if (isAiTurn) {
        console.log('Scheduling AI move...');
        // Delay AI move slightly for better UX
        const timer = setTimeout(() => {
          makeAiMove(game.fen());
        }, 500);
        return () => clearTimeout(timer);
      }
    }
  }, [game, gameMode, isPlayerWhite, isAiThinking, makeAiMove, moveIndex]);

  // Handle piece drop
  function onDrop(sourceSquare: string, targetSquare: string) {
    try {
      // If we're viewing history, don't allow moves
      if (moveIndex !== -1) return false;

      const gameCopy = new Chess(game.fen());

      // Check if move is legal
      const move = gameCopy.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q', // Always promote to queen for simplicity
      });

      if (move === null) return false;

      setGame(gameCopy);
      setMoveHistory(prev => [...prev, move.san]);
      setGameHistory(prev => [...prev, gameCopy.fen()]);
      setMoveIndex(-1); // Reset to current position

      if (gameCopy.isGameOver()) {
        setTimeout(() => {
          let result = 'Draw';
          if (gameCopy.isCheckmate()) {
            result = gameCopy.turn() === 'w' ? 'Black wins!' : 'White wins!';
          } else if (gameCopy.isStalemate()) {
            result = 'Stalemate';
          } else if (gameCopy.isThreefoldRepetition()) {
            result = 'Draw by repetition';
          } else if (gameCopy.isInsufficientMaterial()) {
            result = 'Draw by insufficient material';
          }
          alert(`Game Over! ${result}`);
        }, 100);
      }

      return true;
    } catch (error) {
      return false;
    }
  }

  // Reset game
  function resetGame() {
    const newGame = new Chess();
    setGame(newGame);
    setMoveHistory([]);
    setGameHistory([]);
    setMoveIndex(-1);
    setEvaluation(0);
    setSuggestedMoves([]);
    setCustomArrows([]);
  }

  // Undo last move - properly reconstruct game state
  function undoMove() {
    if (moveHistory.length === 0) return;

    // If we're viewing history, first return to current position
    if (moveIndex !== -1) {
      setMoveIndex(-1);
      // Reconstruct current game state
      const currentGame = new Chess();
      moveHistory.forEach((san) => {
        try {
          currentGame.move(san);
        } catch (e) {
          console.error('Failed to apply move:', san, e);
        }
      });
      setGame(currentGame);
    }

    // Now undo the last move
    const newMoveHistory = moveHistory.slice(0, -1);
    const newGameHistory = gameHistory.slice(0, -1);

    // Reconstruct game from scratch using SAN notation
    const newGame = new Chess();
    newMoveHistory.forEach((san) => {
      try {
        newGame.move(san);
      } catch (e) {
        console.error('Failed to apply move:', san, e);
      }
    });

    setGame(newGame);
    setMoveHistory(newMoveHistory);
    setGameHistory(newGameHistory);
    setMoveIndex(-1);
  }

  // Navigate through move history
  const goToMove = useCallback((index: number) => {
    if (index === -1) {
      // Go to current position
      const newGame = new Chess();
      moveHistory.forEach((san) => {
        try {
          newGame.move(san);
        } catch (e) {
          console.error('Failed to apply move:', san, e);
        }
      });
      setGame(newGame);
      setMoveIndex(-1);
    } else if (index >= 0 && index < gameHistory.length) {
      const fen = gameHistory[index];
      setGame(new Chess(fen));
      setMoveIndex(index);
    }
  }, [moveHistory, gameHistory]);

  // Keyboard navigation - navigate by individual moves (half-moves)
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') {
        // Go back one move (one half-move)
        if (moveIndex === -1 && gameHistory.length > 0) {
          // From current position, go back one move
          goToMove(gameHistory.length - 2 >= 0 ? gameHistory.length - 2 : gameHistory.length - 1);
        } else if (moveIndex > 0) {
          // Go back one move
          goToMove(moveIndex - 1);
        } else if (moveIndex === 0) {
          // At first move, go to starting position
          const newGame = new Chess();
          setGame(newGame);
          setMoveIndex(-2);
        }
        // If moveIndex === -2 (at starting position), do nothing
      } else if (e.key === 'ArrowRight') {
        // Go forward one move (one half-move)
        if (moveIndex === -2) {
          // From starting position to first move
          if (gameHistory.length > 0) {
            goToMove(0);
          }
        } else if (moveIndex >= 0 && moveIndex < gameHistory.length - 1) {
          // Go forward one move
          goToMove(moveIndex + 1);
        } else if (moveIndex === gameHistory.length - 1) {
          // At last historical position, go to current
          goToMove(-1);
        }
        // If moveIndex === -1 (at current position), do nothing
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [moveIndex, gameHistory, goToMove]);

  // Get evaluation bar percentage
  function getEvaluationPercentage(): number {
    // In Stockfish style: positive = white advantage, negative = black advantage
    // For mate positions, treat as maximum advantage
    if (mateIn !== null) {
      return mateIn > 0 ? 100 : 0; // Positive mate = white winning, negative = black winning
    }
    // Convert evaluation to percentage (clamped between -10 and +10 pawns)
    const clampedEval = Math.max(-10, Math.min(10, evaluation));
    return ((clampedEval + 10) / 20) * 100;
  }

  // Get evaluation color - The bar itself should always be white (representing white's territory)
  // The dark background represents black's territory
  function getEvaluationColor(): string {
    return '#ffffff'; // Always white - the bar represents white's advantage area
  }

  // Get evaluation display text
  function getEvaluationText(): string {
    if (mateIn !== null) {
      if (mateIn > 0) {
        return `M${mateIn}`; // Mate for white
      } else {
        return `M${Math.abs(mateIn)}`; // Mate for black (show as positive number)
      }
    }
    return `${evaluation > 0 ? '+' : ''}${evaluation.toFixed(2)}`;
  }

  // Get custom square styles for check highlighting and checkmate
  function getCustomSquareStyles() {
    const styles: { [square: string]: React.CSSProperties } = {};

    // Highlight king in check or checkmate (same color for both)
    if (game.isCheck() || game.isCheckmate()) {
      // The king in check/checkmate is the one whose turn it is
      const kingColor = game.turn();
      const board = game.board();

      // Find the king of the current turn (the one in check)
      for (let rank = 0; rank < 8; rank++) {
        for (let file = 0; file < 8; file++) {
          const piece = board[rank][file];
          if (piece && piece.type === 'k' && piece.color === kingColor) {
            // Calculate square correctly: files are a-h (columns), ranks are 1-8 (rows)
            const fileChar = String.fromCharCode(97 + file); // 0->a, 1->b, etc.
            const rankNum = 8 - rank; // chess.js board: rank 0 = 8, rank 7 = 1
            const square = `${fileChar}${rankNum}` as Square;
            styles[square] = { backgroundColor: 'rgba(255, 0, 0, 0.4)' };
            return styles;
          }
        }
      }
    }

    return styles;
  }

  return (
    <div className="App">
      <div className="chess-container">
        <div className="sidebar left-sidebar">
          <h2>Chess AI</h2>

          <div className="evaluation-container">
            <h3>Position Evaluation</h3>
            <div className="eval-bar-container">
              <div
                className="eval-bar"
                style={{
                  height: `${getEvaluationPercentage()}%`,
                  backgroundColor: getEvaluationColor()
                }}
              />
            </div>
            <div className="eval-value">
              {getEvaluationText()}
            </div>
            <div className="eval-labels">
              <span>Black</span>
              <span>White</span>
            </div>
          </div>

          <div className="game-controls">
            <h3>Game Controls</h3>
            <div className="control-group">
              <label>Mode:</label>
              <select
                value={gameMode}
                onChange={(e) => setGameMode(e.target.value as 'pvp' | 'pva')}
              >
                <option value="pva">Player vs AI</option>
                <option value="pvp">Player vs Player</option>
              </select>
            </div>

            {gameMode === 'pva' && (
              <div className="control-group">
                <label>Play as:</label>
                <select
                  value={isPlayerWhite ? 'white' : 'black'}
                  onChange={(e) => {
                    setIsPlayerWhite(e.target.value === 'white');
                    resetGame();
                  }}
                >
                  <option value="white">White</option>
                  <option value="black">Black</option>
                </select>
              </div>
            )}

            {gameMode === 'pva' && openings.length > 0 && (
              <div className="control-group">
                <label>Bot opening:</label>
                <select
                  value={selectedOpening}
                  onChange={(e) => {
                    setSelectedOpening(e.target.value);
                    resetGame();
                  }}
                >
                  <option value="">Default (varied)</option>
                  {openings.map(o => (
                    <option key={o.id} value={o.id}>{o.name}</option>
                  ))}
                </select>
              </div>
            )}

            <button onClick={resetGame} className="btn btn-primary">
              New Game
            </button>
            <button onClick={undoMove} className="btn btn-secondary">
              Undo Move
            </button>
            <button
              onClick={() => setShowArrows(!showArrows)}
              className={`btn ${showArrows ? 'btn-primary' : 'btn-secondary'}`}
            >
              {showArrows ? 'Hide Arrows' : 'Show Arrows'}
            </button>
          </div>
        </div>

        <div className="board-container">
          <div className="captured-pieces-top">
            {capturedPieces.white.length > 0 ? capturedPieces.white.join(' ') : ''}
          </div>

          <Chessboard
            position={game.fen()}
            onPieceDrop={onDrop}
            boardOrientation={isPlayerWhite ? 'white' : 'black'}
            customArrows={showArrows ? customArrows : []}
            customSquareStyles={getCustomSquareStyles()}
            customBoardStyle={{
              borderRadius: '8px',
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.3)'
            }}
            customDarkSquareStyle={{ backgroundColor: '#779952' }}
            customLightSquareStyle={{ backgroundColor: '#edeed1' }}
          />

          <div className="captured-pieces-bottom">
            {capturedPieces.black.length > 0 ? capturedPieces.black.join(' ') : ''}
          </div>

          <div className="turn-indicator">
            <div className={`turn ${game.turn() === 'w' ? 'active' : ''}`}>
              White to move
            </div>
            <div className={`turn ${game.turn() === 'b' ? 'active' : ''}`}>
              Black to move
            </div>
          </div>
        </div>

        <div className="sidebar right-sidebar">
          <div className="suggested-moves">
            <h3>Suggested Moves</h3>
            {isAiThinking ? (
              <div className="thinking">AI is thinking...</div>
            ) : (
              <ul>
                {suggestedMoves.map((move, idx) => (
                  <li key={idx} className={`move-suggestion rank-${idx + 1}`}>
                    <span className="move-number">{idx + 1}</span>
                    <span className="move-san">{move.san}</span>
                    <span className="move-eval">
                      {move.score > 0 ? '+' : ''}{move.score.toFixed(2)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <h3>Move History</h3>
          <div className="move-history">
            {moveHistory.length === 0 ? (
              <div className="no-moves">No moves yet</div>
            ) : (
              <ol>
                {moveHistory.map((move, idx) => (
                  <li key={idx}>
                    {Math.floor(idx / 2) + 1}.
                    {idx % 2 === 0 && ' '}
                    {move}
                  </li>
                ))}
              </ol>
            )}
          </div>

          <div className="game-info">
            <h3>Game Info</h3>
            <div className="info-item">
              <span className="label">Moves:</span>
              <span className="value">{moveHistory.length}</span>
            </div>
            <div className="info-item">
              <span className="label">Turn:</span>
              <span className="value">{game.turn() === 'w' ? 'White' : 'Black'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
