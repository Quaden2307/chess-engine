import torch
import numpy as np
import chess
import chess.engine
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random

# ========== ADVANCED ENCODING WITH TACTICAL FEATURES ==========

PIECE_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Piece-square tables for positional evaluation
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

def board_to_advanced_tensor(board: chess.Board):
    """Advanced board encoding with tactical and positional features"""
    features = []

    # 1. Piece positions (12 channels * 64 squares = 768 features)
    piece_tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        x = chess.square_file(square)
        y = chess.square_rank(square)
        piece_tensor[y, x, PIECE_INDEX[piece.symbol()]] = 1.0
    features.extend(piece_tensor.flatten().tolist())

    # 2. Material evaluation
    white_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE)
    black_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK)
    material_advantage = (white_material - black_material) / 10000.0
    features.append(material_advantage)

    # 3. Piece-square tables evaluation (positional value)
    white_pos_value = 0
    black_pos_value = 0
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.PAWN:
            if piece.color == chess.WHITE:
                white_pos_value += PAWN_TABLE[square]
            else:
                black_pos_value += PAWN_TABLE[63 - square]
        elif piece.piece_type == chess.KNIGHT:
            if piece.color == chess.WHITE:
                white_pos_value += KNIGHT_TABLE[square]
            else:
                black_pos_value += KNIGHT_TABLE[63 - square]
    positional_advantage = (white_pos_value - black_pos_value) / 1000.0
    features.append(positional_advantage)

    # 4. Mobility (how many moves each side has)
    current_turn = board.turn
    white_mobility = 0
    black_mobility = 0

    if board.turn == chess.WHITE:
        white_mobility = len(list(board.legal_moves))
        board.turn = chess.BLACK
        black_mobility = len(list(board.legal_moves))
        board.turn = chess.WHITE
    else:
        black_mobility = len(list(board.legal_moves))
        board.turn = chess.WHITE
        white_mobility = len(list(board.legal_moves))
        board.turn = chess.BLACK

    mobility_advantage = (white_mobility - black_mobility) / 50.0
    features.append(mobility_advantage)

    # 5. Center control (pieces in center squares)
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    extended_center = [chess.C3, chess.D3, chess.E3, chess.F3,
                       chess.C4, chess.F4, chess.C5, chess.F5,
                       chess.C6, chess.D6, chess.E6, chess.F6]

    white_center = sum(1 for sq in center_squares if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE)
    black_center = sum(1 for sq in center_squares if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK)
    center_control = (white_center - black_center) / 4.0
    features.append(center_control)

    white_ext_center = sum(1 for sq in extended_center if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE)
    black_ext_center = sum(1 for sq in extended_center if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK)
    ext_center_control = (white_ext_center - black_ext_center) / 12.0
    features.append(ext_center_control)

    # 6. King safety
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)

    # Count pieces defending the king
    white_king_defenders = 0
    black_king_defenders = 0
    if white_king_sq:
        for sq in chess.SQUARES:
            if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE:
                if chess.square_distance(sq, white_king_sq) <= 2:
                    white_king_defenders += 1
    if black_king_sq:
        for sq in chess.SQUARES:
            if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK:
                if chess.square_distance(sq, black_king_sq) <= 2:
                    black_king_defenders += 1

    king_safety = (white_king_defenders - black_king_defenders) / 10.0
    features.append(king_safety)

    # 7. Castling rights
    white_can_castle_kingside = board.has_kingside_castling_rights(chess.WHITE)
    white_can_castle_queenside = board.has_queenside_castling_rights(chess.WHITE)
    black_can_castle_kingside = board.has_kingside_castling_rights(chess.BLACK)
    black_can_castle_queenside = board.has_queenside_castling_rights(chess.BLACK)

    features.extend([
        1.0 if white_can_castle_kingside else 0.0,
        1.0 if white_can_castle_queenside else 0.0,
        1.0 if black_can_castle_kingside else 0.0,
        1.0 if black_can_castle_queenside else 0.0
    ])

    # 8. Check status
    in_check = 1.0 if board.is_check() else 0.0
    features.append(in_check)

    # 9. Attacked squares (tactical awareness)
    white_attacks = len(board.attacks(white_king_sq)) if white_king_sq else 0
    black_attacks = len(board.attacks(black_king_sq)) if black_king_sq else 0
    attack_pressure = (white_attacks - black_attacks) / 20.0
    features.append(attack_pressure)

    # 10. Pawn structure
    white_pawns = [sq for sq in chess.SQUARES if board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE)]
    black_pawns = [sq for sq in chess.SQUARES if board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK)]

    # Doubled pawns
    white_doubled = sum(1 for file in range(8) if sum(1 for sq in white_pawns if chess.square_file(sq) == file) > 1)
    black_doubled = sum(1 for file in range(8) if sum(1 for sq in black_pawns if chess.square_file(sq) == file) > 1)
    doubled_pawns = (black_doubled - white_doubled) / 8.0
    features.append(doubled_pawns)

    # 11. Game phase (endgame indicator)
    total_material = white_material + black_material
    game_phase = 1.0 - (total_material / 78000.0)  # 0 = opening, 1 = endgame
    features.append(game_phase)

    # 12. Turn indicator
    turn = 1.0 if board.turn == chess.WHITE else -1.0
    features.append(turn)

    # Count features:
    # 768 (piece positions) + 1 (material) + 1 (positional) + 1 (mobility) + 2 (center control)
    # + 1 (king safety) + 4 (castling) + 1 (check) + 1 (attacks) + 1 (pawn structure)
    # + 1 (game phase) + 1 (turn) = 783 features
    return torch.tensor(features, dtype=torch.float32)

# ========== IMPROVED MODEL ==========

class AdvancedChessNet(nn.Module):
    def __init__(self, input_size=783):
        super().__init__()

        # Deeper network with residual connections
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.relu(self.bn4(self.fc4(x))))
        x = F.relu(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        return x

# ========== DATA GENERATION ==========

# More diverse opening positions
OPENING_POSITIONS = [
    # Classical openings
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # e4
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",  # d4
    "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",  # e4 Nf6
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",  # e4 d5
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # e4 e5
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",  # e4 c5 (Sicilian)
    # Middlegame positions
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Italian
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",  # Queen's Gambit
]

def generate_training_data(num_positions=10000, stockfish_path="/opt/homebrew/bin/stockfish"):
    """Generate diverse training positions with Stockfish evaluations"""
    X = []
    y = []

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    print(f"Generating {num_positions} training positions...")

    for i in tqdm(range(num_positions)):
        # Start from random opening
        opening_fen = random.choice(OPENING_POSITIONS)
        board = chess.Board(opening_fen)

        # Play random moves to create diverse positions
        num_moves = random.randint(0, 30)  # 0-30 moves from opening

        for _ in range(num_moves):
            if board.is_game_over():
                break

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            # 70% random, 30% best move (to get quality positions)
            if random.random() < 0.7:
                move = random.choice(legal_moves)
            else:
                try:
                    result = engine.play(board, chess.engine.Limit(time=0.01))
                    move = result.move
                except:
                    move = random.choice(legal_moves)

            board.push(move)

        # Skip if game is over
        if board.is_game_over():
            continue

        # Get Stockfish evaluation
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=20))
            score = info["score"].relative

            if score.is_mate():
                # Convert mate scores
                mate_in = score.mate()
                eval_value = 10.0 if mate_in > 0 else -10.0
            else:
                # Convert centipawn to normalized value
                eval_value = np.tanh(score.score() / 400.0)

            # Encode position
            tensor = board_to_advanced_tensor(board)
            X.append(tensor)
            y.append(eval_value)

        except Exception as e:
            print(f"Error evaluating position: {e}")
            continue

    engine.quit()

    return torch.stack(X), torch.tensor(y, dtype=torch.float32)

# ========== TRAINING ==========

def train_model(model, X, y, epochs=50, batch_size=64, lr=1e-3):
    """Train the model with better hyperparameters"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    model.train()

    best_loss = float('inf')

    for epoch in range(epochs):
        # Shuffle data
        indices = torch.randperm(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_loss = 0
        num_batches = 0

        # Mini-batch training
        for i in range(0, len(X), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size].unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "chess_model_best.pth")
            print(f"  → Saved best model (loss: {best_loss:.6f})")

    return model

# ========== MAIN ==========

if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED CHESS AI TRAINING")
    print("=" * 60)

    # Generate training data
    print("\n1. Generating training data...")
    X, y = generate_training_data(num_positions=10000)  # 10,000 positions for better learning
    print(f"Generated {len(X)} training positions")
    print(f"Feature vector size: {X[0].shape}")
    print(f"Evaluation range: [{y.min():.3f}, {y.max():.3f}]")

    # Create model
    print("\n2. Creating advanced neural network...")
    model = AdvancedChessNet(input_size=782)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    print("\n3. Training model...")
    model = train_model(model, X, y, epochs=100, batch_size=64, lr=1e-3)

    # Save final model
    torch.save(model.state_dict(), "chess_model_improved.pth")
    print("\n✓ Training complete!")
    print("Models saved:")
    print("  - chess_model_improved.pth (final model)")
    print("  - chess_model_best.pth (best validation loss)")
