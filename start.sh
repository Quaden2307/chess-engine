#!/bin/bash

# Chess AI Startup Script

echo "🚀 Starting Chess AI Application..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Install Python dependencies if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "📦 Activating virtual environment..."
source venv/bin/activate

echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt

# Install frontend dependencies if needed
if [ ! -d "chess-frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd chess-frontend
    npm install
    cd ..
fi

# Start backend in background
echo "🖥️  Starting Flask backend on port 5000..."
python backend.py > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 3

# Check if backend is running
if ! curl -s http://localhost:5000/api/health > /dev/null; then
    echo "⚠️  Backend might not be ready yet. Check backend.log for errors."
else
    echo "✅ Backend is running!"
fi

# Start frontend
echo "🌐 Starting React frontend on port 3000..."
cd chess-frontend
npm start &
FRONTEND_PID=$!

echo ""
echo "✅ Chess AI is starting!"
echo ""
echo "🎮 Frontend: http://localhost:3000"
echo "🔌 Backend API: http://localhost:5000"
echo ""
echo "📝 To stop the application:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "   Or press Ctrl+C and run: ./stop.sh"
echo ""

# Save PIDs for cleanup
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

# Wait for processes
wait
