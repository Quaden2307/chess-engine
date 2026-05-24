#!/bin/bash

# Vercel build script
echo "Building frontend..."
cd chess-frontend
npm install
npm run build
cd ..
echo "Build complete!"
