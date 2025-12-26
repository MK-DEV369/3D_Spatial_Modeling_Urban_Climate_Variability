#!/bin/bash
# Frontend setup script

echo "Setting up React frontend..."

cd frontend

# Install dependencies
echo "Installing Node.js dependencies..."
npm install

echo "Frontend setup complete!"
echo "Next steps:"
echo "1. (Optional) Install ReactBits.dev components:"
echo "   npm install -g jsrepo"
echo "   npx jsrepo init https://reactbits.dev/ts/tailwind/"
echo "2. Start development server: npm run dev"

