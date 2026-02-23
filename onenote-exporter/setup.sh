#!/bin/bash
set -e

echo "🚀 OneNote Exporter Setup"
echo "========================="

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p output cache

# Copy environment template if .env doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file - please edit it with your Azure app details"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file with your TENANT_ID and CLIENT_ID"
    echo "2. Run: docker compose build"
    echo "3. Run: docker compose run --rm onenote-exporter --list"
    echo ""
    echo "For Azure app setup instructions, see README.md"
else
    echo "✅ .env file already exists"
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker compose build

echo ""
echo "🎉 Setup complete! Your OneNote Exporter is ready to use."
echo ""
echo "Quick start:"
echo "  docker compose run --rm onenote-exporter --list"
echo "  docker compose run --rm onenote-exporter --notebook \"My Notes\" --merge"
echo ""
echo "For more usage examples, see README.md"
