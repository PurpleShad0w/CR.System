@echo off
echo 🚀 OneNote Exporter Setup
echo =========================

REM Create necessary directories
echo 📁 Creating directories...
if not exist output mkdir output
if not exist cache mkdir cache

REM Copy environment template if .env doesn't exist
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env >nul
    echo ✅ Created .env file - please edit it with your Azure app details
    echo.
    echo Next steps:
    echo 1. Edit .env file with your TENANT_ID and CLIENT_ID
    echo 2. Run: docker compose build
    echo 3. Run: docker compose run --rm onenote-exporter --list
    echo.
    echo For Azure app setup instructions, see README.md
) else (
    echo ✅ .env file already exists
)

REM Build the Docker image
echo 🔨 Building Docker image...
docker compose build

echo.
echo 🎉 Setup complete! Your OneNote Exporter is ready to use.
echo.
echo Quick start:
echo   docker compose run --rm onenote-exporter --list
echo   docker compose run --rm onenote-exporter --notebook "My Notes" --merge
echo.
echo For more usage examples, see README.md
pause
