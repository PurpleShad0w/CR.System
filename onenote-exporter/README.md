# OneNote Exporter

A complete, Docker-ready tool to export Microsoft OneNote notebooks via Microsoft Graph API into clean Markdown with embedded assets, optional compiled DOCX, and JSONL for LLM ingestion.

**🚀 Zero-dependency setup**: Only requires Docker - everything else is handled automatically!

## Features

- **One-command setup**: Clone, run setup script, start exporting
- **Docker-first**: No Python, dependencies, or complex setup required
- **Authentication**: Device-code flow via MSAL with token caching
- **Notebook Discovery**: List and select notebooks by name or ID
- **Complete Export**: Every page to clean Markdown with downloaded images/attachments
- **Compiled Output**: Single Markdown file + optional DOCX (via pandoc)
- **LLM Ready**: JSONL format (one record per page) for RAG/vector search
- **Cross-Platform**: Works on Windows/macOS/Linux with persistent cache
- **Progress Tracking**: Visual progress bars for large notebooks

## How it works

Uses Microsoft Graph OneNote API through direct REST calls (msal + requests). OneNote HTML is parsed and converted to clean Markdown (beautifulsoup4 + markdownify). Images and attachments are downloaded locally and relinked.

**Important limitations:**

- Only cloud-synced notebooks (Microsoft 365/OneDrive/Personal OneDrive)
- Local-only notebooks are not accessible via Graph API
- Work/school tenants may require admin consent for Graph permissions

## Prerequisites

### Azure App Registration

1. Go to [Azure Portal](https://portal.azure.com) → App registrations → New registration
2. **Name**: OneNote Exporter (or any name)
3. **Account types**: Choose as needed (single/multi-tenant)
4. **Redirect URI**: Leave blank (not needed for device code)
5. **API permissions** (delegated):
   - `Notes.Read` (required)
   - `offline_access` (required for token refresh)
   - `User.Read` (optional, often auto-requested)
   - `Files.Read` (optional, if issues fetching resources)
6. Grant admin consent if required by your organization
7. Note your **Tenant ID** and **Application (client) ID**

### Environment Setup

Create a `.env` file in the project root:

```bash
TENANT_ID=your-tenant-id-or-common
CLIENT_ID=your-app-client-id
ADDITIONAL_SCOPES=
```

## Quick Start

### One-Command Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/username/onenote-exporter.git
cd onenote-exporter

# Choose your container runtime:

# Option A: Docker (recommended)
./setup.sh              # Linux/macOS
# OR
setup.bat               # Windows

# Option B: Podman (rootless alternative)
./podman/setup-podman.sh              # Linux/macOS
# OR
podman\setup-podman.bat               # Windows

# Edit .env file with your Azure app details
# Then start using:
docker compose run --rm onenote-exporter --list
# OR
podman-compose -f podman/podman-compose.yml run --rm onenote-exporter --list
```

### Manual Setup

If you prefer to set up manually:

```bash
# 1. Clone repository
git clone https://github.com/username/onenote-exporter.git
cd onenote-exporter

# 2. Create directories and environment file
mkdir -p output cache
cp .env.example .env

# 3. Edit .env with your Azure app credentials
# TENANT_ID=your-tenant-id
# CLIENT_ID=your-client-id

# 4. Build Docker image
docker compose build

# 5. Start using
docker compose run --rm onenote-exporter --list
```

### Docker Compose Commands

```bash
# List available notebooks
docker compose run --rm onenote-exporter --list

# Export a specific notebook
docker compose run --rm onenote-exporter \
  --notebook "My Novel Notes" \
  --merge \
  --formats md,docx

# Export by notebook ID (exact match)
docker compose run --rm onenote-exporter \
  --notebook-id "1-abc123..." \
  --merge
```

### Alternative: Direct Docker Run

If you prefer not to use docker-compose:

```bash
# Build image
docker build -t onenote-exporter .

# Run with environment variables
docker run --rm -it \
  -e TENANT_ID=your-tenant-id \
  -e CLIENT_ID=your-client-id \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/cache:/app/.cache" \
  onenote-exporter --notebook "My Notes" --merge
```

## Usage

### Command-Line Options

```bash
onenote-exporter [OPTIONS]

Options:
  --list                  List available notebooks
  --notebook TEXT         Notebook name (partial match)
  --notebook-id TEXT      Exact notebook ID
  --merge                Enable merged output
  --formats TEXT         Output formats: md,docx,jsonl (default: md)
  --config FILE          Path to config file (default: .env)
  --output-dir DIR       Output directory (default: ./output)
  --token-cache FILE     Token cache path (default: ./.cache)
  --help                 Show this message and exit
```

### Output Structure

```text
output/
├── my-notebook-slug/          # Individual page exports
│   ├── page-1.md
│   ├── page-2.md
│   ├── assets/                # Downloaded images/attachments
│   │   ├── image1.png
│   │   └── attachment.pdf
│   ├── merged.md              # Complete notebook (if --merge)
│   ├── merged.docx            # DOCX version (if requested)
│   └── merged.jsonl           # JSONL for LLM ingestion
```

### Output Formats

- **Markdown (.md)**: Clean formatted text with local asset links
- **DOCX (.docx)**: Compiled document via pandoc (requires --merge)
- **JSONL (.jsonl)**: One JSON record per page for RAG/vector databases

## Container Runtime Options

This project supports both Docker and Podman for maximum flexibility:

### Docker (Default)

- **Standard**: Most widely supported
- **Cross-platform**: Works on all major platforms
- **Easy setup**: Single daemon, well-documented
- **Files**: `docker-compose.yml`, `setup.sh`, `setup.bat`

### Podman (Alternative)

- **Rootless**: Better security, no root required
- **Daemonless**: No background process needed
- **Lightweight**: Lower resource usage
- **Compatible**: Drop-in Docker replacement
- **Files**: `podman/` directory with dedicated setup

Choose the runtime that best fits your security and infrastructure requirements.

## Development

### Local Environment Setup

```bash
# Clone and setup
git clone <repo-url>
cd onenote-exporter

# Setup development environment
bash scripts/setup_venv.sh
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Development Scripts

```bash
# Format code
bash scripts/format.sh

# Run tests
pytest tests/

# Development mode
bash scripts/dev.sh
```

### Project Structure

```text
src/onenote_exporter/
├── __init__.py         # Package exports
├── auth.py            # MSAL authentication
├── graph.py           # Microsoft Graph API client
├── exporter.py        # Export logic and conversion
├── cli.py             # Command-line interface
├── config.py          # Configuration management
└── utils.py           # Shared utilities

tests/                 # Unit tests
scripts/              # Development scripts
docs/                 # Documentation
.github/workflows/    # CI/CD pipeline
```

## Authentication Details

The tool uses MSAL device code flow:

1. **First run**: Opens browser for Microsoft login
2. **Token caching**: Stores refresh tokens locally (`.cache/` directory)
3. **Automatic refresh**: Handles token renewal transparently
4. **Cross-platform**: Works in Docker containers and local environments

**Required Azure AD permissions:**

- `Notes.Read` - Access OneNote data
- `offline_access` - Token refresh capability

## Troubleshooting

### Common Issues

#### No notebooks found

- Ensure notebooks are synced to OneDrive/Microsoft 365
- Local-only notebooks are not accessible via Graph API
- Check Azure app permissions

#### Authentication failed

- Verify TENANT_ID and CLIENT_ID in `.env`
- Check if admin consent is required for your organization
- Clear token cache: `rm -rf .cache/`

#### Permission denied

- Organization may require admin consent for Graph permissions
- Contact IT admin to grant consent for the Azure app

#### Docker issues on Windows

- Ensure Docker Desktop is running
- Use PowerShell or WSL2 for best compatibility
- Check volume mounting permissions

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# Run with debug output
docker compose run --rm onenote-exporter --notebook "Test" --merge
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run format and tests: `bash scripts/format.sh && pytest`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
