#!/bin/bash
# Token Saver 5000 - Automated MCP Installation Script
# This script automates the Claude Desktop MCP configuration

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "  Token Saver 5000 - MCP Installation Script"
echo "======================================================================"
echo ""

# Step 1: Detect installation directory
INSTALL_DIR=$(pwd)
echo "📂 Installation directory: $INSTALL_DIR"

# Step 2: Verify we're in the right directory
if [ ! -f "$INSTALL_DIR/src/server.py" ]; then
    echo -e "${RED}❌ Error: src/server.py not found${NC}"
    echo "   Please run this script from the token-saver-5000 root directory"
    exit 1
fi
echo -e "${GREEN}✅ Found src/server.py${NC}"

# Step 3: Check Python version
echo ""
echo "🐍 Checking Python version..."
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Error: Python not found${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "   Found: $PYTHON_VERSION"

# Verify Python >= 3.10
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}❌ Error: Python 3.10+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python version OK${NC}"

# Step 4: Detect Claude Desktop config location
echo ""
echo "🔍 Detecting Claude Desktop configuration location..."

if [ "$(uname)" == "Darwin" ]; then
    # macOS
    CONFIG_FILE="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Linux
    CONFIG_FILE="$HOME/.config/claude/claude_desktop_config.json"
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ] || [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
    # Windows (Git Bash)
    CONFIG_FILE="$APPDATA/Claude/claude_desktop_config.json"
else
    echo -e "${YELLOW}⚠️  Unknown OS, using default location${NC}"
    CONFIG_FILE="$HOME/.config/claude/claude_desktop_config.json"
fi

echo "   Config location: $CONFIG_FILE"

# Step 5: Create config directory if needed
CONFIG_DIR=$(dirname "$CONFIG_FILE")
if [ ! -d "$CONFIG_DIR" ]; then
    echo "   Creating config directory: $CONFIG_DIR"
    mkdir -p "$CONFIG_DIR"
fi

# Step 6: Generate MCP config
echo ""
echo "⚙️  Generating MCP configuration..."

MCP_CONFIG=$(cat <<EOF
{
  "mcpServers": {
    "semantic-modulator": {
      "command": "$PYTHON_CMD",
      "args": ["-m", "src.server"],
      "cwd": "$INSTALL_DIR",
      "env": {
        "PYTHONPATH": "$INSTALL_DIR"
      }
    }
  }
}
EOF
)

# Step 7: Merge with existing config or create new
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}⚠️  Existing config found${NC}"
    echo "   Backing up to: ${CONFIG_FILE}.backup"
    cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"

    # Check if semantic-modulator already exists
    if grep -q "semantic-modulator" "$CONFIG_FILE"; then
        echo -e "${YELLOW}⚠️  semantic-modulator already configured${NC}"
        echo "   Updating configuration..."

        # Use Python to merge JSON (safer than manual editing)
        $PYTHON_CMD -c "
import json
import sys

# Read existing config
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

# Update semantic-modulator
if 'mcpServers' not in config:
    config['mcpServers'] = {}

config['mcpServers']['semantic-modulator'] = {
    'command': '$PYTHON_CMD',
    'args': ['-m', 'src.server'],
    'cwd': '$INSTALL_DIR',
    'env': {
        'PYTHONPATH': '$INSTALL_DIR'
    }
}

# Write back
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)

print('✅ Configuration updated')
"
    else
        echo "   Adding semantic-modulator to existing config..."

        # Use Python to merge
        $PYTHON_CMD -c "
import json

# Read existing config
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

# Add semantic-modulator
if 'mcpServers' not in config:
    config['mcpServers'] = {}

config['mcpServers']['semantic-modulator'] = {
    'command': '$PYTHON_CMD',
    'args': ['-m', 'src.server'],
    'cwd': '$INSTALL_DIR',
    'env': {
        'PYTHONPATH': '$INSTALL_DIR'
    }
}

# Write back
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)

print('✅ Configuration merged')
"
    fi
else
    echo "   Creating new config file..."
    echo "$MCP_CONFIG" > "$CONFIG_FILE"
    echo -e "${GREEN}✅ Configuration created${NC}"
fi

# Step 8: Verify config
echo ""
echo "✅ MCP configuration complete!"
echo ""
echo "======================================================================"
echo "  Installation Summary"
echo "======================================================================"
echo "Installation directory: $INSTALL_DIR"
echo "Python command:         $PYTHON_CMD"
echo "Config file:            $CONFIG_FILE"
echo ""
echo "Next steps:"
echo "  1. Install dependencies: pip install -r requirements.txt"
echo "  2. Run setup check:      $PYTHON_CMD check_setup.py"
echo "  3. Restart Claude Desktop"
echo "  4. Test with: Try saying 'Can you use the semantic-modulator MCP server?'"
echo ""
echo "======================================================================"
echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
