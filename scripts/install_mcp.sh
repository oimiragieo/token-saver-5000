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

# Step 4: Install via Python helper
echo ""
echo "⚙️  Installing Claude Desktop MCP configuration..."
$PYTHON_CMD -m src.mcp_install
CONFIG_FILE=$($PYTHON_CMD -c "from src.mcp_install import detect_claude_config_path; print(detect_claude_config_path())")

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
echo "  4. Test with: Try saying 'Can you use the token-saver MCP server?'"
echo ""
echo "======================================================================"
echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
