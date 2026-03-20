#!/usr/bin/env python3
"""
Token Saver 5000 - Quickstart Script

One-command setup and demo for new users.
Automates the entire setup process and runs a demonstration.

Usage:
    python scripts/quickstart.py

This script will:
1. Check Python version (>= 3.10)
2. Install all dependencies
3. Download embedding model (~80MB on first run)
4. Run verification tests
5. Run interactive demo
"""

import subprocess
import sys


def print_header(message):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {message}")
    print("=" * 70 + "\n")


def print_step(step_num, total_steps, message):
    """Print step indicator"""
    print(f"[{step_num}/{total_steps}] {message}")


def check_python_version():
    """Check if Python version is sufficient"""
    version = sys.version_info
    if version < (3, 10):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   Token Saver 5000 requires Python 3.10 or higher")
        print("\n💡 Install Python 3.10+:")
        print("   • macOS: brew install python@3.11")
        print("   • Linux: sudo apt install python3.11")
        print("   • Windows: https://www.python.org/downloads/")
        return False
    print(f"✅ Python {version.major}.{version.minor} is supported\n")
    return True


def install_dependencies():
    """Install project dependencies"""
    print("Installing dependencies (this may take a few minutes)...")
    print("📦 Installing: mcp, sentence-transformers, networkx, torch, and more...\n")

    try:
        # Install quietly to reduce noise
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"❌ Installation failed:\n{result.stderr}")
            return False

        print("✅ All dependencies installed successfully\n")
        return True

    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False


def download_embedding_model():
    """Download embedding model with progress feedback"""
    print("Downloading embedding model (~80MB, one-time download)...")
    print("💡 This model enables local semantic search (no API calls needed)\n")

    try:
        # Import here after dependencies are installed
        from sentence_transformers import SentenceTransformer

        print("  Downloading from HuggingFace...", end=" ", flush=True)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅")

        # Quick test
        print("  Testing model...", end=" ", flush=True)
        model.encode(["test sentence"])
        print("✅")

        print("\n✅ Model downloaded and ready\n")
        return True

    except Exception as e:
        print(f"\n❌ Model download failed: {e}")
        print("\n💡 Troubleshooting:")
        print("  • Check internet connection")
        print("  • Try running: python scripts/check_setup.py")
        return False


def run_tests():
    """Run verification tests"""
    print("Running verification tests...")
    print("(Checking core functionality)\n")

    try:
        result = subprocess.run(
            ["pytest", "tests/test_functional.py", "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Show output
        print(result.stdout)

        if result.returncode != 0:
            print("⚠️  Some tests failed, but setup may still work")
            print("   Run 'pytest tests/ -v' for details\n")
            return True  # Don't block on test failures

        print("✅ All tests passed\n")
        return True

    except subprocess.TimeoutExpired:
        print("⚠️  Tests timed out (may be slow system)")
        print("   Continuing anyway...\n")
        return True
    except FileNotFoundError:
        print("⚠️  pytest not found, skipping tests")
        print("   Install with: pip install pytest\n")
        return True
    except Exception as e:
        print(f"⚠️  Test error: {e}")
        print("   Continuing anyway...\n")
        return True


def run_demo():
    """Run interactive demo"""
    print("Running interactive demo...")
    print("(This demonstrates document compression in action)\n")

    try:
        # Run example with output shown
        result = subprocess.run(
            [sys.executable, "examples/example_usage.py"],
            timeout=30,
        )

        if result.returncode == 0:
            print("\n✅ Demo completed successfully")
        else:
            print("\n⚠️  Demo exited with errors (see above)")

        return True

    except subprocess.TimeoutExpired:
        print("\n⚠️  Demo timed out")
        return True
    except FileNotFoundError:
        print("\n❌ Demo script not found")
        print("   Make sure you're running from the project root")
        return False
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        return False


def print_next_steps():
    """Print helpful next steps"""
    print_header("🎉 Setup Complete!")

    print("You're ready to use Token Saver 5000!\n")

    print("📚 Quick Start Commands:\n")
    print("  # Start the MCP server (for Claude Desktop integration)")
    print("  token-saver-mcp\n")

    print("  # Run examples")
    print("  python examples/example_usage.py      # Document compression")
    print("  python examples/afm_demo.py          # Dialogue memory")
    print("  python examples/code_compression_example.py  # Code compression\n")

    print("  # Run tests")
    print("  pytest tests/ -v                     # All tests")
    print("  pytest tests/test_functional.py -v  # Core features")
    print("  pytest tests/test_edge_cases.py -v  # Error handling\n")

    print("  # Development tools")
    print("  black src/ tests/ examples/          # Format code")
    print("  ruff check src/ tests/               # Lint code")
    print("  pytest tests/ --cov=src --cov-report=html  # Coverage\n")

    print("📖 Documentation:")
    print("  • GETTING_STARTED.md  - Step-by-step guide")
    print("  • README.md           - Feature overview")
    print("  • HOW_IT_WORKS.md     - Technical deep dive")
    print("  • TROUBLESHOOTING.md  - Common issues\n")

    print("🆕 NEW in v0.4.0: File Sync & Version Management!")
    print("  ✨ Automatic staleness detection - Never use outdated cache")
    print("  📜 Full version history - Track all document changes")
    print("  🔄 Smart refresh workflow - Update cache when files change")
    print("  💾 Proactive health monitoring - See resource usage anytime\n")
    print("  Try these new MCP tools:")
    print("    • check_file_sync - See if cached docs are stale")
    print("    • diff_cached_file - View changes since last ingest")
    print("    • get_version_history - Browse version timeline")
    print("    • check_resource_health - Monitor storage & memory")
    print("    • refresh_document - Update cache from disk\n")

    print("🔗 MCP Integration:")
    print("  • Run: token-saver-install-mcp")
    print("  • Or create project config: token-saver-install-mcp --project-config")
    print(
        "  • Or create portable project config: token-saver-install-mcp --portable-project-config"
    )
    print("  • Or inspect setup health: token-saver-install-mcp --doctor")
    print("  • Or print raw JSON: token-saver-install-mcp --print-config > .mcp.json")
    print("  • Or manually configure Claude Desktop (see README.md)\n")


def main():
    """Run complete quickstart process"""
    print_header("🚀 Token Saver 5000 - Quickstart Setup")

    print("This script will set up Token Saver 5000 from scratch.")
    print("Total time: ~5-10 minutes (depending on internet speed)\n")

    # Confirm with user
    try:
        response = input("Continue? [Y/n]: ").strip().lower()
        if response and response != "y":
            print("\nSetup cancelled.")
            return 1
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        return 1

    total_steps = 5
    current_step = 0

    # Step 1: Check Python
    current_step += 1
    print_header(f"Step {current_step}/{total_steps}: Checking Python Version")
    if not check_python_version():
        return 1

    # Step 2: Install dependencies
    current_step += 1
    print_header(f"Step {current_step}/{total_steps}: Installing Dependencies")
    if not install_dependencies():
        print("\n❌ Setup failed at dependency installation")
        print("   Try manually: pip install -r requirements.txt")
        return 1

    # Step 3: Download model
    current_step += 1
    print_header(f"Step {current_step}/{total_steps}: Downloading Embedding Model")
    if not download_embedding_model():
        print("\n⚠️  Model download failed, but you can try setup manually")
        print("   Run: python scripts/check_setup.py")

    # Step 4: Run tests
    current_step += 1
    print_header(f"Step {current_step}/{total_steps}: Running Verification Tests")
    run_tests()  # Don't fail on test issues

    # Step 5: Run demo
    current_step += 1
    print_header(f"Step {current_step}/{total_steps}: Running Interactive Demo")
    run_demo()

    # Show next steps
    print_next_steps()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
