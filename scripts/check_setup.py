#!/usr/bin/env python3
"""
Setup Verification Script for Token Saver 5000

Checks all requirements and verifies the installation is working correctly.
Run this script after installation to ensure everything is set up properly.

Usage:
    python check_setup.py
    # or
    python3 check_setup.py
"""

import sys


def check_python_version():
    """Verify Python version >= 3.10"""
    print("=" * 70)
    print("1. Checking Python Version...")
    print("=" * 70)

    major = sys.version_info.major
    minor = sys.version_info.minor
    version = f"{major}.{minor}"

    print(f"Current Python version: {sys.version}")

    if major >= 3 and minor >= 10:
        print(f"✅ Python {version} is supported (requirement: >= 3.10)")
        return True
    else:
        print(f"❌ Python {version} is NOT supported (requirement: >= 3.10)")
        print("   Please upgrade to Python 3.10 or higher")
        return False


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n" + "=" * 70)
    print("2. Checking Dependencies...")
    print("=" * 70)

    required_packages = [
        ("mcp", "Model Context Protocol"),
        ("sentence_transformers", "Sentence Transformers for embeddings"),
        ("networkx", "NetworkX for graph analysis"),
        ("sklearn", "scikit-learn for ML utilities"),
        ("numpy", "NumPy for numerical computing"),
        ("torch", "PyTorch for neural networks"),
        ("chromadb", "ChromaDB for vector database"),
        ("pydantic", "Pydantic for data validation"),
        ("tiktoken", "TikToken for token counting"),
        ("tqdm", "TQDM for progress bars"),
    ]

    all_installed = True
    installed_count = 0

    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package:25s} - {description}")
            installed_count += 1
        except ImportError:
            print(f"❌ {package:25s} - {description} (NOT INSTALLED)")
            all_installed = False

    print(f"\n📊 Installed: {installed_count}/{len(required_packages)}")

    if not all_installed:
        print("\n⚠️  Missing dependencies detected!")
        print("   Install with: pip install -r requirements.txt")
        return False

    return True


def check_imports():
    """Test importing all core modules"""
    print("\n" + "=" * 70)
    print("3. Checking Module Imports...")
    print("=" * 70)

    modules = [
        "src.semantic_compressor",
        "src.code_compressor",
        "src.multimodal_compressor",
        "src.scar_compressor",
        "src.adaptive_rate_allocator",
        "src.blind_spot_detector",
        "src.semantic_ssim",
        "src.training_utils",
        "src.server",
    ]

    all_imported = True
    imported_count = 0

    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
            imported_count += 1
        except Exception as e:
            print(f"❌ {module} - Error: {str(e)[:60]}")
            all_imported = False

    print(f"\n📊 Imported: {imported_count}/{len(modules)}")

    if not all_imported:
        print("\n⚠️  Module import errors detected!")
        return False

    return True


def check_embedding_model():
    """Test if embedding model can be loaded"""
    print("\n" + "=" * 70)
    print("4. Checking Embedding Model...")
    print("=" * 70)

    try:
        from sentence_transformers import SentenceTransformer
        import time

        print("Loading all-MiniLM-L6-v2 model...")
        print("(First run: downloading ~80MB model from HuggingFace)")
        print("💡 Tip: Progress bar will show download status automatically")
        print("")

        # Retry logic for model download with exponential backoff
        max_retries = 3
        model = None

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"  Retry attempt {attempt + 1}/{max_retries}...")
                model = SentenceTransformer("all-MiniLM-L6-v2")
                print("✅ Model loaded successfully")
                break
            except Exception as e:
                print(f"❌ Download failed: {str(e)[:80]}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                    print(f"  ⏳ Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("  ⚠️  All retry attempts exhausted")
                    raise

        if model is None:
            raise RuntimeError("Failed to load model after all retries")

        print("✅ Model loaded successfully")

        # Test encoding
        test_text = "This is a test sentence"
        embedding = model.encode([test_text])
        print(f"✅ Model can encode text (embedding dimension: {embedding.shape[1]})")

        return True
    except Exception as e:
        print(f"❌ Failed to load embedding model: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("  1. Check internet connection")
        print(
            "  2. Try manual download: python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer(\"all-MiniLM-L6-v2\")'"
        )
        print("  3. See TROUBLESHOOTING.md for more help")
        return False


def quick_smoke_test():
    """Run a quick functional test"""
    print("\n" + "=" * 70)
    print("5. Running Smoke Test...")
    print("=" * 70)

    try:
        from src.semantic_compressor import SemanticCompressor, FidelityLevel

        print("Initializing SemanticCompressor...")
        compressor = SemanticCompressor()

        print("Ingesting test document...")
        test_doc = """
        Quantum computing is a revolutionary technology.

        Quantum computers use qubits instead of classical bits.
        This enables parallel computation through superposition.

        Error correction remains a major challenge.
        """

        result = compressor.ingest_file(test_doc, "test_doc")

        print(f"✅ Document ingested: {result.total_nodes} nodes")
        print(f"✅ Compression: {result.total_tokens} → {result.skeleton_tokens} tokens")
        print(f"✅ Ratio: {result.compression_ratio:.1f}x")

        # Test search
        print("\nTesting semantic search...")
        search_results = compressor.search_semantic("quantum computers", "test_doc", top_k=2)
        print(f"✅ Search returned {len(search_results)} results")

        # Test modulation
        print("\nTesting fidelity modulation...")
        if search_results:
            content = compressor.modulate_region([search_results[0]], FidelityLevel.ABSTRACT)
            print(f"✅ Retrieved content at ABSTRACT fidelity ({len(content)} chars)")

        return True
    except Exception as e:
        print(f"❌ Smoke test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all checks"""
    print("\n" + "=" * 70)
    print("  TOKEN SAVER 5000 - SETUP VERIFICATION")
    print("=" * 70)

    results = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Module Imports": check_imports(),
        "Embedding Model": check_embedding_model(),
        "Smoke Test": quick_smoke_test(),
    }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check}")

    print("\n" + "=" * 70)
    print(f"Result: {passed}/{total} checks passed")
    print("=" * 70)

    if passed == total:
        print("\n🎉 All checks passed! Token Saver 5000 is ready to use.")
        print("\nNext steps:")
        print("  1. Run examples: python examples/example_usage.py")
        print("  2. Run tests: pytest tests/ -v")
        print("  3. Start MCP server: python -m src.server")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Upgrade Python: https://www.python.org/downloads/")
        print("  - Check system requirements: See GETTING_STARTED.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
