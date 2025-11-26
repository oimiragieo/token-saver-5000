#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACE Framework Example Usage

This script demonstrates the Agentic Context Engineering (ACE) framework:
1. Initializing ACE playbook for a domain
2. Generate-Reflect-Curate cycle
3. Continuous learning workflow
4. Integration with semantic compression

ACE achieves 32% quality improvement with 4x shorter contexts through
self-evolving playbooks via Generate->Reflect->Curate cycles.

Reference: arXiv:2510.04618v1
"""

import sys
import os
import io

# Configure stdout for UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ace_framework import ACEFramework, BulletType, ACEContext
from src.semantic_compressor import SemanticCompressor, FidelityLevel


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_playbook_stats(context: ACEContext, title: str = "Playbook Statistics"):
    """Print formatted playbook statistics"""
    stats = context.get_performance_stats()
    print(f"\n{title}:")
    print(f"  Total bullets: {stats['total_bullets']}")
    print(f"  Average confidence: {stats['avg_confidence']:.2f}")
    print(f"  Average success rate: {stats['avg_success_rate']:.2f}")
    print(f"  Total usage: {stats['total_usage']}")
    print("  Bullets by type:")
    for bullet_type, count in stats["by_type"].items():
        if count > 0:
            print(f"    - {bullet_type}: {count}")


def demo_1_initialize_playbook():
    """
    Demo 1: Initializing ACE Playbook for a Domain

    Shows how to create a domain-specific playbook for code review,
    seeded with PRINCIPLE, STRATEGY, TACTIC, and CONSTRAINT bullets.
    """
    print_section("Demo 1: Initializing ACE Playbook for a Domain")

    print("Creating ACE Framework...")
    ace = ACEFramework(
        deduplication_threshold=0.85,  # Semantic similarity threshold
        max_bullets=100,  # Maximum bullets before pruning
    )

    print("\nCreating 'code_review' playbook with seed bullets...")

    # Define seed bullets for code review domain
    seed_bullets = [
        # Principles (high-level guidance)
        ("Focus on security vulnerabilities first", BulletType.PRINCIPLE),
        ("Prioritize readability over cleverness", BulletType.PRINCIPLE),
        # Strategies (tactical approaches)
        ("Check for proper error handling", BulletType.STRATEGY),
        ("Verify input validation at all boundaries", BulletType.STRATEGY),
        ("Look for potential race conditions", BulletType.STRATEGY),
        # Tactics (specific techniques)
        ("Review authentication logic for bypass opportunities", BulletType.TACTIC),
        ("Check SQL queries for injection vulnerabilities", BulletType.TACTIC),
        ("Verify proper cleanup in finally blocks", BulletType.TACTIC),
        # Constraints (hard requirements)
        ("No hardcoded credentials allowed", BulletType.CONSTRAINT),
        ("All database queries must use parameterized statements", BulletType.CONSTRAINT),
        ("Secrets must not be logged", BulletType.CONSTRAINT),
    ]

    # Create context with seed bullets
    context = ace.create_initial_context(initial_bullets=seed_bullets)

    print(f"\n[OK] Created playbook with {len(context.bullets)} seed bullets")
    print(f"   Context ID: {context.context_id[:16]}...")
    print(f"   Version: v{context.version}")

    # Display the bullets
    print("\nSeed Bullets by Type:")
    for bullet_type in BulletType:
        bullets = context.get_bullets_by_type(bullet_type)
        if bullets:
            print(f"\n  {bullet_type.value.upper()}:")
            for bullet in bullets:
                print(f"    - {bullet.text}")

    print_playbook_stats(context, "Initial Playbook Statistics")

    return ace, context


def demo_2_generate_reflect_curate(ace: ACEFramework, context: ACEContext):
    """
    Demo 2: Generate-Reflect-Curate Cycle

    Demonstrates a complete ACE cycle:
    1. Generate reasoning trajectory for a code review task
    2. Simulate task outcome (success with findings)
    3. Reflect on the outcome to extract insights
    4. Curate insights into playbook with semantic deduplication
    """
    print_section("Demo 2: Generate-Reflect-Curate Cycle")

    # Task: Review authentication function
    task = """
    Review this authentication function for security issues:

    def authenticate(username, password):
        query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
        result = db.execute(query)
        return result is not None
    """

    print("Task: Review authentication function for security issues")
    print("\n🔄 Step 1: Generate reasoning trajectory...")

    # Generate trajectory using current playbook
    trajectory = ace.generator.generate_trajectory(
        task=task,
        context=context,
        max_steps=3,  # 3-step reasoning process
        top_k_bullets=5,  # Consider top 5 relevant bullets per step
    )

    print(f"\n✅ Generated trajectory with {len(trajectory)} steps:")
    for step in trajectory:
        print(f"\n   Step {step['step_number']} (confidence: {step['confidence']:.2f}):")
        print(f"   Applied {len(step['relevant_bullets'])} bullets:")
        for bullet in step["relevant_bullets"][:2]:  # Show first 2 bullets
            print(f"     • [{bullet['bullet_type']}] {bullet['text'][:60]}...")

    # Simulate task outcome
    outcome = """
    Found critical SQL injection vulnerability in authentication function.
    The query uses string formatting instead of parameterized statements.
    Attack vector: username="admin'--" bypasses password check.
    Also found: passwords stored in plaintext (no hashing).
    Missing: rate limiting for failed login attempts.
    """

    success = True  # Task succeeded in finding issues

    print("\n🔄 Step 2: Reflect on outcome to extract insights...")
    print(f"   Outcome: {outcome.strip()[:100]}...")
    print(f"   Success: {success}")

    # Reflect on the trajectory
    insights = ace.reflector.reflect_on_trajectory(
        trajectory=trajectory, outcome=outcome, success=success
    )

    print(f"\n✅ Extracted {len(insights)} insights from reflection:")
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. [{insight['bullet_type']}] {insight['text'][:80]}...")
        print(f"      Confidence: {insight['confidence']:.2f}")

    print("\n🔄 Step 3: Curate insights into playbook...")
    print(f"   Before: {len(context.bullets)} bullets, version v{context.version}")

    # Curate insights (with semantic deduplication)
    updated_context = ace.curator.curate_insights(
        context=context, insights=insights, max_bullets=ace.max_bullets
    )

    print(f"   After: {len(updated_context.bullets)} bullets, version v{updated_context.version}")

    # Show delta history (recent changes)
    if updated_context.delta_history:
        print("\n   Recent changes:")
        for delta in updated_context.delta_history[-3:]:  # Last 3 changes
            print(
                f"     v{delta['version']}: {delta['operation']} - {delta['description'][:60]}..."
            )

    print_playbook_stats(updated_context, "Updated Playbook Statistics")

    print("\n✅ Complete cycle executed: Generate → Reflect → Curate")

    return updated_context


def demo_3_continuous_learning(ace: ACEFramework, context: ACEContext):
    """
    Demo 3: Continuous Learning Workflow

    Demonstrates ACE playbook evolution over multiple tasks:
    - Execute multiple tasks in sequence
    - Show playbook improving over time
    - Track confidence scores and performance
    """
    print_section("Demo 3: Continuous Learning Workflow")

    print("Simulating 5 sequential code review tasks...")
    print("Playbook should evolve and improve with each task.")

    # Define 5 tasks with outcomes
    tasks = [
        {
            "task": "Review API endpoint for authorization issues",
            "outcome": "Found missing authorization check on DELETE endpoint. Any authenticated user can delete any resource.",
            "success": True,
        },
        {
            "task": "Review file upload handler for security issues",
            "outcome": "Found unrestricted file upload. No validation of file type or size. Potential for malicious file execution.",
            "success": True,
        },
        {
            "task": "Review password reset function",
            "outcome": "Missing rate limiting. Token generation uses weak random. Token doesn't expire.",
            "success": True,
        },
        {
            "task": "Review logging implementation",
            "outcome": "Found sensitive data (passwords, tokens) being logged. GDPR/compliance violation.",
            "success": True,
        },
        {
            "task": "Review error handling in payment processing",
            "outcome": "Error messages leak internal details. No retry logic. Missing transaction rollback on failure.",
            "success": False,  # One failure to show learning from errors
        },
    ]

    # Track metrics across tasks
    version_history = [context.version]
    bullet_count_history = [len(context.bullets)]
    confidence_history = [context.get_performance_stats()["avg_confidence"]]

    for i, task_data in enumerate(tasks, 1):
        print(f"\n--- Task {i}/5: {task_data['task'][:50]}... ---")

        # Execute ACE cycle
        updated_context, trajectory = ace.execute_ace_cycle(
            task=task_data["task"],
            context=context,
            outcome=task_data["outcome"],
            success=task_data["success"],
            max_trajectory_steps=2,  # Shorter for demo
        )

        context = updated_context

        # Track metrics
        version_history.append(context.version)
        bullet_count_history.append(len(context.bullets))
        stats = context.get_performance_stats()
        confidence_history.append(stats["avg_confidence"])

        print(f"   Outcome: {'✅ Success' if task_data['success'] else '❌ Failure'}")
        print(f"   Trajectory steps: {len(trajectory)}")
        print(f"   Version: v{context.version}")
        print(f"   Bullets: {len(context.bullets)}")
        print(f"   Avg confidence: {stats['avg_confidence']:.3f}")

    # Show evolution over time
    print("\n📊 Playbook Evolution Over Time:")
    print("\nTask | Version | Bullets | Avg Confidence")
    print("-----|---------|---------|---------------")
    for i in range(len(version_history)):
        print(
            f"  {i}  |   v{version_history[i]:<2}   |   {bullet_count_history[i]:<2}    |     {confidence_history[i]:.3f}"
        )

    # Show top bullets by confidence
    top_bullets = context.get_top_bullets(k=5, min_confidence=0.5)
    print(f"\n🏆 Top {len(top_bullets)} Bullets by Confidence:")
    for i, bullet in enumerate(top_bullets, 1):
        print(f"\n   {i}. [{bullet.bullet_type.value}] (confidence: {bullet.confidence:.2f})")
        print(f"      {bullet.text}")
        print(f"      Usage: {bullet.total_usage} times | Success rate: {bullet.success_rate:.1%}")

    print_playbook_stats(context, "Final Playbook Statistics")

    print(
        "\n✅ Continuous learning demonstrated: playbook evolved from v1 to v{} over 5 tasks".format(
            context.version
        )
    )

    return context


def demo_4_integration_with_compression(ace: ACEFramework, context: ACEContext):
    """
    Demo 4: Integration with Semantic Compression

    Shows how ACE playbook guides semantic node selection:
    1. Ingest a code file with semantic compression
    2. Use ACE playbook to guide which nodes to retrieve
    3. Demonstrate meta-level context optimization
    """
    print_section("Demo 4: Integration with Semantic Compression")

    print("Demonstrating ACE + Semantic Compression integration...")

    # Sample code document
    code_document = """
    # User Authentication Module

    ## Authentication Handler

    def authenticate_user(username, password):
        '''Authenticates user credentials against database'''
        # SECURITY: Uses parameterized queries to prevent SQL injection
        query = "SELECT * FROM users WHERE username = ? AND password_hash = ?"
        user = db.execute(query, (username, hash_password(password)))

        if user:
            # Check if account is locked
            if user.failed_attempts >= 5:
                raise AccountLockedException("Account locked after 5 failed attempts")

            # Reset failed attempts on success
            db.execute("UPDATE users SET failed_attempts = 0 WHERE id = ?", (user.id,))
            return create_session(user)
        else:
            # Increment failed attempts
            db.execute("UPDATE users SET failed_attempts = failed_attempts + 1 WHERE username = ?", (username,))
            raise AuthenticationFailedException("Invalid credentials")

    ## Password Hashing

    def hash_password(password):
        '''Hashes password using bcrypt with salt'''
        import bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt)

    ## Session Management

    def create_session(user):
        '''Creates authenticated session with secure token'''
        import secrets
        token = secrets.token_urlsafe(32)
        expiry = datetime.now() + timedelta(hours=24)

        db.execute(
            "INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)",
            (user.id, token, expiry)
        )

        return {"token": token, "expires_at": expiry}

    ## Rate Limiting

    def check_rate_limit(ip_address):
        '''Enforces rate limiting per IP address'''
        # Allow max 10 attempts per minute
        attempts = redis.get(f"auth_attempts:{ip_address}")
        if attempts and int(attempts) >= 10:
            raise RateLimitExceededException("Too many authentication attempts")

        redis.incr(f"auth_attempts:{ip_address}")
        redis.expire(f"auth_attempts:{ip_address}", 60)

    ## Logging

    def log_auth_event(username, success, ip_address):
        '''Logs authentication events for audit'''
        # SECURITY: Do not log passwords or tokens
        logger.info(
            "Authentication attempt",
            extra={
                "username": username,
                "success": success,
                "ip": ip_address,
                "timestamp": datetime.now()
            }
        )
    """

    print("\n🔄 Step 1: Ingest code with semantic compression...")

    try:
        compressor = SemanticCompressor(
            model_name="all-MiniLM-L6-v2", similarity_threshold=0.75, skeleton_ratio=0.2
        )

        skeleton = compressor.ingest_file(
            text=code_document,
            file_id="auth_module",
            metadata={"type": "code", "language": "python"},
        )

        print("\n✅ Document compressed:")
        print(f"   Original tokens: {skeleton.total_tokens:,}")
        print(f"   Skeleton tokens: {skeleton.skeleton_tokens:,}")
        print(f"   Compression ratio: {skeleton.compression_ratio:.1f}x")
        print(f"   Token savings: {(1 - skeleton.skeleton_tokens/skeleton.total_tokens)*100:.1f}%")

    except Exception as e:
        print(f"\n⚠️  Compression demo skipped: {e}")
        print("   (This demo requires sentence-transformers model)")
        return

    print("\n🔄 Step 2: Use ACE playbook to guide node selection...")

    # Use ACE playbook bullets to create a search query
    # Get top security-related bullets from playbook
    security_bullets = [
        b
        for b in context.bullets.values()
        if any(
            keyword in b.text.lower() for keyword in ["security", "vulnerability", "sql", "auth"]
        )
    ][:3]

    print("\n   Top security-related bullets from playbook:")
    for bullet in security_bullets:
        print(f"     • {bullet.text}")

    # Search for security-critical nodes
    query = "authentication security sql injection rate limiting"
    print(f"\n   Semantic search query: '{query}'")

    results = compressor.search_semantic(query, file_id="auth_module", top_k=3)

    print(f"\n✅ Found {len(results)} relevant nodes:")
    for i, node_id in enumerate(results, 1):
        node = compressor.chunks[node_id]
        summary = compressor._generate_summary(node.text, max_length=80)
        print(f"\n   {i}. [{node_id}] (importance: {node.importance:.3f})")
        print(f"      {summary}")

    print("\n🔄 Step 3: Retrieve at different fidelity levels...")

    # Retrieve most important node at STRUCTURE level
    if results:
        top_node = results[0]
        print(f"\n   Retrieving node {top_node} at STRUCTURE level:")

        content = compressor.modulate_region([top_node], FidelityLevel.STRUCTURE)
        print("\n" + "─" * 60)
        print(content[:300] + "..." if len(content) > 300 else content)
        print("─" * 60)

    print("\n🎯 Meta-Level Context Optimization:")
    print("\n   How ACE + Compression work together:")
    print("   1. ACE playbook defines security review priorities")
    print("   2. Compression creates semantic graph of code")
    print("   3. ACE bullets guide which nodes to retrieve")
    print("   4. Only relevant code sections retrieved at needed fidelity")
    print("   5. Both systems evolve: playbook learns + graph adapts")

    print("\n✅ Integration demonstrated: ACE playbook guides semantic node selection")


def main():
    print_section("ACE Framework - Example Usage")

    print("This demonstration shows Agentic Context Engineering (ACE):")
    print("  - Self-evolving playbooks via Generate->Reflect->Curate cycles")
    print("  - 32% quality improvement with 4x shorter contexts")
    print("  - Delta updates with semantic deduplication")
    print("  - Integration with semantic compression")
    print("\nReference: arXiv:2510.04618v1")

    try:
        # Demo 1: Initialize playbook
        ace, context = demo_1_initialize_playbook()

        # Demo 2: Generate-Reflect-Curate cycle
        context = demo_2_generate_reflect_curate(ace, context)

        # Demo 3: Continuous learning
        context = demo_3_continuous_learning(ace, context)

        # Demo 4: Integration with compression
        demo_4_integration_with_compression(ace, context)

        # Final summary
        print_section("✅ All ACE Demos Complete!")

        print("Key Takeaways:")
        print("  • ACE playbooks evolve through experience (Generate→Reflect→Curate)")
        print("  • Semantic deduplication prevents context collapse")
        print("  • Delta updates enable incremental, localized changes")
        print("  • Integration with compression enables meta-level optimization")
        print("  • Confidence scores track bullet performance over time")

        print("\n📊 Final Playbook State:")
        print(f"  Context ID: {context.context_id[:16]}...")
        print(f"  Version: v{context.version}")
        print(f"  Total bullets: {len(context.bullets)}")

        stats = context.get_performance_stats()
        print(f"  Average confidence: {stats['avg_confidence']:.2f}")
        print(f"  Total usage: {stats['total_usage']}")

        print("\n🎓 Research Foundation:")
        print("  Agentic Context Engineering (arXiv:2510.04618v1)")
        print("  Section 3: Delta Updates and Semantic Deduplication")
        print("  Section 4: Empirical Results (32% quality boost)")

        print("\n💡 Next Steps:")
        print("  • Try creating playbooks for your own domains")
        print("  • Experiment with different deduplication thresholds (0.75-0.95)")
        print("  • Combine ACE with blind spot detection for self-correction")
        print("  • Use ACE to guide semantic compression strategies")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        print(
            "\n💡 Please report this issue: https://github.com/oimiragieo/token-saver-5000/issues"
        )
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        sys.exit(1)
