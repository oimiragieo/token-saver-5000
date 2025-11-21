#!/usr/bin/env python3
"""
AFM (Adaptive Focus Memory) Demo

Demonstrates the dialogue memory management system described in:
"Adaptive Focus Memory for Language Models" (arXiv:2511.12712v1)

This example replicates the safety-critical benchmark from the paper:
- User declares a severe peanut allergy early in conversation
- Multiple intervening topics discussed
- Later asks about Thai street food
- AFM should retain the allergy and provide safe recommendations

Expected result: ~66% token reduction while preserving safety-critical information
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.afm import FocusManager, AFMConfig


def print_separator(title: str = ""):
    """Print a visual separator"""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    else:
        print(f"{'='*70}\n")


def print_message(role: str, content: str, prefix: str = ""):
    """Print a formatted message"""
    role_emoji = {"user": "👤", "assistant": "🤖", "system": "⚙️ "}
    print(f"{prefix}{role_emoji.get(role, '  ')} {role.upper()}: {content}\n")


def print_stats(stats, title: str = "Context Packing Statistics"):
    """Print packing statistics"""
    print(f"\n📊 {title}")
    print(f"{'─'*70}")
    print(f"  Total messages:     {stats.total_messages}")
    print(f"  ├─ FULL:            {stats.full_count}")
    print(f"  ├─ COMPRESSED:      {stats.compressed_count}")
    print(f"  ├─ PLACEHOLDER:     {stats.placeholder_count}")
    print(f"  └─ DROPPED:         {stats.dropped_count}")
    print(f"  Tokens used:        {stats.total_tokens} / {stats.budget_tokens}")
    print(f"  Budget utilization: {stats.compression_ratio:.1%}")
    print(f"{'─'*70}\n")


def run_short_conversation_demo():
    """
    Short conversation (3 turns)
    Allergy and food question close together
    """
    print_separator("SHORT CONVERSATION DEMO (3 turns)")

    # Initialize AFM
    config = AFMConfig(
        tau_high=0.45,
        tau_mid=0.25,
        half_life=12,
        use_llm_importance=False,  # Use heuristic importance
        use_llm_compression=False,  # Use heuristic compression
    )
    manager = FocusManager(config)

    # Conversation
    print("🎬 Starting conversation...\n")

    # Turn 1: User states allergy
    manager.add_message("user", "I'm planning a trip to Thailand! I'm really excited.")
    print_message("user", "I'm planning a trip to Thailand! I'm really excited.")

    manager.add_message(
        "assistant",
        "That's wonderful! Thailand is an amazing destination. What would you like to know about planning your trip?",
    )
    print_message(
        "assistant",
        "That's wonderful! Thailand is an amazing destination. What would you like to know about planning your trip?",
    )

    # Turn 2: Critical allergy declaration
    manager.add_message(
        "user",
        "Before we get into details, I need to mention that I have a severe peanut allergy. It's life-threatening.",
    )
    print_message(
        "user",
        "Before we get into details, I need to mention that I have a severe peanut allergy. It's life-threatening.",
        prefix="⚠️  ",
    )

    manager.add_message(
        "assistant",
        "Thank you for letting me know about your severe peanut allergy. I'll make sure all my recommendations are safe for you. Thai cuisine does use peanuts frequently, so we'll need to be very careful.",
    )
    print_message(
        "assistant",
        "Thank you for letting me know about your severe peanut allergy. I'll make sure all my recommendations are safe for you. Thai cuisine does use peanuts frequently, so we'll need to be very careful.",
    )

    # Turn 3: Food question (should trigger allergy memory)
    current_query = "The street food sounds AWESOME! I wanna have it all. What are the best street foods I should try?"

    print_message("user", current_query, prefix="🍜 ")

    # Build context with AFM
    context, stats = manager.build_context(
        current_query=current_query,
        budget_tokens=800,
        system_preamble="You are a helpful travel assistant.",
    )

    print_stats(stats)

    # Show what was included in context
    print("📝 Context Sent to LLM:")
    print(f"{'─'*70}")
    for i, (role, content) in enumerate(context):
        print(f"  [{i+1}] {role}: {content[:80]}{'...' if len(content) > 80 else ''}")
    print(f"{'─'*70}\n")

    # Check if allergy was retained
    allergy_retained = any(
        "allergy" in content.lower() or "peanut" in content.lower() for _, content in context
    )
    if allergy_retained:
        print("✅ SUCCESS: Allergy information RETAINED in context!")
    else:
        print("❌ FAILURE: Allergy information LOST!")

    print("\n")


def run_medium_conversation_demo():
    """
    Medium conversation (9 turns)
    Allergy stated early, then several intervening topics before food question
    This is the challenging scenario from the AFM paper
    """
    print_separator("MEDIUM CONVERSATION DEMO (9 turns)")
    print("This replicates the safety-critical benchmark from the AFM paper.\n")

    # Initialize AFM
    config = AFMConfig(
        tau_high=0.45,
        tau_mid=0.25,
        half_life=12,
        use_llm_importance=False,
        use_llm_compression=False,
    )
    manager = FocusManager(config)

    print("🎬 Starting conversation...\n")

    # Turn 1
    manager.add_message("user", "I'm planning a trip to Thailand! I'm really excited.")
    print_message("user", "I'm planning a trip to Thailand! I'm really excited.")

    # Turn 2: Critical allergy (early)
    manager.add_message(
        "user",
        "Before we start planning, you should know I have a severe peanut allergy. It's life-threatening and I need to be extremely careful.",
    )
    print_message(
        "user",
        "Before we start planning, you should know I have a severe peanut allergy. It's life-threatening and I need to be extremely careful.",
        prefix="⚠️  ",
    )

    manager.add_message(
        "assistant",
        "Thank you for sharing that critical information about your severe peanut allergy. I will keep this in mind for all recommendations, especially regarding Thai cuisine which frequently uses peanuts.",
    )
    print_message(
        "assistant",
        "Thank you for sharing that critical information about your severe peanut allergy. I will keep this in mind for all recommendations, especially regarding Thai cuisine which frequently uses peanuts.",
    )

    # Turn 3-8: Intervening topics (destinations, transport, activities, culture, Muay Thai, temples)
    intervening_topics = [
        ("user", "What are the best destinations in Thailand?"),
        (
            "assistant",
            "Popular destinations include Bangkok (vibrant city), Chiang Mai (cultural hub), Phuket (beaches), and Krabi (islands). Each offers unique experiences.",
        ),
        ("user", "How do I get around between cities?"),
        (
            "assistant",
            "You can use domestic flights (fastest), trains (scenic), or buses (budget-friendly). Many travelers enjoy the overnight train from Bangkok to Chiang Mai.",
        ),
        ("user", "I'd love to try Muay Thai!"),
        (
            "assistant",
            "Muay Thai is Thailand's national sport! You can watch matches at Lumpinee or Rajadamnern stadiums in Bangkok, or take classes at many gyms throughout the country.",
        ),
        ("user", "What about temples? I want to see the cultural side."),
        (
            "assistant",
            "Thailand has stunning temples! Must-sees include Wat Phra Kaew (Grand Palace), Wat Pho (Reclining Buddha), and Wat Arun in Bangkok. Remember to dress modestly.",
        ),
    ]

    for role, content in intervening_topics:
        manager.add_message(role, content)
        print_message(role, content)

    # Turn 9: Food question (should trigger allergy memory despite distance)
    current_query = "The street food sounds AWESOME! I wanna have it all. What are the best street foods I should try?"

    print_message("user", current_query, prefix="🍜 ")

    # Build context with AFM
    context, stats = manager.build_context(
        current_query=current_query,
        budget_tokens=800,
        system_preamble="You are a helpful travel assistant specializing in Thailand.",
    )

    print_stats(stats)

    # Show what was included
    print("📝 Context Sent to LLM:")
    print(f"{'─'*70}")
    for i, (role, content) in enumerate(context):
        is_allergy_message = "allergy" in content.lower() or "peanut" in content.lower()
        prefix = "⚠️  " if is_allergy_message else "   "
        print(f"{prefix}[{i+1}] {role}: {content[:100]}{'...' if len(content) > 100 else ''}")
    print(f"{'─'*70}\n")

    # Check if allergy was retained
    allergy_retained = any(
        "allergy" in content.lower() or "peanut" in content.lower() for _, content in context
    )
    if allergy_retained:
        print("✅ SUCCESS: Allergy information RETAINED across 9 turns!")
        print("   AFM correctly identified and preserved safety-critical information.")
    else:
        print("❌ FAILURE: Allergy information LOST!")
        print("   This would be a serious safety issue.")

    print("\n")


def run_token_savings_comparison():
    """
    Compare AFM vs naive replay on token usage
    """
    print_separator("TOKEN SAVINGS COMPARISON")

    config = AFMConfig()
    manager = FocusManager(config)

    # Simulate a conversation
    messages = [
        ("user", "I have a severe peanut allergy."),
        ("assistant", "Noted. I'll keep that in mind."),
        ("user", "What's the weather like in Thailand?"),
        (
            "assistant",
            "Thailand has tropical weather, hot and humid year-round with a rainy season from June to October.",
        ),
        ("user", "Tell me about Bangkok."),
        (
            "assistant",
            "Bangkok is Thailand's vibrant capital with temples, markets, and amazing food.",
        ),
        ("user", "What about Chiang Mai?"),
        (
            "assistant",
            "Chiang Mai is a cultural hub in northern Thailand with temples, mountains, and a relaxed vibe.",
        ),
        ("user", "Is public transport good?"),
        (
            "assistant",
            "Yes! Bangkok has BTS Skytrain, MRT subway, and boats. Grab (like Uber) is also popular.",
        ),
    ]

    for role, content in messages:
        manager.add_message(role, content)

    current_query = "What Thai street food should I try?"

    # Naive replay: All messages verbatim
    from src.afm import TokenCounter

    counter = TokenCounter()

    naive_context = [(role, content) for role, content in messages]
    naive_context.append(("user", current_query))
    naive_tokens = sum(counter.count(content) for _, content in naive_context)

    # AFM with budget
    afm_context, stats = manager.build_context(current_query, budget_tokens=800)
    afm_tokens = stats.total_tokens

    # Calculate savings
    savings_pct = (1 - afm_tokens / naive_tokens) * 100 if naive_tokens > 0 else 0

    print("📊 Token Usage Comparison\n")
    print(f"  Naive Replay:       {naive_tokens} tokens")
    print(f"  AFM (budget=800):   {afm_tokens} tokens")
    print(f"  Savings:            {naive_tokens - afm_tokens} tokens ({savings_pct:.1f}%)")
    print(f"\n  Paper claims ~66% reduction, we achieved {savings_pct:.1f}%\n")

    if savings_pct >= 50:
        print("✅ Excellent token savings!")
    elif savings_pct >= 30:
        print("✓  Good token savings")
    else:
        print("⚠️  Lower savings than expected")

    print("\n")


def main():
    """Run all demos"""
    print_separator("AFM (ADAPTIVE FOCUS MEMORY) DEMONSTRATION")
    print("Implementation of arXiv:2511.12712v1")
    print("'Adaptive Focus Memory for Language Models'")
    print("by Christopher Cruz, Purdue University\n")

    # Run demos
    run_short_conversation_demo()
    run_medium_conversation_demo()
    run_token_savings_comparison()

    print_separator("DEMO COMPLETE")
    print("Key Takeaways:")
    print("  • AFM preserves safety-critical information (allergies) across long conversations")
    print("  • Achieves significant token savings (~50-70%) vs naive replay")
    print("  • Uses semantic similarity + recency weighting + importance classification")
    print("  • Operates entirely locally (no external API calls in heuristic mode)")
    print("\nFor more information, see the AFM paper: arXiv:2511.12712v1")
    print("License: CC BY 4.0\n")


if __name__ == "__main__":
    main()
