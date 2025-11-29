Store critical information in dialogue memory for future reference.

This command uses Adaptive Focus Memory (AFM) to ensure important information
is retained throughout our conversation and never forgotten.

Steps:
1. Analyze the information to determine priority:
   - CRITICAL: Safety info (allergies, medical), hard constraints
   - HIGH: Strong preferences, important context
   - NORMAL: General information

2. Use `mcp__token-saver__afm_add_message` with:
   - role: "user"
   - content: The information to remember

3. Confirm storage with priority level

4. Explain retention behavior:
   - CRITICAL items are NEVER compressed
   - HIGH items are preserved with high fidelity
   - All items can be retrieved with `mcp__token-saver__afm_build_context`

Information to remember: $ARGUMENTS
