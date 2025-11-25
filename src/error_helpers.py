"""
Smart Error Message System with Fuzzy Matching

This module provides enhanced error messages with "Did you mean?" suggestions
for common user mistakes like typos in file IDs, node IDs, and enum values.

**Why This Matters:**
- Users frequently make typos when specifying file_ids, node_ids, fidelity levels
- Standard errors are unhelpful: "file_id 'quantum_papper' not found"
- Smart errors suggest corrections: "Did you mean 'quantum_paper'?"

**Features:**
1. Fuzzy matching using difflib (Python stdlib - no dependencies)
2. Configurable similarity threshold (default: 0.6 = 60% match)
3. Multiple suggestion support (top 3 closest matches)
4. Context-aware error messages with helpful tips

Version: 0.4.1
Author: Token Saver 5000 Team
"""

import difflib
from typing import List, Optional, Any


class SmartError:
    """
    Generates helpful error messages with fuzzy-matched suggestions.

    Uses Python's difflib.get_close_matches() to find similar values
    when users make typos or misremember exact names.

    Example:
        >>> valid_ids = ["quantum_paper", "neural_nets", "blockchain_intro"]
        >>> SmartError.file_id_not_found("quantum_papper", valid_ids)
        ValueError: file_id 'quantum_papper' not found
        💡 Did you mean 'quantum_paper'?
           Available: quantum_paper, neural_nets, blockchain_intro
    """

    @staticmethod
    def file_id_not_found(
        invalid_id: str,
        available_ids: List[str],
        n_suggestions: int = 3,
        cutoff: float = 0.6,
    ) -> ValueError:
        """
        Generate error for invalid file_id with fuzzy match suggestions.

        Args:
            invalid_id: The file_id that wasn't found
            available_ids: List of valid file_ids to match against
            n_suggestions: Maximum number of suggestions (default: 3)
            cutoff: Similarity threshold 0.0-1.0 (default: 0.6 = 60% match)

        Returns:
            ValueError with helpful message and suggestions

        Example:
            >>> raise SmartError.file_id_not_found("quntum_paper", ["quantum_paper"])
            ValueError: file_id 'quntum_paper' not found
            💡 Did you mean 'quantum_paper'?
        """
        # Get fuzzy matches
        matches = difflib.get_close_matches(
            invalid_id, available_ids, n=n_suggestions, cutoff=cutoff
        )

        # Build error message
        msg = f"file_id '{invalid_id}' not found"

        if matches:
            if len(matches) == 1:
                msg += f"\n💡 Did you mean '{matches[0]}'?"
            else:
                suggestions = "', '".join(matches)
                msg += f"\n💡 Did you mean one of: '{suggestions}'?"

        # Always show available options (limit to 5 for brevity)
        if len(available_ids) <= 5:
            available_str = ", ".join(available_ids)
            msg += f"\n   Available: {available_str}"
        else:
            available_str = ", ".join(available_ids[:5])
            msg += f"\n   Available ({len(available_ids)} total): {available_str}, ..."

        return ValueError(msg)

    @staticmethod
    def node_id_not_found(
        invalid_id: str,
        available_ids: List[str],
        file_id: str,
        n_suggestions: int = 3,
        cutoff: float = 0.6,
    ) -> ValueError:
        """
        Generate error for invalid node_id with fuzzy match suggestions.

        Args:
            invalid_id: The node_id that wasn't found
            available_ids: List of valid node_ids for this file
            file_id: The file_id context
            n_suggestions: Maximum number of suggestions (default: 3)
            cutoff: Similarity threshold 0.0-1.0 (default: 0.6)

        Returns:
            ValueError with helpful message and suggestions

        Example:
            >>> raise SmartError.node_id_not_found(
            ...     "quantum_paper_n99",
            ...     ["quantum_paper_n0", "quantum_paper_n1"],
            ...     "quantum_paper"
            ... )
            ValueError: node_id 'quantum_paper_n99' not found in file 'quantum_paper'
            💡 Did you mean 'quantum_paper_n1'?
        """
        matches = difflib.get_close_matches(
            invalid_id, available_ids, n=n_suggestions, cutoff=cutoff
        )

        msg = f"node_id '{invalid_id}' not found in file '{file_id}'"

        if matches:
            if len(matches) == 1:
                msg += f"\n💡 Did you mean '{matches[0]}'?"
            else:
                suggestions = "', '".join(matches)
                msg += f"\n💡 Did you mean one of: '{suggestions}'?"

        # Show node count and format example
        msg += f"\n   File '{file_id}' has {len(available_ids)} nodes"
        if available_ids:
            msg += f"\n   Format: '{available_ids[0].rsplit('_n', 1)[0]}_nX' where X = 0 to {len(available_ids)-1}"

        return ValueError(msg)

    @staticmethod
    def invalid_enum_value(
        invalid_value: str,
        valid_values: List[str],
        enum_name: str = "value",
        n_suggestions: int = 3,
        cutoff: float = 0.6,
    ) -> ValueError:
        """
        Generate error for invalid enum value with fuzzy match suggestions.

        Args:
            invalid_value: The value that wasn't recognized
            valid_values: List of valid enum values
            enum_name: Name of the parameter (e.g., "fidelity_level", "use_case")
            n_suggestions: Maximum number of suggestions (default: 3)
            cutoff: Similarity threshold 0.0-1.0 (default: 0.6)

        Returns:
            ValueError with helpful message and suggestions

        Example:
            >>> raise SmartError.invalid_enum_value(
            ...     "DETALED",
            ...     ["ABSTRACT", "OUTLINE", "STRUCTURE", "DETAILED", "RAW"],
            ...     "fidelity_level"
            ... )
            ValueError: Invalid fidelity_level: 'DETALED'
            💡 Did you mean 'DETAILED'?
        """
        matches = difflib.get_close_matches(
            invalid_value, valid_values, n=n_suggestions, cutoff=cutoff
        )

        msg = f"Invalid {enum_name}: '{invalid_value}'"

        if matches:
            if len(matches) == 1:
                msg += f"\n💡 Did you mean '{matches[0]}'?"
            else:
                suggestions = "', '".join(matches)
                msg += f"\n💡 Did you mean one of: '{suggestions}'?"

        # Show valid options
        valid_str = ", ".join(valid_values)
        msg += f"\n   Valid options: {valid_str}"

        return ValueError(msg)

    @staticmethod
    def invalid_parameter(
        param_name: str,
        param_value: Any,
        expected_type: str,
        constraints: Optional[str] = None,
    ) -> ValueError:
        """
        Generate error for invalid parameter value.

        Args:
            param_name: Name of the parameter
            param_value: The invalid value provided
            expected_type: Description of expected type (e.g., "positive integer", "string")
            constraints: Optional description of constraints (e.g., "between 1-1000")

        Returns:
            ValueError with helpful message

        Example:
            >>> raise SmartError.invalid_parameter(
            ...     "num_nodes",
            ...     -5,
            ...     "positive integer",
            ...     "between 1-1000"
            ... )
            ValueError: Invalid num_nodes: -5
            💡 Expected: positive integer (between 1-1000)
        """
        msg = f"Invalid {param_name}: {param_value}"
        msg += f"\n💡 Expected: {expected_type}"

        if constraints:
            msg += f" ({constraints})"

        return ValueError(msg)

    @staticmethod
    def missing_required_field(field_name: str, parent_object: str = "input") -> ValueError:
        """
        Generate error for missing required field.

        Args:
            field_name: Name of the missing field
            parent_object: Context where field was expected (default: "input")

        Returns:
            ValueError with helpful message

        Example:
            >>> raise SmartError.missing_required_field("file_id", "ingest_context")
            ValueError: Missing required field 'file_id' in ingest_context
            💡 Tip: This field is required and cannot be omitted
        """
        msg = f"Missing required field '{field_name}' in {parent_object}"
        msg += "\n💡 Tip: This field is required and cannot be omitted"
        return ValueError(msg)


# Convenience functions for backward compatibility
def suggest_file_id(invalid_id: str, available_ids: List[str]) -> str:
    """
    Get fuzzy-matched suggestions for file_id.

    Args:
        invalid_id: The file_id that wasn't found
        available_ids: List of valid file_ids

    Returns:
        Formatted suggestion string or empty string if no matches

    Example:
        >>> suggest_file_id("quantum_papper", ["quantum_paper", "neural_nets"])
        "Did you mean 'quantum_paper'?"
    """
    matches = difflib.get_close_matches(invalid_id, available_ids, n=1, cutoff=0.6)
    if matches:
        return f"Did you mean '{matches[0]}'?"
    return ""


def suggest_node_id(invalid_id: str, available_ids: List[str]) -> str:
    """
    Get fuzzy-matched suggestions for node_id.

    Args:
        invalid_id: The node_id that wasn't found
        available_ids: List of valid node_ids

    Returns:
        Formatted suggestion string or empty string if no matches

    Example:
        >>> suggest_node_id("quantum_paper_n99", ["quantum_paper_n0", "quantum_paper_n1"])
        "Did you mean 'quantum_paper_n1'?"
    """
    matches = difflib.get_close_matches(invalid_id, available_ids, n=1, cutoff=0.6)
    if matches:
        return f"Did you mean '{matches[0]}'?"
    return ""


def suggest_enum(invalid_value: str, valid_values: List[str]) -> str:
    """
    Get fuzzy-matched suggestions for enum value.

    Args:
        invalid_value: The value that wasn't recognized
        valid_values: List of valid enum values

    Returns:
        Formatted suggestion string or empty string if no matches

    Example:
        >>> suggest_enum("DETALED", ["ABSTRACT", "DETAILED", "RAW"])
        "Did you mean 'DETAILED'?"
    """
    matches = difflib.get_close_matches(invalid_value, valid_values, n=1, cutoff=0.6)
    if matches:
        return f"Did you mean '{matches[0]}'?"
    return ""
