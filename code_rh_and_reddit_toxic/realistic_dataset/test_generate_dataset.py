#!/usr/bin/env python3
"""
End-to-end tests for generate_dataset.py

These tests run the script once and verify all expected behaviors.
"""

import pytest
from realistic_dataset.generate_dataset import CMVDatasetProcessor


def test_normalize_text():
    """Test that normalize_text correctly cleans special characters and decodes HTML entities."""
    normalize = CMVDatasetProcessor.normalize_text

    assert normalize("“Smart” and ‘dumb’") == "\"Smart\" and 'dumb'"

    assert normalize("Wait… what?") == "Wait... what?"

    assert (
        normalize("This is important — really important.")
        == "This is important - really important."
    )
    assert normalize("Range: 1–10") == "Range: 1-10"

    # 5. Non-breaking space
    assert (
        normalize("This\u00A0is\u00A0a\u00A0test") == "This is a test"
    ) 

    # 6. Unicode punctuation (e.g. “”, ‘’)
    assert normalize("“Quote” and ‘apostrophe’") == "\"Quote\" and 'apostrophe'"

    assert normalize("test–dash") == "test-dash"
    assert normalize("test—dash") == "test-dash"
    assert normalize("wait…") == "wait..."

    # 7. HTML entity for zero-width space
    assert normalize("foo&#x200B;bar") == "foobar"
    # 8. HTML entity for non-breaking space
    assert normalize("foo&nbsp;bar") == "foo bar"
    # 9. Actual zero-width space character
    assert normalize("foo\u200Bbar") == "foobar"
    # 10. Actual zero-width non-joiner character
    assert normalize("foo\u200Cbar") == "foobar"
    # 11. Actual zero-width joiner character
    assert normalize("foo\u200Dbar") == "foobar"
    # 12. Actual zero-width no-break space
    assert normalize("foo\ufeffbar") == "foobar"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
