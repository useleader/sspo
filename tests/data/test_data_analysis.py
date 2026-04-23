#!/usr/bin/env python3
"""
Tests for data analysis - verifies downloaded data characteristics.

These tests validate:
1. Dataset sizes match paper expectations
2. Sample structure is correct
3. Field types are as expected
4. No corrupted entries
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import pytest


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def ultrafeedback_path():
    return Path("data/ultrafeedback/train.json")


@pytest.fixture
def ultrachat_train_path():
    return Path("data/ultrachat/train_sft.json")


@pytest.fixture
def ultrafeedback_data(ultrafeedback_path):
    """Load UltraFeedback data as JSONL."""
    if not ultrafeedback_path.exists():
        pytest.skip("UltraFeedback not downloaded")
    data = []
    with open(ultrafeedback_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


@pytest.fixture
def ultrachat_data(ultrachat_train_path):
    """Load UltraChat data as JSONL."""
    if not ultrachat_train_path.exists():
        pytest.skip("UltraChat not downloaded")
    data = []
    with open(ultrachat_train_path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ============================================================================
# Dataset Size Tests (from Paper)
# ============================================================================

class TestDatasetSize:
    """Verify dataset sizes match paper expectations."""

    def test_ultrafeedback_has_expected_samples(self, ultrafeedback_data):
        """Paper: UltraFeedback has ~64K samples (256K with 4 responses each)."""
        # The dataset should have tens of thousands of samples
        n = len(ultrafeedback_data)
        assert 50000 < n < 100000, \
            f"UltraFeedback should have ~64K samples, got {n}"

    def test_ultrachat_has_expected_samples(self, ultrachat_data):
        """Paper: UltraChat has ~200K samples."""
        n = len(ultrachat_data)
        assert 150000 < n < 250000, \
            f"UltraChat should have ~200K samples, got {n}"


# ============================================================================
# UltraFeedback Structure Tests
# ============================================================================

class TestUltraFeedbackStructure:
    """Verify UltraFeedback has correct structure."""

    def test_has_instruction_field(self, ultrafeedback_data):
        """UltraFeedback should have instruction field."""
        sample = ultrafeedback_data[0]
        assert "instruction" in sample, "Missing 'instruction' field"

    def test_has_preference_fields(self, ultrafeedback_data):
        """UltraFeedback should have chosen_response and rejected_response."""
        sample = ultrafeedback_data[0]
        assert "chosen_response" in sample, "Missing 'chosen_response' field"
        assert "rejected_response" in sample, "Missing 'rejected_response' field"

    def test_chosen_has_rating(self, ultrafeedback_data):
        """Chosen response should have a rating score."""
        sample = ultrafeedback_data[0]
        assert "chosen_avg_rating" in sample, "Missing 'chosen_avg_rating'"
        assert isinstance(sample["chosen_avg_rating"], (int, float)), \
            "chosen_avg_rating should be numeric"

    def test_rejected_has_rating(self, ultrafeedback_data):
        """Rejected response should have a rating score."""
        sample = ultrafeedback_data[0]
        assert "rejected_avg_rating" in sample, "Missing 'rejected_avg_rating'"
        assert isinstance(sample["rejected_avg_rating"], (int, float)), \
            "rejected_avg_rating should be numeric"

    def test_chosen_better_than_rejected(self, ultrafeedback_data):
        """Chosen rating should be higher than rejected rating."""
        for sample in ultrafeedback_data[:100]:
            chosen = sample.get("chosen_avg_rating", 0)
            rejected = sample.get("rejected_avg_rating", 0)
            assert chosen >= rejected, \
                f"Chosen ({chosen}) should >= rejected ({rejected})"

    def test_has_source(self, ultrafeedback_data):
        """UltraFeedback should track source of instruction."""
        sample = ultrafeedback_data[0]
        assert "source" in sample, "Missing 'source' field"

    def test_no_corrupted_entries(self, ultrafeedback_data):
        """Verify no entries are corrupted (missing essential fields)."""
        for i, sample in enumerate(ultrafeedback_data):
            assert "instruction" in sample, f"Sample {i} missing instruction"
            assert "chosen_response" in sample, f"Sample {i} missing chosen_response"
            assert "rejected_response" in sample, f"Sample {i} missing rejected_response"


# ============================================================================
# UltraChat Structure Tests
# ============================================================================

class TestUltraChatStructure:
    """Verify UltraChat has correct structure."""

    def test_has_messages_field(self, ultrachat_data):
        """UltraChat should have messages field (conversation)."""
        sample = ultrachat_data[0]
        assert "messages" in sample, "Missing 'messages' field"
        assert isinstance(sample["messages"], list), \
            "messages should be a list"

    def test_messages_have_roles(self, ultrachat_data):
        """Messages should have role and content fields."""
        sample = ultrachat_data[0]
        for msg in sample["messages"]:
            assert "role" in msg, "Message missing 'role'"
            assert "content" in msg, "Message missing 'content'"

    def test_message_roles_are_valid(self, ultrachat_data):
        """Message roles should be valid (user/assistant/system)."""
        valid_roles = {"user", "assistant", "system", "human", "gpt"}
        for sample in ultrachat_data[:100]:  # Check first 100
            for msg in sample.get("messages", []):
                assert msg["role"] in valid_roles, \
                    f"Invalid role: {msg['role']}"

    def test_conversation_has_multiple_turns(self, ultrachat_data):
        """UltraChat conversations typically have multiple turns."""
        sample = ultrachat_data[0]
        n_messages = len(sample["messages"])
        assert n_messages >= 2, \
            f"Conversation should have multiple turns, got {n_messages}"

    def test_no_corrupted_entries(self, ultrachat_data):
        """Verify no entries are corrupted."""
        for i, sample in enumerate(ultrachat_data):
            assert "messages" in sample, f"Sample {i} missing messages"
            assert len(sample["messages"]) > 0, f"Sample {i} has empty messages"
            for msg in sample["messages"]:
                assert "role" in msg and "content" in msg, \
                    f"Sample {i} message missing role/content"


# ============================================================================
# Data Quality Tests
# ============================================================================

class TestDataQuality:
    """Verify data quality."""

    def test_no_empty_instructions(self, ultrafeedback_data):
        """Instructions should not be empty."""
        for i, sample in enumerate(ultrafeedback_data[:1000]):
            assert sample.get("instruction", "").strip(), \
                f"Sample {i} has empty instruction"

    def test_no_empty_responses(self, ultrafeedback_data):
        """Chosen responses should not be empty (rejected may be empty for some samples)."""
        empty_rejected = 0
        for i, sample in enumerate(ultrafeedback_data[:1000]):
            chosen = sample.get("chosen_response", "")
            rejected = sample.get("rejected_response", "")
            assert chosen.strip(), f"Sample {i} has empty chosen_response"
            if not rejected.strip():
                empty_rejected += 1
        # Some samples may not have rejected responses
        # Log this for awareness but don't fail
        if empty_rejected > 0:
            print(f"Note: {empty_rejected}/1000 samples have empty rejected_response")

    def test_no_empty_messages(self, ultrachat_data):
        """Messages should not have empty content."""
        for i, sample in enumerate(ultrachat_data[:1000]):
            for j, msg in enumerate(sample.get("messages", [])):
                assert msg.get("content", "").strip(), \
                    f"Sample {i}, message {j} is empty"


# ============================================================================
# Length Distribution Tests
# ============================================================================

class TestLengthDistribution:
    """Verify length characteristics for tokenization planning."""

    def test_instruction_length_reasonable(self, ultrafeedback_data):
        """Instructions should be reasonable length (not too short/long)."""
        lengths = []
        for sample in ultrafeedback_data[:1000]:
            inst = sample.get("instruction", "")
            lengths.append(len(inst.split()))  # word count

        avg = sum(lengths) / len(lengths)
        assert 5 < avg < 500, \
            f"Instruction avg word count {avg:.1f} seems off"

    def test_response_length_reasonable(self, ultrafeedback_data):
        """Chosen/rejected responses should be reasonable length."""
        lengths = []
        for sample in ultrafeedback_data[:1000]:
            chosen = sample.get("chosen_response", "")
            rejected = sample.get("rejected_response", "")
            lengths.append(len(chosen.split()))
            lengths.append(len(rejected.split()))

        avg = sum(lengths) / len(lengths)
        assert 10 < avg < 2000, \
            f"Response avg word count {avg:.1f} seems off"

    def test_conversation_length_reasonable(self, ultrachat_data):
        """UltraChat conversations should have reasonable turns."""
        turns = []
        for sample in ultrachat_data[:1000]:
            turns.append(len(sample.get("messages", [])))

        avg = sum(turns) / len(turns)
        assert 2 < avg < 50, \
            f"Conversation avg turns {avg:.1f} seems off"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
