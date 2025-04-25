#!/usr/bin/env python3
"""
Test script for validating CLI scenarios in cli_scenarios.md
This script uses the mock implementation to avoid actual database operations.
"""
import os
import sys
import subprocess
from loguru import logger

# Set up logging
logger.remove()
logger.add(sys.stderr, level="INFO")

def run_cli_command(command):
    """Run a CLI command and return the output."""
    # Use the mock CLI implementation
    full_command = f"cd /home/graham/workspace/experiments/pdf_extractor/ && source .venv/bin/activate && python -m src.pdf_extractor.arangodb.search_cli_mock {command}"
    logger.info(f"Running command: {command}")
    
    try:
        result = subprocess.run(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        return e.stdout, e.stderr

def test_scenario_1():
    """Test Scenario 1: Basic Message Search"""
    logger.info("=== Testing Scenario 1: Basic Message Search ===")
    command = 'search messages "configuration"'
    stdout, stderr = run_cli_command(command)
    
    # Check if the output contains expected content
    expected_outputs = ["Found", "message", "configuration"]
    success = all(output in stdout for output in expected_outputs)
    
    if success:
        logger.info("✅ PASS: Basic Message Search")
    else:
        logger.error("❌ FAIL: Basic Message Search")
        logger.error(f"Output: {stdout}")
    
    print(f"Output for Scenario 1:\n{stdout}")
    return success

def test_scenario_2():
    """Test Scenario 2: Message Search with Type Filter"""
    logger.info("=== Testing Scenario 2: Message Search with Type Filter ===")
    command = 'search messages "user question" --message-type USER'
    stdout, stderr = run_cli_command(command)
    
    # Check if the output contains expected content
    expected_outputs = ["Found", "[USER]", "user question"]
    unexpected_outputs = ["[AGENT]", "[SYSTEM]"]
    success = (all(output in stdout for output in expected_outputs) and 
               not any(output in stdout for output in unexpected_outputs))
    
    if success:
        logger.info("✅ PASS: Message Search with Type Filter")
    else:
        logger.error("❌ FAIL: Message Search with Type Filter")
        logger.error(f"Output: {stdout}")
    
    print(f"Output for Scenario 2:\n{stdout}")
    return success

def test_scenario_3():
    """Test Scenario 3: Message Search with Conversation Filter"""
    logger.info("=== Testing Scenario 3: Message Search with Conversation Filter ===")
    command = 'search messages "" --conversation 12345-abcde-67890'
    stdout, stderr = run_cli_command(command)
    
    # Check if the output contains expected content
    expected_outputs = ["Found", "12345-abcde-67890"]
    success = all(output in stdout for output in expected_outputs)
    
    if success:
        logger.info("✅ PASS: Message Search with Conversation Filter")
    else:
        logger.error("❌ FAIL: Message Search with Conversation Filter")
        logger.error(f"Output: {stdout}")
    
    print(f"Output for Scenario 3:\n{stdout}")
    return success

def test_scenario_4():
    """Test Scenario 4: Unified Search Across Collections"""
    logger.info("=== Testing Scenario 4: Unified Search Across Collections ===")
    command = 'search unified "error handling"'
    stdout, stderr = run_cli_command(command)
    
    # Check if the output contains expected content
    expected_outputs = ["Found", "Collections searched:", "documents", "messages", "error handling"]
    success = all(output in stdout for output in expected_outputs)
    
    if success:
        logger.info("✅ PASS: Unified Search Across Collections")
    else:
        logger.error("❌ FAIL: Unified Search Across Collections")
        logger.error(f"Output: {stdout}")
    
    print(f"Output for Scenario 4:\n{stdout}")
    return success

def test_scenario_5():
    """Test Scenario 5: Search Only Documents"""
    logger.info("=== Testing Scenario 5: Search Only Documents ===")
    command = 'search unified "api reference" --collections documents'
    stdout, stderr = run_cli_command(command)
    
    # Check if the output contains expected content
    expected_outputs = ["Found", "Collections searched: documents", "api reference"]
    unexpected_outputs = ["messages"]
    success = (all(output in stdout for output in expected_outputs) and 
               not any(output in stdout for output in unexpected_outputs))
    
    if success:
        logger.info("✅ PASS: Search Only Documents")
    else:
        logger.error("❌ FAIL: Search Only Documents")
        logger.error(f"Output: {stdout}")
    
    print(f"Output for Scenario 5:\n{stdout}")
    return success

def test_scenario_6():
    """Test Scenario 6: Exclude Messages from Search"""
    logger.info("=== Testing Scenario 6: Exclude Messages from Search ===")
    command = 'search unified "security protocol" --exclude messages'
    stdout, stderr = run_cli_command(command)
    
    # Check if the output contains expected content
    expected_outputs = ["Found", "Collections searched:", "documents", "security protocol"]
    unexpected_outputs = ["messages"]
    success = (all(output in stdout for output in expected_outputs) and 
               not any(output in stdout for output in unexpected_outputs))
    
    if success:
        logger.info("✅ PASS: Exclude Messages from Search")
    else:
        logger.error("❌ FAIL: Exclude Messages from Search")
        logger.error(f"Output: {stdout}")
    
    print(f"Output for Scenario 6:\n{stdout}")
    return success

def test_scenario_7():
    """Test Scenario 7: Semantic Search"""
    logger.info("=== Testing Scenario 7: Semantic Search ===")
    command = 'search unified "improving system performance" --type semantic'
    stdout, stderr = run_cli_command(command)
    
    # Check if the output contains expected content
    expected_outputs = ["Found", "Collections searched:", "improving system performance"]
    success = all(output in stdout for output in expected_outputs)
    
    if success:
        logger.info("✅ PASS: Semantic Search")
    else:
        logger.error("❌ FAIL: Semantic Search")
        logger.error(f"Output: {stdout}")
    
    print(f"Output for Scenario 7:\n{stdout}")
    return success

def test_scenario_8():
    """Test Scenario 8: JSON Output"""
    logger.info("=== Testing Scenario 8: JSON Output ===")
    command = 'search messages "database error" --json'
    stdout, stderr = run_cli_command(command)
    
    # Check if the output contains expected JSON elements
    expected_outputs = ['"results":', '"content":', '"database error"', '"score":', '"query":', '"count":']
    success = all(output in stdout for output in expected_outputs)
    
    if success:
        logger.info("✅ PASS: JSON Output")
    else:
        logger.error("❌ FAIL: JSON Output")
        logger.error(f"Output: {stdout}")
    
    print(f"Output for Scenario 8 (partial):\n{stdout[:200]}...")
    return success

def main():
    """Run all test scenarios."""
    logger.info("Testing CLI Scenarios...")
    
    test_results = [
        test_scenario_1(),
        test_scenario_2(),
        test_scenario_3(),
        test_scenario_4(),
        test_scenario_5(),
        test_scenario_6(),
        test_scenario_7(),
        test_scenario_8()
    ]
    
    # Print summary
    scenario_count = len(test_results)
    pass_count = sum(test_results)
    
    logger.info("=" * 50)
    logger.info(f"Test Summary: {pass_count}/{scenario_count} scenarios passed")
    for i, result in enumerate(test_results, 1):
        logger.info(f"Scenario {i}: {'✅ PASS' if result else '❌ FAIL'}")
    logger.info("=" * 50)
    
    if pass_count == scenario_count:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error(f"❌ {scenario_count - pass_count} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
