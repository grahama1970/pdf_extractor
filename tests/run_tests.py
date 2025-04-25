#!/usr/bin/env python3
"""
Main test runner for pdf_extractor validation tests.
Runs all tests in the tests/pdf_extractor/arangodb/search_api directory.
"""

import sys
import os
import unittest
import argparse
from loguru import logger

def main():
    """Run the specified tests."""
    parser = argparse.ArgumentParser(description='Run pdf_extractor validation tests')
    parser.add_argument('--pattern', default='test_*.py', help='Test file pattern to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    args = parser.parse_args()

    # Configure logging
    log_level = "ERROR" if args.quiet else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Get the tests directory
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    test_suites = loader.discover(
        start_dir=os.path.join(tests_dir, 'pdf_extractor', 'arangodb', 'search_api'),
        pattern=args.pattern
    )
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(test_suites)
    
    # Report results
    print("\n===== TEST SUMMARY =====")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Exit with status code
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed, see above for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
