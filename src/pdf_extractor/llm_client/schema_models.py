# Schema Models for LLM Responses
#
# This module defines Pydantic models that represent structured responses from LLMs.
# These models are used for response validation and schema enforcement.

from typing import Optional
from pydantic import BaseModel, Field

class QuestionAnswer(BaseModel):
    """Model for question-answering responses."""
    question: str = Field(description="The original question asked")
    answer: str = Field(description="The answer to the question - exact only")
    confidence: str = Field(description="Confidence level: high, medium, or low")

class CodeResponse(BaseModel):
    """Model for code generation responses."""
    code: str = Field(description="The generated code")
    language: str = Field(description="The programming language")
    explanation: Optional[str] = Field(None, description="How the code works")

if __name__ == "__main__":
    import sys
    import json
    
    # Test data
    qa_data = {
        "question": "What is 2+2?",
        "answer": "4",
        "confidence": "high"
    }
    
    code_data = {
        "code": "def add(a, b): return a + b",
        "language": "python",
        "explanation": "A simple function that adds two numbers"
    }
    
    tests_passed_count = 0
    tests_failed_count = 0
    total_tests = 3 # QA Instantiation, CodeResponse Instantiation, JSON Test
    
    print("--- Running Schema Model Validation ---")

    # Initialize model variable to satisfy static analysis
    qa_model = None
    
    # 1. Validate QuestionAnswer model instantiation and attributes
    qa_test_passed = False
    try:
        qa_model = QuestionAnswer(**qa_data)
        assert qa_model.question == "What is 2+2?", f"QA Question mismatch: Expected 'What is 2+2?', Got '{qa_model.question}'"
        assert qa_model.answer == "4", f"QA Answer mismatch: Expected '4', Got '{qa_model.answer}'"
        assert qa_model.confidence == "high", f"QA Confidence mismatch: Expected 'high', Got '{qa_model.confidence}'"
        tests_passed_count += 1
        qa_test_passed = True
        print("✅ Test 'QuestionAnswer Instantiation': PASSED")
    except Exception as e:
        tests_failed_count += 1
        print(f"❌ Test 'QuestionAnswer Instantiation': FAILED - {e}")
        
    # 2. Validate CodeResponse model instantiation and attributes
    code_test_passed = False
    try:
        code_model = CodeResponse(**code_data)
        assert code_model.code == "def add(a, b): return a + b", f"Code mismatch: Expected 'def add(a, b): return a + b', Got '{code_model.code}'"
        assert code_model.language == "python", f"Language mismatch: Expected 'python', Got '{code_model.language}'"
        assert code_model.explanation == "A simple function that adds two numbers", f"Explanation mismatch: Expected 'A simple function...', Got '{code_model.explanation}'"
        tests_passed_count += 1
        code_test_passed = True
        print("✅ Test 'CodeResponse Instantiation': PASSED")
    except Exception as e:
        tests_failed_count += 1
        print(f"❌ Test 'CodeResponse Instantiation': FAILED - {e}")
    
    # 3. Test serialization and deserialization (only if QA model instantiated correctly)
    json_test_passed = False
    # Check if the prerequisite test passed before attempting JSON tests
    if qa_test_passed:
        # Add explicit check for None to satisfy Pylance
        if qa_model is not None:
            try:
                # Serialize
                qa_json = qa_model.model_dump_json()
                # Deserialize
                qa_from_json = QuestionAnswer.model_validate_json(qa_json)
                # Verify
                assert qa_from_json.question == qa_model.question, "JSON Deserialized Question mismatch"
                assert qa_from_json.answer == qa_model.answer, "JSON Deserialized Answer mismatch"
                assert qa_from_json.confidence == qa_model.confidence, "JSON Deserialized Confidence mismatch"
                tests_passed_count += 1
                json_test_passed = True
                print("✅ Test 'JSON Serialization/Deserialization': PASSED")
            except Exception as e:
                tests_failed_count += 1
                print(f"❌ Test 'JSON Serialization/Deserialization': FAILED - {e}")
        else:
            # This case should theoretically not be reached due to qa_test_passed check,
            # but handle defensively and mark test as failed.
            tests_failed_count += 1
            print("❌ Test 'JSON Serialization/Deserialization': FAILED - qa_model was None despite qa_test_passed being True.")
    else:
        # Mark JSON test as failed if prerequisite failed
        tests_failed_count += 1
        print("⚠️ Test 'JSON Serialization/Deserialization': SKIPPED (due to QA Instantiation failure)")


    # --- Report validation status ---
    print(f"\n--- Test Summary ---")
    print(f"Total Tests Attempted: {total_tests}") # Note: JSON test might be skipped but counted
    print(f"Passed: {tests_passed_count}")
    print(f"Failed: {tests_failed_count}")
    
    if tests_failed_count == 0:
        print("\n✅ VALIDATION COMPLETE - All schema model tests passed.")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED - Some schema model tests failed.")
        sys.exit(1)