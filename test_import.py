"""
Test script to validate pdf_extractor.litellm package imports.
"""
import sys
import asyncio
from pathlib import Path

async def test_imports():
    """Test importing all major components of the litellm package."""
    try:
        # Core imports
        from pdf_extractor.litellm import (
            litellm_call,
            handle_complex_query,
            BatchRequest,
            BatchResponse,
            TaskItem,
            ResultItem,
            LessonQueryRequest,
            LessonQueryResponse,
            LessonResultItem,
            substitute_results,
            substitute_placeholders,
            retry_llm_call
        )

        # Utils imports (Check if needed based on __init__)
        from pdf_extractor.litellm.utils import (
            create_embedding_with_openai,
            load_text_file,
            get_project_root,
            load_env_file,
            process_image_input,
            compress_image,
            decode_base64_image,
            convert_image_to_base64,
            get_spacy_model,
            count_tokens,
            truncate_text_by_tokens,
            truncate_vector_for_display,
            format_embedding_for_debug,
            get_vector_stats
        )

        # DB utils imports (Check if needed based on __init__)
        from pdf_extractor.litellm.utils.db import (
            connect_to_arango_client,
            insert_object,
            handle_relationships,
            get_lessons,
            upsert_lesson,
            update_lesson,
            delete_lesson,
            query_lessons_by_keyword,
            query_lessons_by_concept,
            query_lessons_by_similarity
        )

        print("✅ All imports successful")
        return True

    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    # Ensure src directory is in the Python path
    # This might be necessary if running from the root directory
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path.parent)) # Add project root

    print(f"Python Path: {sys.path}") # Debug print

    success = asyncio.run(test_imports())
    sys.exit(0 if success else 1)