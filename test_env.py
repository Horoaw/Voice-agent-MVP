import sys

def test_imports():
    print("Testing core imports...")
    try:
        from faster_whisper import WhisperModel
        print("✓ Faster-Whisper OK")
    except Exception as e:
        print(f"✗ Faster-Whisper Failed: {e}")

    try:
        import edge_tts
        print("✓ edge-tts OK")
    except Exception as e:
        print(f"✗ edge-tts Failed: {e}")

    try:
        from langchain_core.prompts import ChatPromptTemplate
        print("✓ langchain-core OK")
    except Exception as e:
        print(f"✗ langchain-core Failed: {e}")

    try:
        # Test new or old imports
        try:
            from langchain_ollama import ChatOllama
        except:
            from langchain_community.chat_models import ChatOllama
        print("✓ Ollama LLM OK")
    except Exception as e:
        print(f"✗ Ollama LLM Failed: {e}")

    try:
        from fastapi import FastAPI
        print("✓ FastAPI OK")
    except Exception as e:
        print(f"✗ FastAPI Failed: {e}")

if __name__ == "__main__":
    test_imports()
