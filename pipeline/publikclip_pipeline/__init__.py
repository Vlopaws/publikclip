"""publikclip pipeline: long video in, scored vertical clips out.

Everything heavy runs locally. The only network calls are ingest downloads,
model-weight fetches, and the scoring/music LLM calls — three *kinds* of
call, but roughly 35 T1 scorings plus two per finalist, so on the order of
60 requests for an hour-long source. Every one is cached on its prompt, so a
re-run of the same job spends nothing.

The backend is yours: Gemini, any OpenAI-compatible endpoint (NVIDIA Build,
OpenRouter, OpenAI, Groq, a self-hosted vLLM), or Ollama for fully local and
free. See scoring/llm.py.
"""

__version__ = "0.1.0"
