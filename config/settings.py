"""Application runtime settings loaded from ``.env`` and environment variables.

Consumed by FastAPI, :class:`~graph.graph.RetrievalGraph`, :class:`~retriever.retriever.Retriever`,
and middleware. HotpotQA benchmarks use a separate settings module under ``benchmarking/hotpotqa/``.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pydantic settings for the production RAG pipeline (see ``.env.example``)."""

    # --- OpenAI defaults ---
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    openai_summary_model: str = "gpt-4.1-mini"
    openai_temperature: float = 0

    # --- Per-graph-node LLM models ---
    query_normalisation_model: str = "gpt-5-mini"
    recall_check_model: str = "gpt-5.1"
    intent_check_model: str = "gpt-5-mini"
    final_answer_model: str = "gpt-5.1"
    query_decomposition_model: str = "gpt-5-mini"
    gap_fill_model: str = "gpt-5-mini"
    intent_correction_rewriter_model: str = "gpt-5-mini"

    # --- Qdrant collection and vector names ---
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = ""
    qdrant_dense_vector_name: str = "dense"
    qdrant_bm25_vector_name: str = "bm25"
    qdrant_colbert_vector_name: str = "colbert"
    qdrant_bm25_model: str = "Qdrant/bm25"

    # --- Retrieval feature flags (client setup; strategy picks branches per request) ---
    use_bm25: bool = True
    use_late_interaction: bool = True
    use_mmr: bool = True

    retrieval_top_k: int = 8
    retrieval_top_k_max: int = 64
    retrieval_candidate_dense_mmr: int = 100
    retrieval_candidate_dense_mmr_max: int = 500
    retrieval_candidate_bm25: int = 100
    retrieval_candidate_bm25_max: int = 500
    retrieval_candidate_for_late_interaction: int = 100
    retrieval_candidate_for_late_interaction_max: int = 500
    retrieval_mmr_diversity: float = 0.5

    # --- Jina ColBERT multi-vector API ---
    jina_api_key: str = ""
    jina_colbert_model: str = "jina-colbert-v2"
    jina_colbert_dimensions: int = 128
    jina_multi_vector_url: str = "https://api.jina.ai/v1/multi-vector"

    # --- Langfuse observability ---
    langfuse_tracing_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_base_url: str = ""

    # --- LangGraph checkpointing ---
    checkpointer_use_postgres: bool = False
    database_url: str = ""

    # --- Graph execution limits ---
    graph_recursion_limit: int = 100
    graph_max_concurrency: int = 2
    retrieval_loop_max_retries: int = 3

    # --- Long-context summarization (optional; see middleware.context_editing) ---
    message_summary_token_threshold: int = 100000
    message_summary_keep_recent: int = 10

    # --- OpenAI request rate limiting ---
    openai_rate_limit_enabled: bool = True
    openai_rate_limit_requests_per_second: float = 1.0
    openai_rate_limit_check_every_n_seconds: float = 0.1
    openai_rate_limit_max_bucket_size: float = 5.0

    request_timeout_seconds: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
