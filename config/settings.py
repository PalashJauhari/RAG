from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from .env and process environment."""

    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    openai_summary_model: str = "gpt-4.1-mini"
    openai_temperature: float = 0

    orchestrator_model: str = "gpt-4.1-mini"
    information_evaluator_model: str = "gpt-4.1-mini"
    final_answer_model: str = "gpt-4.1-mini"
    query_decomposition_model: str = "gpt-4.1-mini"
    query_expansion_model: str = "gpt-4.1-mini"

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = ""
    qdrant_dense_vector_name: str = "dense"
    qdrant_bm25_vector_name: str = "bm25"
    qdrant_colbert_vector_name: str = "colbert"
    qdrant_bm25_model: str = "Qdrant/bm25"

    use_bm25: bool = True
    use_late_interaction: bool = True
    use_mmr: bool = True

    retrieval_top_k: int = 8
    retrieval_candidate_limit: int = 100
    retrieval_mmr_diversity: float = 0.5

    jina_api_key: str = ""
    jina_colbert_model: str = "jina-colbert-v2"
    jina_colbert_dimensions: int = 128
    jina_multi_vector_url: str = "https://api.jina.ai/v1/multi-vector"

    langfuse_tracing_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_base_url: str = ""

    checkpointer_use_postgres: bool = False
    database_url: str = ""

    graph_recursion_limit: int = 100
    graph_max_concurrency: int = 2
    information_evaluation_max_retries: int = 5

    message_summary_token_threshold: int = 100000
    message_summary_keep_recent: int = 10

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
