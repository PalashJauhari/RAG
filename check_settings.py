"""Development utility: list ``Settings`` fields with no ``settings.<field>`` usage in ``*.py``.

Not used in production. Run from repo root: ``python check_settings.py``.
"""

import os
import re

fields = [
    "openai_api_key", "openai_llm_model", "openai_embedding_model", "openai_embedding_dimensions",
    "openai_summary_model", "openai_temperature", "qdrant_url", "qdrant_api_key", "qdrant_collection_name",
    "query_normalisation_model", "query_complexity_model", "query_rewriter_model",
    "recall_check_model", "intent_check_model", "fact_gap_query_model", "strategy_upgrade_model",
    "final_answer_model", "query_decomposition_model",
    "query_expansion_model", "gap_fill_model", "intent_correction_rewriter_model",
    "qdrant_dense_vector_name", "qdrant_bm25_vector_name", "qdrant_colbert_vector_name", "qdrant_bm25_model",
    "use_bm25", "use_late_interaction", "use_mmr", "retrieval_top_k", "retrieval_candidate_limit",
    "retrieval_mmr_diversity", "jina_api_key", "jina_colbert_model", "jina_colbert_dimensions",
    "jina_multi_vector_url", "langfuse_tracing_enabled", "langfuse_public_key", "langfuse_secret_key",
    "langfuse_host", "langfuse_base_url", "checkpointer_use_postgres", "database_url",
    "graph_recursion_limit", "graph_max_concurrency", "retrieval_loop_max_retries",
    "message_summary_token_threshold",
    "message_summary_keep_recent", "openai_rate_limit_enabled", "openai_rate_limit_requests_per_second",
    "openai_rate_limit_check_every_n_seconds", "openai_rate_limit_max_bucket_size",
    "request_timeout_seconds",
    "hotpotqa_dataset_name", "hotpotqa_dataset_config", "hotpotqa_split", "hotpotqa_max_questions",
    "hotpotqa_upload_batch_size", "hotpotqa_ragas_model",
    "processed_dataset_path", "retrieval_results_path", "ragas_results_path"
]

unused = []
for f in fields:
    # search across all .py files excluding check_settings.py and the settings files where they are defined
    cmd = f'grep -r "\\.{f}" . --include="*.py" | grep -v "config/settings.py" | grep -v "benchmarking/hotpotqa/settings.py" | grep -v "check_settings.py"'
    res = os.popen(cmd).read().strip()
    # also check if accessed as dict or other ways? Usually config properties are accessed as `settings.foo` or `self.config.foo`
    if not res:
        unused.append(f)

print("Unused config properties found:")
if unused:
    for u in unused:
        print(u)
else:
    print("(none)")
