"""Prepare normalized HotpotQA validation JSON for retrieval benchmarking.

Downloads ``hotpotqa/hotpot_qa`` (fullwiki validation), optionally subsamples with
deterministic stratified sampling (seed 42) by ``(type, level)``, enriches each context
with LLM metadata, and writes ``data/processed/hotpotqa_eval.json``.

Run: ``python -m benchmarking.hotpotqa.dataset.prepare_eval_data``
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import defaultdict
from typing import Any

from datasets import load_dataset
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import propagate_attributes

from benchmarking.hotpotqa.output_validation.context_enrichment import ContextEnrichmentResult
from benchmarking.hotpotqa.prompts.context_enrichment import SYSTEM_PROMPT
from benchmarking.hotpotqa.settings import HotpotQASettings
from benchmarking.hotpotqa import tracing as hotpotqa_tracing
from middleware.llm_client import get_llm_client

logger = logging.getLogger(__name__)


def build_records(dataset) -> list[dict[str, Any]]:
    """Load HF rows into benchmark JSON records with joined ``text`` per context."""

    records: list[dict[str, Any]] = []
    for row in dataset:
        support = row["supporting_facts"]
        support_by_title: dict[str, set[int]] = {}
        for title, sent_id in zip(support["title"], support["sent_id"]):
            support_by_title.setdefault(title, set()).add(sent_id)

        contexts = []
        for index, (title, sentences) in enumerate(
            zip(row["context"]["title"], row["context"]["sentences"])
        ):
            supporting_sentence_ids = sorted(support_by_title.get(title, set()))
            text = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
            contexts.append(
                {
                    "context_id": f"{row['id']}:{index}",
                    "title": title,
                    "sentences": sentences,
                    "text": text,
                    "is_supporting": bool(supporting_sentence_ids),
                    "supporting_sentence_ids": supporting_sentence_ids,
                }
            )

        records.append(
            {
                "id": row["id"],
                "question": row["question"],
                "reference_answer": row["answer"],
                "type": row["type"],
                "level": row["level"],
                "supporting_facts": [
                    [title, sent_id] for title, sent_id in zip(support["title"], support["sent_id"])
                ],
                "contexts": contexts,
            }
        )
    return records


async def enrich_context(
    context: dict[str, Any],
    *,
    settings: HotpotQASettings,
    llm: Any,
    model: str,
    max_retries: int,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
    batch_span: Any | None = None,
) -> None:
    """Attach ``enrichment`` to one context; set null after retries exhausted."""

    passage = str(context.get("text") or "").strip()
    if not passage:
        context["enrichment"] = None
        return

    async with semaphore:
        for attempt in range(1, max_retries + 1):
            started = time.perf_counter()
            try:
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=passage),
                ]
                if batch_span is not None:
                    gen = hotpotqa_tracing.start_parallel_generation(
                        batch_span,
                        name="context_enrichment-llm",
                        model=model,
                        input_text=passage,
                        metadata={
                            "context_id": context.get("context_id"),
                            "attempt": attempt,
                            "index": index,
                        },
                    )
                    finished = False
                    try:
                        result = await llm.ainvoke(messages)
                        parsed: ContextEnrichmentResult = result["parsed"]
                        hotpotqa_tracing.finish_generation(
                            gen,
                            model=model,
                            raw=result.get("raw"),
                            output=parsed.model_dump(),
                            metadata={
                                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                                "context_id": context.get("context_id"),
                                "attempt": attempt,
                            },
                        )
                        finished = True
                    except Exception as exc:
                        gen.update(level="ERROR", status_message=str(exc))
                        raise
                    finally:
                        if not finished:
                            gen.end()
                else:
                    result = await llm.ainvoke(messages)
                    parsed = result["parsed"]

                context["enrichment"] = parsed.model_dump()
                print(f"Enriched {index}/{total} ({context.get('context_id')})")
                return
            except Exception as exc:
                if attempt >= max_retries:
                    logger.warning(
                        "Enrichment failed for %s after %s attempts: %s",
                        context.get("context_id"),
                        max_retries,
                        exc,
                    )
                    context["enrichment"] = None
                    print(f"Enriched {index}/{total} ({context.get('context_id')}) — skipped")
                    return
                await asyncio.sleep(0.5 * attempt)


async def enrich_all_contexts(records: list[dict[str, Any]], settings: HotpotQASettings) -> None:
    """Run parallel LLM enrichment for every context in ``records``."""

    contexts: list[dict[str, Any]] = []
    for record in records:
        for context in record.get("contexts") or []:
            if str(context.get("text") or "").strip():
                contexts.append(context)

    if not contexts:
        return

    llm = get_llm_client(
        model=settings.hotpotqa_context_enrichment_model,
        output_schema=ContextEnrichmentResult,
        include_raw=True,
    )
    model = settings.hotpotqa_context_enrichment_model
    semaphore = asyncio.Semaphore(settings.hotpotqa_enrichment_concurrency)
    total = len(contexts)

    if hotpotqa_tracing.langfuse_enabled(settings):
        langfuse = hotpotqa_tracing.get_client(settings)
        with langfuse.start_as_current_observation(
            as_type="span",
            name="hotpotqa_context_enrichment",
        ) as batch_span:
            with propagate_attributes(
                session_id="hotpotqa_prepare",
                metadata={"pipeline": "context_enrichment"},
            ):
                batch_span.update(
                    input={
                        "context_count": total,
                        "model": model,
                        "concurrency": settings.hotpotqa_enrichment_concurrency,
                    }
                )
                await asyncio.gather(
                    *[
                        enrich_context(
                            context,
                            settings=settings,
                            llm=llm,
                            model=model,
                            max_retries=settings.hotpotqa_enrichment_max_retries,
                            semaphore=semaphore,
                            index=index,
                            total=total,
                            batch_span=batch_span,
                        )
                        for index, context in enumerate(contexts, start=1)
                    ]
                )
                batch_span.update(
                    output={
                        "context_count": total,
                        "enriched": sum(1 for context in contexts if context.get("enrichment")),
                    }
                )
        hotpotqa_tracing.flush()
    else:
        await asyncio.gather(
            *[
                enrich_context(
                    context,
                    settings=settings,
                    llm=llm,
                    model=model,
                    max_retries=settings.hotpotqa_enrichment_max_retries,
                    semaphore=semaphore,
                    index=index,
                    total=total,
                )
                for index, context in enumerate(contexts, start=1)
            ]
        )


def main() -> None:
    """Load HF dataset, enrich contexts, write JSON."""

    logging.basicConfig(level=logging.INFO)
    settings = HotpotQASettings()
    dataset = load_dataset(
        settings.hotpotqa_dataset_name,
        settings.hotpotqa_dataset_config,
        split=settings.hotpotqa_split,
    )

    if settings.hotpotqa_max_questions > 0:
        groups = defaultdict(list)
        for i, row in enumerate(dataset):
            groups[(row["type"], row["level"])].append(i)

        keys = sorted(groups.keys())
        random.seed(42)
        for key in keys:
            random.shuffle(groups[key])

        target_per_group = settings.hotpotqa_max_questions // len(keys)
        selected_indices: list[int] = []
        for key in keys:
            selected_indices.extend(groups[key][:target_per_group])

        dataset = dataset.select(sorted(selected_indices))

    records = build_records(dataset)
    print(f"Enriching contexts for {len(records)} questions...")
    asyncio.run(enrich_all_contexts(records, settings))

    settings.processed_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    settings.processed_dataset_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} records to {settings.processed_dataset_path}")


if __name__ == "__main__":
    main()
