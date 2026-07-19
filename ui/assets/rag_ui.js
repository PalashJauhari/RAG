/**
 * Dash clientside SSE consumer for POST /run/stream.
 * Event shape matches api.main.get_stream_event (node, type, label, node-specific fields).
 * Inline progress card: bold = LangGraph node id; remainder = human-readable detail.
 */
window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.rag_ui = window.dash_clientside.rag_ui || {};
window.dash_clientside.rag_ui._inFlight = false;

function setComposerDisabled(disabled) {
  const btn = document.getElementById("send-button");
  const input = document.getElementById("message-input");
  if (btn) btn.disabled = !!disabled;
  if (input) input.disabled = !!disabled;
}

function truncate(s, maxLen) {
  if (!s) return "";
  if (s.length <= maxLen) return s;
  return s.slice(0, maxLen - 1) + "…";
}

function loopSuffix(ev) {
  const n = ev.retrieval_loop_count ?? ev.retrieval_retry_count;
  return n != null ? " · loop " + n : "";
}

function chatScrollBottom() {
  const scroll = document.querySelector(".rag-chat-scroll");
  if (scroll) scroll.scrollTop = scroll.scrollHeight;
}

function progressScrollBottom(progressEl) {
  if (progressEl) progressEl.scrollTop = progressEl.scrollHeight;
}

function ensureThinkingPanelHost() {
  let panel = document.getElementById("rag-thinking-panel");
  if (panel) return panel;
  const composer = document.querySelector(".rag-composer-outer");
  if (!composer || !composer.parentElement) return null;
  panel = document.createElement("div");
  panel.id = "rag-thinking-panel";
  panel.className = "rag-thinking-panel rag-thinking-panel--hidden";
  composer.parentElement.insertBefore(panel, composer);
  return panel;
}

function buildThinkingCardDOM() {
  const card = document.createElement("div");
  card.className = "rag-thinking-card";

  const bar = document.createElement("button");
  bar.type = "button";
  bar.className = "rag-thinking-bar";
  bar.setAttribute("aria-expanded", "true");

  const spinner = document.createElement("span");
  spinner.className = "rag-thinking-spinner";
  spinner.setAttribute("aria-hidden", "true");

  const label = document.createElement("span");
  label.className = "rag-thinking-label";
  label.textContent = "Thinking";

  const summary = document.createElement("span");
  summary.className = "rag-thinking-summary";
  summary.textContent = "Connecting…";

  const chevron = document.createElement("span");
  chevron.className = "rag-thinking-chevron";
  chevron.textContent = "▾";
  chevron.setAttribute("aria-hidden", "true");

  bar.appendChild(spinner);
  bar.appendChild(label);
  bar.appendChild(summary);
  bar.appendChild(chevron);

  bar.addEventListener("click", function () {
    const collapsed = card.classList.toggle("rag-thinking-card--collapsed");
    bar.setAttribute("aria-expanded", collapsed ? "false" : "true");
  });

  const body = document.createElement("div");
  body.className = "rag-thinking-body";

  const progressEl = document.createElement("div");
  progressEl.className = "rag-inline-progress";

  body.appendChild(progressEl);
  card.appendChild(bar);
  card.appendChild(body);

  return { card: card, progressEl: progressEl };
}

function rebuildThinkingCard() {
  const panel = ensureThinkingPanelHost();
  if (!panel) return null;
  panel.className = "rag-thinking-panel";
  panel.innerHTML = "";
  const built = buildThinkingCardDOM();
  panel.appendChild(built.card);
  return built.progressEl;
}

function resolveProgressEl(fallback) {
  const el = document.querySelector("#rag-thinking-panel .rag-inline-progress");
  if (el && el.isConnected) return el;
  if (fallback && fallback.isConnected) return fallback;
  if (window.dash_clientside.rag_ui._inFlight) return rebuildThinkingCard();
  return fallback || null;
}

function updateThinkingSummary(boldText, restText) {
  const summary = document.querySelector(".rag-thinking-summary");
  if (!summary) return;
  const text = (boldText || "") + (restText || "");
  summary.textContent = truncate(text.trim(), 120) || "Working…";
}

function commitChatState(chat) {
  if (window.dash_clientside && window.dash_clientside.set_props) {
    window.dash_clientside.set_props("chat-state", { data: chat });
  }
}

function openThinkingPanel() {
  const panel = ensureThinkingPanelHost();
  if (!panel) return null;
  clearStreamUI();
  panel.className = "rag-thinking-panel";
  panel.innerHTML = "";
  const built = buildThinkingCardDOM();
  panel.appendChild(built.card);
  return built.progressEl;
}

function clearStreamUI() {
  const panel = document.getElementById("rag-thinking-panel");
  if (panel) {
    panel.innerHTML = "";
    panel.className = "rag-thinking-panel rag-thinking-panel--hidden";
  }
}

/** Bold segment is the LangGraph node id (``ev.node``); rest is detail text. */
function progressBoldRest(ev) {
  if (!ev || typeof ev !== "object") return { bold: "event", rest: " — " + String(ev) };
  const node = ev.node || "";

  if (ev.type === "error") {
    return { bold: "error", rest: ev.error ? ": " + ev.error : "" };
  }
  if (ev.type === "final") {
    const bold = node || "final";
    const preview = truncate(ev.answer || "", 160);
    return { bold: bold, rest: preview ? ": " + preview : "" };
  }

  const boldName = node || "node";

  if (node === "query_normalisation" && ev.normalized_query) {
    return { bold: boldName, rest: " — " + truncate(ev.normalized_query, 200) };
  }
  if (node === "query_complexity") {
    const split =
      ev.needs_split === true || (ev.fact_count != null && ev.fact_count > 1)
        ? "needs_split"
        : "simple_query";
    const ex = ev.explanation ? " — " + truncate(ev.explanation, 120) : "";
    return { bold: boldName, rest: ": " + split + ex };
  }
  if (node === "fact_decomposition") {
    const n = ev.fact_count || 0;
    return { bold: boldName, rest: n ? " (" + n + " facts)" : "" };
  }
  if (node === "retrieval") {
    const strat = ev.retrieval_strategy ? " · " + ev.retrieval_strategy : "";
    const n = ev.retrieved_doc_count != null ? " (" + ev.retrieved_doc_count + " docs)" : "";
    const add = ev.new_doc_count > 0 ? " +" + ev.new_doc_count : "";
    return { bold: boldName, rest: strat + n + add + loopSuffix(ev) };
  }
  if (node === "recall_check") {
    const ok = ev.recall_sufficient ? "sufficient" : "insufficient";
    const unsupported = ev.unsupported_fact_count || 0;
    const docs = ev.retrieved_doc_count != null ? " · " + ev.retrieved_doc_count + " docs" : "";
    return { bold: boldName, rest: ": " + ok + docs + (unsupported ? " · " + unsupported + " unsupported" : "") + loopSuffix(ev) };
  }
  if (node === "strategy_upgrade") {
    const rs = ev.retrieval_strategy ? " · " + ev.retrieval_strategy : "";
    return { bold: boldName, rest: " · deterministic" + rs + loopSuffix(ev) };
  }
  if (node === "query_splitter") {
    const pq = (ev.active_retrieval_queries || []).join(" · ");
    return { bold: boldName, rest: pq ? " — " + truncate(pq, 180) : "" };
  }
  if (node === "create_queries_for_unsupported_facts") {
    const q = (ev.active_retrieval_queries || []).join(" · ");
    return { bold: boldName, rest: q ? " — " + truncate(q, 160) : "" };
  }
  if (node === "answer" || node === "partial_answer") {
    const n = (ev.cited_document_ids || []).length;
    const conf = ev.confidence ? " · " + ev.confidence : "";
    return {
      bold: boldName,
      rest: n ? " (" + n + " cited ids)" + conf : conf || (ev.label ? " — " + truncate(ev.label, 120) : ""),
    };
  }
  if (node === "faithfulness") {
    const status =
      ev.faithfulness_ok === true ? "ok" : ev.faithfulness_ok === false ? "retry" : "";
    const retries =
      ev.faithfulness_retry_count != null ? " · attempt " + ev.faithfulness_retry_count : "";
    return { bold: boldName, rest: (status ? ": " + status : "") + retries };
  }
  if (node === "error_answer") {
    return { bold: boldName, rest: ev.label ? " — " + truncate(ev.label, 120) : "" };
  }

  const label = ev.label || "";
  return { bold: boldName, rest: label ? " — " + truncate(label, 120) : "" };
}

function appendProgressBold(progressEl, boldText, restText) {
  progressEl = resolveProgressEl(progressEl);
  if (!progressEl) return;
  const row = document.createElement("div");
  row.className = "rag-progress-line";
  if (boldText) {
    const s = document.createElement("strong");
    s.textContent = boldText;
    row.appendChild(s);
  }
  if (restText) row.appendChild(document.createTextNode(restText));
  progressEl.appendChild(row);
  progressScrollBottom(progressEl);
  updateThinkingSummary(boldText, restText);
}

async function parseSSEStream(response, progressEl) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalPayload = null;
  let retrievedDocCount = 0;
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const lines = frame.split("\n");
      for (let i = 0; i < lines.length; i++) {
        let line = lines[i];
        if (line.endsWith("\r")) line = line.slice(0, -1);
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6);
        let payload;
        try {
          payload = JSON.parse(raw);
        } catch (e) {
          appendProgressBold(progressEl, "sse", " — could not parse SSE frame");
          continue;
        }
        if (payload.type === "final") {
          const answerText =
            payload.answer != null ? String(payload.answer).trim() : "";
          if (!finalPayload || answerText) {
            finalPayload = payload;
          }
        }
        if (payload.type === "done" && payload.retrieved_doc_count != null)
          retrievedDocCount = payload.retrieved_doc_count;
        if (payload.type === "error") {
          const pr = progressBoldRest(payload);
          appendProgressBold(progressEl, pr.bold, pr.rest);
          throw new Error(payload.error || "Stream error");
        }
        if (payload.type === "done") {
          appendProgressBold(progressEl, "done", " — stream finished");
          continue;
        }
        const pr = progressBoldRest(payload);
        appendProgressBold(progressEl, pr.bold, pr.rest);
      }
    }
  }
  return { finalPayload: finalPayload, retrievedDocCount: retrievedDocCount };
}

function clockNowMs() {
  return typeof performance !== "undefined" && performance.now
    ? performance.now()
    : Date.now();
}

function formatElapsedLabel(ms) {
  if (ms == null || typeof ms !== "number" || ms < 0 || isNaN(ms)) return "";
  if (ms < 1000) return Math.round(ms) + " ms";
  var sec = ms / 1000;
  var decimals = sec < 10 ? 1 : 0;
  return sec.toFixed(decimals) + " s";
}

function appendTimeTakenFooter(content, elapsedMs) {
  var label = formatElapsedLabel(elapsedMs);
  if (!label) return content;
  return content + "\n\n*Time taken: " + label + "*";
}

window.dash_clientside.rag_ui.clear_progress = function (_session_gen) {
  clearStreamUI();
  window.dash_clientside.rag_ui._inFlight = false;
  setComposerDisabled(false);
  return "";
};

window.dash_clientside.rag_ui.submit_message = async function (
  n_clicks,
  n_submit,
  message,
  session_id,
  chat_state,
  pending_interrupt,
  api_base
) {
  const nu = window.dash_clientside.no_update;
  const raw = (message || "").trim();
  if (!raw) {
    return [nu, nu, nu, ""];
  }

  if (window.dash_clientside.rag_ui._inFlight) {
    return [nu, nu, nu, "Please wait for the current response."];
  }

  window.dash_clientside.rag_ui._inFlight = true;
  setComposerDisabled(true);

  let chat = Array.isArray(chat_state) ? chat_state.slice() : [];
  chat.push({ role: "user", content: raw });
  commitChatState(chat);

  const base = (api_base || "http://127.0.0.1:8000").replace(/\/$/, "");
  const progressEl = pending_interrupt ? null : openThinkingPanel();
  chatScrollBottom();

  try {
    if (pending_interrupt) {
      var startResume = clockNowMs();
      const r = await fetch(base + "/resume", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ session_id: session_id, answer: raw }),
      });
      let data;
      try {
        data = await r.json();
      } catch (e) {
        throw new Error("Resume response was not JSON");
      }
      if (!r.ok) {
        const detail = data.detail !== undefined ? data.detail : data;
        const msg =
          typeof detail === "string" ? detail : JSON.stringify(detail);
        throw new Error(msg);
      }
      if (data.interrupted) {
        chat.push({
          role: "assistant",
          content: data.question || "Please clarify.",
        });
        return [chat, true, "", ""];
      }
      var elapsedResume = clockNowMs() - startResume;
      let content = data.answer || "Done.";
      const sources = data.sources || [];
      if (sources.length)
        content += "\n\nSources: " + sources.map(String).join(", ");
      const catalog = data.document_catalog || {};
      const catalogCount = Object.keys(catalog).length;
      if (catalogCount) content += "\n\nRetrieved " + catalogCount + " passages.";
      content = appendTimeTakenFooter(content, elapsedResume);
      chat.push({ role: "assistant", content: content });
      return [chat, false, "", ""];
    }

    var startStream = clockNowMs();
    appendProgressBold(progressEl, "…", " — connecting");
    const r = await fetch(base + "/run/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ session_id: session_id, message: raw }),
    });
    if (!r.ok) {
      let errText = await r.text();
      try {
        const ej = JSON.parse(errText);
        if (ej.detail !== undefined)
          errText =
            typeof ej.detail === "string" ? ej.detail : JSON.stringify(ej.detail);
      } catch (e) {}
      throw new Error(errText || r.statusText);
    }

    const { finalPayload, retrievedDocCount } = await parseSSEStream(r, progressEl);
    var elapsedStream = clockNowMs() - startStream;

    if (!finalPayload || !(finalPayload.answer && String(finalPayload.answer).trim())) {
      chat.push({
        role: "assistant",
        content: appendTimeTakenFooter(
          "Run finished without a final answer event. Check API logs or graph configuration.",
          elapsedStream
        ),
      });
      clearStreamUI();
      return [chat, false, "", ""];
    }

    let content = finalPayload.answer || "Done.";
    const sources = finalPayload.sources || [];
    if (sources.length)
      content += "\n\nSources: " + sources.map(String).join(", ");
    const finalDocCount =
      retrievedDocCount || finalPayload.retrieved_doc_count || 0;
    if (finalDocCount) content += "\n\nRetrieved " + finalDocCount + " passages.";
    content = appendTimeTakenFooter(content, elapsedStream);
    chat.push({ role: "assistant", content: content });
    clearStreamUI();
    return [chat, false, "", ""];
  } catch (e) {
    const msg = e && e.message ? e.message : String(e);
    if (progressEl) appendProgressBold(progressEl, "error", " — " + truncate(msg, 400));
    if (!pending_interrupt && chat.length) {
      chat.push({
        role: "assistant",
        content: "Error: " + truncate(msg, 2000),
      });
    }
    clearStreamUI();
    return [chat, !!pending_interrupt, "", msg];
  } finally {
    window.dash_clientside.rag_ui._inFlight = false;
    setComposerDisabled(false);
  }
};
