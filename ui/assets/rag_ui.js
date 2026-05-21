/**
 * Dash clientside SSE consumer for POST /run/stream.
 * Event shape matches api.main.get_stream_event (node, type, label, node-specific fields).
 * Progress column: bold = LangGraph node id; remainder = human-readable detail.
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
    const c = ev.complexity || "";
    const rs = ev.retrieval_strategy ? " · " + ev.retrieval_strategy : "";
    const ex = ev.explanation ? " — " + truncate(ev.explanation, 120) : "";
    return { bold: boldName, rest: (c ? ": " + c : "") + rs + ex };
  }
  if (node === "retrieval") {
    const strat = ev.retrieval_strategy ? " · " + ev.retrieval_strategy : "";
    const n = ev.retrieved_doc_count != null ? " (" + ev.retrieved_doc_count + " docs)" : "";
    const q = (ev.retrieval_queries || []).join(" · ");
    const newDocs = ev.new_retrieved_documents || [];
    const preview =
      newDocs.length > 0 && newDocs[0].text
        ? " — " + truncate(newDocs[0].text, 120)
        : "";
    return { bold: boldName, rest: strat + n + (q ? " — " + truncate(q, 160) : "") + preview };
  }
  if (node === "recall_check") {
    const ok = ev.recall_sufficient ? "sufficient" : "insufficient";
    const mf = (ev.missing_facts || []).length;
    const rc = ev.retrieval_retry_count != null ? " · retry " + ev.retrieval_retry_count : "";
    return { bold: boldName, rest: ": " + ok + (mf ? " · " + mf + " missing" : "") + rc };
  }
  if (node === "intent_check") {
    const ia = ev.intent_aligned ? "aligned" : "misaligned";
    return { bold: boldName, rest: ": " + ia };
  }
  if (node === "fact_gap_retrieval") {
    const n = ev.fact_gap_doc_count != null ? ev.fact_gap_doc_count + " docs" : "";
    return { bold: boldName, rest: n ? " — " + n : "" };
  }
  if (node === "strategy_upgrade") {
    const up = ev.apply_strategy_upgrade === true ? "upgrade" : ev.apply_strategy_upgrade === false ? "keep tier" : "";
    const rs = ev.retrieval_strategy ? " · " + ev.retrieval_strategy : "";
    const rc = ev.retrieval_retry_count != null ? " · retry " + ev.retrieval_retry_count : "";
    return { bold: boldName, rest: (up ? up : "") + rs + rc };
  }
  if (["query_splitter", "query_expansion", "query_rewriter"].indexOf(node) !== -1) {
    const pq = (ev.active_retrieval_queries || []).join(" · ");
    return { bold: boldName, rest: pq ? " — " + truncate(pq, 180) : "" };
  }
  if (node === "gap_fill") {
    const q = (ev.active_retrieval_queries || []).join(" · ");
    const rs = ev.retrieval_strategy ? " · " + ev.retrieval_strategy : "";
    return { bold: boldName, rest: rs + (q ? " — " + truncate(q, 160) : "") };
  }
  if (node === "intent_correction_rewriter") {
    const q = (ev.active_retrieval_queries || []).join(" · ");
    const rs = ev.retrieval_strategy ? " · " + ev.retrieval_strategy : "";
    return { bold: boldName, rest: rs + (q ? " — " + truncate(q, 160) : "") };
  }
  if (node === "clear_turn_trace") {
    return { bold: boldName, rest: " — turn trace cleared" };
  }

  const label = ev.label || "";
  return { bold: boldName, rest: label ? " — " + truncate(label, 120) : "" };
}

function appendProgressBold(progressEl, boldText, restText) {
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
  progressEl.scrollTop = progressEl.scrollHeight;
}

async function parseSSEStream(response, progressEl) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalPayload = null;
  let retrievedDocs = [];
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
        if (payload.type === "final") finalPayload = payload;
        if (payload.type === "done" && payload.retrieved_docs)
          retrievedDocs = payload.retrieved_docs;
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
  return { finalPayload: finalPayload, retrievedDocs: retrievedDocs };
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
  const el = document.getElementById("rag-stream-progress");
  if (el) el.innerHTML = "";
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

  const base = (api_base || "http://127.0.0.1:8000").replace(/\/$/, "");
  const progressEl = document.getElementById("rag-stream-progress");
  if (progressEl) progressEl.innerHTML = "";

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
      const rd = data.retrieved_docs || [];
      if (rd.length) content += "\n\nRetrieved " + rd.length + " passages.";
      content = appendTimeTakenFooter(content, elapsedResume);
      chat.push({ role: "assistant", content: content });
      return [chat, false, "", ""];
    }

    var startStream = clockNowMs();
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

    const { finalPayload, retrievedDocs } = await parseSSEStream(r, progressEl);
    var elapsedStream = clockNowMs() - startStream;

    if (!finalPayload) {
      chat.push({
        role: "assistant",
        content: appendTimeTakenFooter(
          "Run finished without a final answer event. Check API logs or graph configuration.",
          elapsedStream
        ),
      });
      return [chat, false, "", ""];
    }

    let content = finalPayload.answer || "Done.";
    const sources = finalPayload.sources || [];
    if (sources.length)
      content += "\n\nSources: " + sources.map(String).join(", ");
    if (retrievedDocs.length)
      content += "\n\nRetrieved " + retrievedDocs.length + " passages.";
    content = appendTimeTakenFooter(content, elapsedStream);
    chat.push({ role: "assistant", content: content });
    return [chat, false, "", ""];
  } catch (e) {
    const msg = e && e.message ? e.message : String(e);
    return [chat, !!pending_interrupt, "", msg];
  } finally {
    window.dash_clientside.rag_ui._inFlight = false;
    setComposerDisabled(false);
  }
};
