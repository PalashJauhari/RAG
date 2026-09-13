"""Dash chat UI for the RAG orchestrator.

Layout: left sidebar (session), center chat + inline streaming panel + composer.
Streaming uses clientside JS (``assets/rag_ui.js``) calling ``POST /run/stream`` on the
FastAPI service (``API_URL`` / ``dcc.Store`` ``api-base-url``). ``session-id`` maps to
graph ``thread_id``; ``pending-interrupt`` supports future ``/resume`` clarification flows.
"""

import os
import sys
from pathlib import Path
from uuid import uuid4

from dash import ClientsideFunction, Dash, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder=str(Path(__file__).resolve().parent / "assets"),
)
app.title = "Citeflow"


# --- Layout (stores + chat shell) ---
# Lambda layout so session-id is fresh per page load; stores hold API URL and chat state.

app.layout = lambda: html.Div(
    className="rag-shell",
    children=[
        dcc.Store(id="session-id", data=str(uuid4())),
        dcc.Store(
            id="api-base-url",
            data=os.getenv("API_URL", "http://127.0.0.1:8000"),
        ),
        dcc.Store(id="chat-state", data=[]),
        dcc.Store(id="pending-interrupt", data=False),
        dcc.Store(id="session-gen", data=0),
        html.Div(id="clientside-dummy-out", style={"display": "none"}),
        html.Div(
            className="rag-frame",
            children=[
                html.Div(
                    className="rag-sidebar",
                    children=[
                        html.Div(
                            className="rag-brand-row",
                            children=[
                                html.Div("G", className="rag-avatar rag-avatar-lg"),
                                html.Div(
                                    children=[
                                        html.Span("Citeflow", className="rag-brand-title"),
                                        html.Span(" RAG Chat", className="rag-brand-sub"),
                                    ]
                                ),
                            ],
                        ),
                        html.P(
                            "Retrieval-augmented answers from your knowledge base.",
                            className="rag-tagline",
                        ),
                        html.P("Session", className="rag-section-label"),
                        html.Div(id="session-line", className="rag-session-id"),
                        html.Button(
                            "New chat",
                            id="btn-new-session",
                            type="button",
                            n_clicks=0,
                            className="rag-btn-new-chat",
                        ),
                    ],
                ),
                html.Div(
                    className="rag-main",
                    children=[
                        html.Div(
                            className="rag-banner-slot",
                            children=[html.Div(id="interrupt-banner")],
                        ),
                        html.Div(
                            className="rag-chat-scroll",
                            children=[html.Div(className="rag-chat-inner", id="chat-area")],
                        ),
                        html.Div(
                            id="rag-thinking-panel",
                            className="rag-thinking-panel rag-thinking-panel--hidden",
                        ),
                        html.Div(
                            className="rag-composer-outer",
                            children=[
                                html.Div(
                                    className="rag-composer-inner",
                                    children=[
                                        dcc.Input(
                                            id="message-input",
                                            type="text",
                                            placeholder="Message RAG…",
                                            debounce=False,
                                        ),
                                        html.Button(
                                            "Send",
                                            id="send-button",
                                            type="button",
                                            n_clicks=0,
                                        ),
                                    ],
                                ),
                                html.Div(id="error", className="rag-error"),
                                html.P(
                                    "Answers use your connected knowledge sources.",
                                    className="rag-footer-hint",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# --- Server-side callbacks ---


@callback(Output("session-line", "children"), Input("session-id", "data"))
def render_session_line(session_id: str | None):
    """Show a truncated session id in the sidebar."""
    sid = session_id or ""
    return sid[:20] + "…" if len(sid) > 20 else sid


@callback(Output("interrupt-banner", "children"), Input("pending-interrupt", "data"))
def render_interrupt_banner(pending):
    """Show banner when the API reports an interrupt awaiting ``/resume``."""
    if pending:
        return html.Div(
            className="rag-banner",
            children=[
                html.Strong("Clarification needed — "),
                html.Span("reply below, then press send."),
            ],
        )
    return html.Div()


def _meta_row(*, is_user: bool) -> html.Div:
    """Small timestamp/read-receipt footer matching the forecasting UI."""
    checks = (
        html.Span("✓✓", className="rag-meta-checks")
        if is_user
        else None
    )
    return html.Div(
        className="rag-meta-row",
        children=[html.Span("Just now"), checks] if checks else [html.Span("Just now")],
    )


def _cited_media_children(citations: list | None) -> list:
    """Render cited source URL, page, tables, and images when present."""

    if not citations:
        return []
    blocks: list = []
    for row in citations:
        if not isinstance(row, dict):
            continue
        kids: list = []
        source = str(row.get("source") or "").strip()
        page = row.get("page_number")
        meta: list = []
        if source:
            if source.startswith("http://") or source.startswith("https://"):
                meta.append(
                    html.A(
                        source,
                        href=source,
                        target="_blank",
                        rel="noopener noreferrer",
                    )
                )
            else:
                meta.append(html.Span(source))
        if page is not None and str(page).strip() != "":
            if meta:
                meta.append(html.Span(" · "))
            meta.append(html.Span(f"p. {page}"))
        if meta:
            kids.append(html.Div(meta, className="rag-cited-meta"))
        for table in row.get("table_html") or []:
            html_src = str(table or "").strip()
            if html_src:
                kids.append(
                    dcc.Markdown(
                        html_src,
                        dangerously_allow_html=True,
                        className="rag-cited-table",
                    )
                )
        for image in row.get("images_base64") or []:
            data = str(image or "").strip()
            if data:
                kids.append(
                    html.Img(
                        src=f"data:image/png;base64,{data}",
                        className="rag-cited-image",
                    )
                )
        if kids:
            blocks.append(html.Div(kids, className="rag-cited-item"))
    if not blocks:
        return []
    return [html.Div([html.Div("Sources", className="rag-cited-heading"), *blocks], className="rag-cited")]


@callback(Output("chat-area", "children"), Input("chat-state", "data"))
def render_chat(messages):
    """Render ``chat-state`` as user/assistant bubbles (Markdown for assistant)."""
    msgs = messages or []
    if not msgs:
        return html.Div(
            className="rag-empty-state",
            children=[
                html.H2("How can I help you today?"),
                html.P("Send a message below to get started."),
            ],
        )

    blocks: list = []
    for m in msgs:
        role = m.get("role", "assistant")
        content = str(m.get("content") or "")
        if role == "user":
            blocks.append(
                html.Div(
                    className="rag-msg-row-user",
                    children=[
                        html.Div(
                            className="rag-bubble-user",
                            children=[
                                html.Div(content, style={"whiteSpace": "pre-wrap"}),
                                _meta_row(is_user=True),
                            ],
                        )
                    ],
                )
            )
        else:
            blocks.append(
                html.Div(
                    className="rag-msg-row-assistant",
                    children=[
                        html.Div("G", className="rag-avatar"),
                        html.Div(
                            className="rag-bubble-assistant",
                            children=[
                                dcc.Markdown(
                                    content,
                                    dangerously_allow_html=False,
                                    className="rag-md",
                                ),
                                *_cited_media_children(m.get("citations")),
                                _meta_row(is_user=False),
                            ],
                        ),
                    ],
                )
            )

    return html.Div(children=blocks)


@callback(
    Output("session-id", "data"),
    Output("chat-state", "data"),
    Output("pending-interrupt", "data"),
    Output("session-gen", "data"),
    Output("message-input", "value"),
    Output("error", "children"),
    Input("btn-new-session", "n_clicks"),
    State("session-gen", "data"),
    prevent_initial_call=True,
)
def on_new_session(n_clicks, gen):
    """Reset session id, chat history, interrupt flag, and bump ``session-gen`` for clientside cleanup."""
    if not n_clicks:
        raise PreventUpdate
    next_gen = int(gen or 0) + 1
    return str(uuid4()), [], False, next_gen, "", ""


# --- Clientside callbacks (SSE stream + inline progress panel in rag_ui.js) ---

app.clientside_callback(
    ClientsideFunction(namespace="rag_ui", function_name="clear_progress"),
    Output("clientside-dummy-out", "children"),
    Input("session-gen", "data"),
    prevent_initial_call=True,
)


app.clientside_callback(
    ClientsideFunction(namespace="rag_ui", function_name="submit_message"),
    Output("chat-state", "data", allow_duplicate=True),
    Output("pending-interrupt", "data", allow_duplicate=True),
    Output("message-input", "value", allow_duplicate=True),
    Output("error", "children", allow_duplicate=True),
    Input("send-button", "n_clicks"),
    Input("message-input", "n_submit"),
    State("message-input", "value"),
    State("session-id", "data"),
    State("chat-state", "data"),
    State("pending-interrupt", "data"),
    State("api-base-url", "data"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    _rs = str(_PROJECT_ROOT)
    if _rs not in sys.path:
        sys.path.insert(0, _rs)
    app.run(debug=True, host="0.0.0.0", port=8050)
