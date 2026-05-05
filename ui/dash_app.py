import json
import os
from uuid import uuid4

import requests
from dash import Dash, Input, Output, State, dcc, html
from dotenv import load_dotenv


load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

app = Dash(__name__)
app.title = "RAG Retrieval Chat"

app.layout = html.Main(
    [
        dcc.Store(id="session-id", data=str(uuid4())),
        dcc.Store(id="chat-state", data=[]),
        dcc.Store(id="pending-interrupt", data=False),
        html.Header(
            [
                html.H1("RAG Retrieval Chat"),
                html.P("Ask a question. The graph will retrieve context and answer from it."),
            ],
            style={"padding": "24px 28px 12px"},
        ),
        html.Section(
            id="chat-log",
            style={
                "height": "58vh",
                "overflowY": "auto",
                "padding": "16px 28px",
                "borderTop": "1px solid #e5e7eb",
                "borderBottom": "1px solid #e5e7eb",
            },
        ),
        html.Div(
            [
                dcc.Input(
                    id="message-input",
                    type="text",
                    placeholder="Type your message",
                    autoComplete="off",
                    style={
                        "flex": "1",
                        "height": "42px",
                        "padding": "0 12px",
                        "border": "1px solid #cbd5e1",
                        "borderRadius": "6px",
                    },
                ),
                html.Button(
                    "Send",
                    id="send-button",
                    style={
                        "height": "42px",
                        "padding": "0 18px",
                        "border": "0",
                        "borderRadius": "6px",
                        "background": "#111827",
                        "color": "white",
                        "cursor": "pointer",
                    },
                ),
            ],
            style={
                "display": "flex",
                "gap": "10px",
                "padding": "18px 28px 8px",
            },
        ),
        html.Div(id="error", style={"color": "#b91c1c", "padding": "0 28px 18px"}),
    ],
    style={
        "fontFamily": "Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
        "maxWidth": "920px",
        "margin": "0 auto",
        "color": "#111827",
    },
)


@app.callback(Output("chat-log", "children"), Input("chat-state", "data"))
def render_chat(messages):
    return [
        html.Div(
            [
                html.Strong("You" if message["role"] == "user" else "Assistant"),
                html.Div(message["content"], style={"whiteSpace": "pre-wrap", "marginTop": "4px"}),
            ],
            style={
                "padding": "10px 0",
                "borderBottom": "1px solid #f1f5f9",
            },
        )
        for message in messages
    ]


@app.callback(
    Output("chat-state", "data"),
    Output("pending-interrupt", "data"),
    Output("message-input", "value"),
    Output("error", "children"),
    Input("send-button", "n_clicks"),
    State("message-input", "value"),
    State("session-id", "data"),
    State("chat-state", "data"),
    State("pending-interrupt", "data"),
    prevent_initial_call=True,
)
def submit_message(_, message, session_id, chat_state, pending_interrupt):
    message = (message or "").strip()
    if not message:
        return chat_state, pending_interrupt, "", ""

    endpoint = "/resume" if pending_interrupt else "/run"
    payload = (
        {"session_id": session_id, "answer": message}
        if pending_interrupt
        else {"session_id": session_id, "message": message}
    )
    chat_state = [*(chat_state or []), {"role": "user", "content": message}]

    try:
        api_response = requests.post(
            f"{API_URL}{endpoint}",
            json=payload,
            timeout=120,
        )
        api_response.raise_for_status()
        data = api_response.json()
    except requests.RequestException as exc:
        return chat_state, pending_interrupt, "", str(exc)

    if data["status"] == "interrupted":
        chat_state.append({"role": "assistant", "content": data["question"]})
        return chat_state, True, "", ""

    response = data["response"]
    content = (
        response.get("answer")
        if isinstance(response, dict) and response.get("answer")
        else json.dumps(response, indent=2)
    )
    chat_state.append({"role": "assistant", "content": content})
    return chat_state, False, "", ""


if __name__ == "__main__":
    app.run(debug=True, port=8050)
