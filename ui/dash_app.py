from uuid import uuid4

from dash import Dash, Input, Output, State, dcc, html
from dotenv import load_dotenv

from ui.api_client import RagApiClient

load_dotenv()
api = RagApiClient()

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
    Input("message-input", "n_submit"),
    State("message-input", "value"),
    State("session-id", "data"),
    State("chat-state", "data"),
    State("pending-interrupt", "data"),
    prevent_initial_call=True,
)
def submit_message(_clicks, _submits, message, session_id, chat_state, pending_interrupt):
    message = (message or "").strip()
    if not message:
        return chat_state, pending_interrupt, "", ""

    chat_state = [*(chat_state or []), {"role": "user", "content": message}]

    data = api.resume(message, session_id) if pending_interrupt else api.run(message, session_id)
    if data.get("error"):
        return chat_state, pending_interrupt, "", data["error"]

    if data.get("interrupted"):
        chat_state.append({"role": "assistant", "content": data.get("question") or "Please clarify."})
        return chat_state, True, "", ""

    content = data.get("answer") or "Done."
    sources = data.get("sources") or []
    if sources:
        content = f"{content}\n\nSources: {', '.join(str(source) for source in sources)}"
    chat_state.append({"role": "assistant", "content": content})
    return chat_state, False, "", ""


if __name__ == "__main__":
    app.run(debug=True, port=8050)
