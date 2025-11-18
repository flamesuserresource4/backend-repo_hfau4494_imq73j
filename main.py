import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from database import create_document, get_documents, db

app = FastAPI(title="Flames Multi-Provider LLM Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Utility
# -----------------------------

def serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(doc)
    _id = d.pop("_id", None)
    if _id is not None:
        d["id"] = str(_id)
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


# -----------------------------
# Root & Health
# -----------------------------

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response: Dict[str, Any] = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = getattr(db, "name", "✅ Connected")
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:  # noqa: BLE001
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:  # noqa: BLE001
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# -----------------------------
# Providers & Models (Mock discovery for now)
# -----------------------------

class ProviderConfig(BaseModel):
    provider: Literal[
        "openrouter",
        "openai",
        "gemini",
        "claude",
        "grok",
        "minimax",
        "custom",
    ]
    base_url: str
    api_key: Optional[str] = None
    organization: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    display_name: str
    tools: bool = True
    images: bool = True
    reasoning: bool = False
    streaming: bool = True


MOCK_MODELS: Dict[str, List[ModelInfo]] = {
    "openrouter": [
        ModelInfo(id="openrouter/anthropic/claude-3.5-sonnet", display_name="Claude 3.5 Sonnet"),
        ModelInfo(id="openrouter/openai/gpt-4o", display_name="GPT-4o"),
        ModelInfo(id="openrouter/google/gemini-1.5-pro", display_name="Gemini 1.5 Pro"),
    ],
    "openai": [
        ModelInfo(id="gpt-4o", display_name="GPT-4o"),
        ModelInfo(id="gpt-4.1", display_name="GPT-4.1"),
        ModelInfo(id="gpt-4o-mini", display_name="GPT-4o mini"),
    ],
    "gemini": [
        ModelInfo(id="gemini-1.5-pro", display_name="Gemini 1.5 Pro"),
        ModelInfo(id="gemini-1.5-flash", display_name="Gemini 1.5 Flash"),
    ],
    "claude": [
        ModelInfo(id="claude-3-5-sonnet", display_name="Claude 3.5 Sonnet"),
        ModelInfo(id="claude-3-opus", display_name="Claude 3 Opus", reasoning=True),
    ],
    "grok": [ModelInfo(id="grok-2", display_name="Grok 2")],
    "minimax": [ModelInfo(id="abab6.5s", display_name="MiniMax abab6.5s")],
    "custom": [ModelInfo(id="custom-compatible-model", display_name="Custom Compatible Model")],
}


@app.post("/api/providers/models")
def list_models(cfg: ProviderConfig) -> Dict[str, Any]:
    # In a later version, call provider list API. For now, return a mock set based on provider.
    models = MOCK_MODELS.get(cfg.provider, MOCK_MODELS["custom"])
    return {"provider": cfg.provider, "models": [m.model_dump() for m in models]}


class ProviderTestRequest(ProviderConfig):
    protocol: Literal["HTTP", "HTTPS", "SSE", "WebSocket"] = "HTTP"


@app.post("/api/providers/test-connection")
def test_connection(req: ProviderTestRequest) -> Dict[str, Any]:
    ok = bool(req.base_url)
    return {
        "ok": ok,
        "provider": req.provider,
        "protocol": req.protocol,
        "message": "Connection parameters look valid" if ok else "Missing base URL",
    }


# -----------------------------
# Private Server for Development
# -----------------------------

class PrivateServerConfig(BaseModel):
    enabled: bool = False
    name: Optional[str] = None
    host: Optional[str] = None
    port: Optional[str] = None
    username: Optional[str] = None
    token: Optional[str] = None
    protocol: Literal["HTTP", "HTTPS", "SSE", "WebSocket"] = "HTTP"
    base_url: Optional[str] = None

    def build_base_url(self) -> Optional[str]:
        if self.base_url:
            return self.base_url
        if not self.host:
            return None
        scheme = "https" if self.protocol == "HTTPS" else "http"
        if self.port:
            return f"{scheme}://{self.host}:{self.port}"
        return f"{scheme}://{self.host}"


class PrivateServerTestRequest(PrivateServerConfig):
    path: str = "/"
    timeout_seconds: int = 5


@app.post("/api/private-server/test-connection")
def private_server_test(req: PrivateServerTestRequest) -> Dict[str, Any]:
    import json
    import urllib.request
    import urllib.error

    base = req.build_base_url()
    if not base:
        return {"ok": False, "message": "Missing host or base URL"}

    url = base.rstrip("/") + (req.path if req.path.startswith("/") else f"/{req.path}")
    headers = {"User-Agent": "FlamesPrivateClient/1.0"}
    if req.token:
        headers["Authorization"] = f"Bearer {req.token}"
    try:
        request = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=req.timeout_seconds) as resp:  # nosec B310
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            ok = 200 <= status < 400
            # Try to parse a small JSON if available (best-effort)
            body = None
            try:
                if content_type.startswith("application/json"):
                    body = json.loads(resp.read().decode("utf-8"))
            except Exception:  # noqa: BLE001
                body = None
            return {
                "ok": ok,
                "status": status,
                "content_type": content_type,
                "url": url,
                "body": body,
                "protocol": req.protocol,
            }
    except urllib.error.HTTPError as e:  # noqa: BLE001
        return {"ok": False, "status": e.code, "message": str(e), "url": url}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": str(e), "url": url}


class ExecRequest(BaseModel):
    connection: PrivateServerConfig
    command: str = "echo Hello"
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None


@app.post("/api/private-server/exec/stream")
def private_server_exec_stream(req: ExecRequest):
    # Mock streaming logs; later this can proxy to the private server over SSE/WebSocket
    def generator():
        import time
        import json
        start = datetime.now(timezone.utc).isoformat()
        header = {"type": "start", "command": req.command, "cwd": req.cwd, "start": start}
        yield f"data: {json.dumps(header)}\n\n"
        lines = [
            "Preparing environment...",
            "Connecting to private server...",
            f"Protocol: {req.connection.protocol}",
            f"Executing: {req.command}",
            "Running...",
        ]
        for i, line in enumerate(lines, 1):
            payload = {"type": "log", "line": line, "seq": i, "ts": datetime.now(timezone.utc).isoformat()}
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.2)
        # Fake output
        out_lines = ["stdout: Hello from Private Server", "stdout: Process complete"]
        for l in out_lines:
            payload = {"type": "output", "data": l, "ts": datetime.now(timezone.utc).isoformat()}
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.15)
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


class FilesListRequest(BaseModel):
    connection: PrivateServerConfig
    path: str = "."
    depth: int = 1


@app.post("/api/private-server/files/list")
def private_server_files_list(req: FilesListRequest) -> Dict[str, Any]:
    # Mock response; later this can proxy to the private server file API
    now = datetime.now(timezone.utc).isoformat()
    return {
        "ok": True,
        "path": req.path,
        "items": [
            {"name": "app/", "type": "dir", "size": None, "modified": now},
            {"name": "app/main.py", "type": "file", "size": 1240, "modified": now},
            {"name": "README.md", "type": "file", "size": 342, "modified": now},
        ],
    }


# -----------------------------
# Conversations & Messages
# -----------------------------

class Attachment(BaseModel):
    type: Literal["text", "image", "audio", "file"] = "file"
    name: str
    url: Optional[str] = None
    mime: Optional[str] = None
    size: Optional[int] = None


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    created_at: Optional[datetime] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    tool_name: Optional[str] = None
    attachments: Optional[List[Attachment]] = None


class Conversation(BaseModel):
    title: str
    mode: Literal[
        "chat",
        "dev",
        "devops",
        "testing",
        "marketing",
        "multi_parallel",
        "multi_sequential",
    ] = "chat"
    provider: str = "openrouter"
    model: str = "openrouter/openai/gpt-4o"
    backend: Literal["private_server", "web_interpreter"] = "private_server"
    updated_at: Optional[datetime] = None
    messages: List[Message] = Field(default_factory=list)


@app.post("/api/conversations")
def create_conversation(conv: Conversation) -> Dict[str, Any]:
    conv_dict = conv.model_dump()
    now = datetime.now(timezone.utc)
    conv_dict["created_at"] = now
    conv_dict["updated_at"] = now
    cid = create_document("conversation", conv_dict)
    return {"id": cid}


@app.get("/api/conversations")
def list_conversations() -> List[Dict[str, Any]]:
    docs = get_documents("conversation", limit=50)
    return [serialize_doc(d) for d in docs]


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> Dict[str, Any]:
    from bson import ObjectId  # type: ignore

    try:
        docs = get_documents("conversation", {"_id": ObjectId(conversation_id)}, limit=1)
        if not docs:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return serialize_doc(docs[0])
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid conversation id")


# -----------------------------
# Streaming Chat (SSE - mock stream)
# -----------------------------

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    provider: str
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    thinking: Optional[bool] = False


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    # This is a mock SSE stream to demonstrate streaming UI. Replace with real provider calls later.
    def event_generator():
        import time

        intro = "Thanks for trying the multi-provider chat. This is a demo stream. "
        body = (
            "We will later connect real providers like OpenRouter, OpenAI, Claude, and Gemini. "
            "You can also enable the Private Server for Development to run code and tests."
        )
        full_text = intro + body
        for i in range(0, len(full_text), 6):
            chunk = full_text[i : i + 6]
            data = {"type": "chunk", "text": chunk}
            yield f"data: {data}\n\n"
            time.sleep(0.03)
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
