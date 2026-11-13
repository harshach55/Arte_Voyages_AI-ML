"""Small local API for the travel enquiry extractor.

Run from the project root:
    python app/uc2_api.py                 (then open http://127.0.0.1:8001/docs)
    python app/uc2_api.py --port 8002 --no-model

Endpoints:
    POST /extract         {"text": "...", "mode": "rules" | "rules_model" | "rules_tagger" | "rules_tagger_v2"}
    POST /extract/batch   {"texts": ["...", "..."], "mode": ...}   at most 50 texts
    GET  /health          whether the model files and the trained taggers are loaded

Privacy: the API never logs or echoes enquiry text. Error messages describe
the problem without repeating the input, and access logging is switched off.
The server listens on 127.0.0.1 only.
"""

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Callable, Literal, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, field_validator  # noqa: E402

from src.uc2_extractor import semantic, tagger  # noqa: E402
from src.uc2_extractor.extract import extract_profile  # noqa: E402
from src.uc2_extractor.schema import ClientProfile  # noqa: E402

MAX_CHARS = 20_000
MAX_BODY_BYTES = 200_000
MAX_BATCH = 50
MAX_BATCH_BODY_BYTES = MAX_BATCH * MAX_CHARS * 4 + 10_000
Mode = Literal["rules", "rules_model", "rules_tagger", "rules_tagger_v2"]
MODE_HELP = ("'rules' (default), 'rules_model' to add the optional model layer, 'rules_tagger' to let the first "
             "trained tagger fill what the rules left empty, or 'rules_tagger_v2' for the second tagger, also "
             "trained on synthetic enquiries.")


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="The enquiry, in English, French or German.", max_length=MAX_CHARS)
    mode: Mode = Field(default="rules", description=MODE_HELP)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text is empty")
        return v


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[Annotated[str, Field(max_length=MAX_CHARS)]] = Field(
        min_length=1, max_length=MAX_BATCH, description=f"1 to {MAX_BATCH} enquiries, each at most {MAX_CHARS} characters."
    )
    mode: Mode = Field(default="rules", description=MODE_HELP)

    @field_validator("texts")
    @classmethod
    def _none_blank(cls, v: list[str]) -> list[str]:
        blank = [str(i) for i, t in enumerate(v) if not t.strip()]
        if blank:
            raise ValueError(f"text number {', '.join(blank)} is empty (counting from 0)")
        return v


class BatchResponse(BaseModel):
    mode_requested: Mode
    mode_used: Mode
    note: Optional[str] = None
    count: int
    profiles: list[ClientProfile]


class ExtractResponse(BaseModel):
    mode_requested: Mode
    mode_used: Mode
    note: Optional[str] = Field(default=None, description="Set when the model or tagger was requested but not available.")
    profile: ClientProfile = Field(
        description="Validated client profile: values, evidence (source text with its position), "
                    "flags, and provenance (rule, model or crf, with the confidence of model and crf values)."
    )


class Health(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_message: str
    threshold: Optional[float]
    tagger_loaded: bool
    tagger_message: str
    tagger_v2_loaded: bool
    tagger_v2_message: str


def _short_errors(exc: RequestValidationError) -> list[str]:
    """Describe validation errors without repeating the submitted input."""
    out = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err.get("loc", ()) if p != "body") or "body"
        kind = err.get("type", "")
        if kind == "string_too_long":
            msg = f"must be at most {MAX_CHARS} characters"
        elif kind == "too_long":
            msg = f"must have at most {MAX_BATCH} texts"
        elif kind == "too_short":
            msg = "must have at least one text"
        elif kind == "list_type":
            msg = "must be a list of texts"
        elif kind == "missing":
            msg = "is required"
        elif kind == "literal_error":
            msg = "must be 'rules', 'rules_model', 'rules_tagger' or 'rules_tagger_v2'"
        elif kind == "extra_forbidden":
            msg = "is not an accepted field"
        elif kind in ("json_invalid", "model_attributes_type", "dict_type"):
            where, msg = "body", "must be a JSON object such as {\"text\": \"...\", \"mode\": \"rules\"}"
        elif kind == "value_error":
            msg = str(err.get("ctx", {}).get("error", "is not valid"))
        else:
            msg = "is not valid"
        out.append(f"{where} {msg}")
    return out


def create_app(load_model: bool = True, embedder: Optional[Callable] = None, load_tagger: bool = True,
               crf_tagger=None, crf_tagger_v2=None) -> FastAPI:
    """Build the API. `embedder` and the `crf_tagger` arguments replace the real ones (used in tests)."""
    state = {"embed": None, "message": "model not loaded", "tagger": None, "tagger_message": "tagger not loaded",
             "tagger_v2": None, "tagger_v2_message": "new tagger not loaded"}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if embedder is not None:
            state["embed"], state["message"] = embedder, "test embedder in use"
        elif not load_model:
            state["message"] = "model switched off (--no-model); rules only"
        else:
            try:
                embed = semantic.default_embedder()
                embed(["warm-up"])
                state["embed"] = embed
                state["message"] = f"model loaded (threshold {semantic.DEFAULT_THRESHOLD})"
            except Exception as exc:  # missing files or packages
                state["message"] = f"model not available ({type(exc).__name__}); rules only"
        if crf_tagger is not None:
            state["tagger"], state["tagger_message"] = crf_tagger, "test tagger in use"
        elif not load_tagger:
            state["tagger_message"] = "tagger switched off; rules only"
        else:
            available, state["tagger_message"] = tagger.check_tagger()
            state["tagger"] = tagger.default_tagger() if available else None
        if crf_tagger_v2 is not None:
            state["tagger_v2"], state["tagger_v2_message"] = crf_tagger_v2, "test tagger in use"
        elif load_tagger and crf_tagger is None:
            available, state["tagger_v2_message"] = tagger.check_tagger("v2")
            state["tagger_v2"] = tagger.default_tagger_v2() if available else None
        yield

    app = FastAPI(
        title="Travel Enquiry Extractor API",
        version="1.0",
        description="Turns a free-text travel enquiry (English, French or German) into a validated client "
                    "profile. Rules only by default; a trained tagger or an optional model layer can fill "
                    "fields the rules leave empty. Scored only on synthetic test sets; budget bands are "
                    "placeholders.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        size = request.headers.get("content-length")
        limit = MAX_BATCH_BODY_BYTES if request.url.path == "/extract/batch" else MAX_BODY_BYTES
        if size is not None and size.isdigit() and int(size) > limit:
            return JSONResponse(status_code=413, content={"errors": ["request body is too large"]})
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"errors": _short_errors(exc)})

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(
            status="ok",
            model_loaded=state["embed"] is not None,
            model_message=state["message"],
            threshold=semantic.DEFAULT_THRESHOLD if state["embed"] is not None else None,
            tagger_loaded=state["tagger"] is not None,
            tagger_message=state["tagger_message"],
            tagger_v2_loaded=state["tagger_v2"] is not None,
            tagger_v2_message=state["tagger_v2_message"],
        )

    def choose(mode: str) -> tuple[str, Optional[str], dict]:
        """The mode that can really be used, a note if it differs, and the arguments for the extractor."""
        needs = {"rules_model": ("embed", "The model is not available, so only the rules were used."),
                 "rules_tagger": ("tagger", "The trained tagger is not available, so only the rules were used."),
                 "rules_tagger_v2": ("tagger_v2", "The new tagger is not available, so only the rules were used.")}
        if mode in needs and state[needs[mode][0]] is None:
            return "rules", needs[mode][1], {}
        if mode == "rules_model":
            return mode, None, {"model": state["embed"]}
        if mode in ("rules_tagger", "rules_tagger_v2"):
            return mode, None, {"tagger": state[needs[mode][0]]}
        return mode, None, {}

    @app.post("/extract", response_model=ExtractResponse)
    def extract(req: ExtractRequest) -> ExtractResponse:
        mode_used, note, options = choose(req.mode)
        profile = extract_profile(req.text, **options)
        return ExtractResponse(mode_requested=req.mode, mode_used=mode_used, note=note, profile=profile)

    @app.post("/extract/batch", response_model=BatchResponse)
    def extract_batch(req: BatchRequest) -> BatchResponse:
        """Several enquiries in one call, in the same order. No text is logged or echoed in errors."""
        mode_used, note, options = choose(req.mode)
        profiles = [extract_profile(text, **options) for text in req.texts]
        return BatchResponse(mode_requested=req.mode, mode_used=mode_used, note=note, count=len(profiles),
                             profiles=profiles)

    return app


app = create_app()


def main(argv=None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description="Local API for the travel enquiry extractor.")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-model", action="store_true", help="do not load the model (rules only)")
    parser.add_argument("--no-tagger", action="store_true", help="do not load the trained tagger")
    args = parser.parse_args(argv)
    print(f"Travel Enquiry Extractor API at http://127.0.0.1:{args.port}  (docs: /docs)")
    uvicorn.run(create_app(load_model=not args.no_model, load_tagger=not args.no_tagger), host="127.0.0.1",
                port=args.port, access_log=False, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
