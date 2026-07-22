"""
Configuration management for APS-RAG.

Precedence (highest first):
  1. Environment variables (secrets live here; see .env / .env.example)
  2. YAML config:  config/config.yaml  (your copy — gitignored)
                   → falls back to config/config.example.yaml if present
                   → else config/environments/base.yaml (shipped safe defaults)
  3. The Pydantic defaults in this file.

Secrets (API keys, passwords) are read from the environment — never hard-code
them in YAML. A local `.env` file is auto-loaded if present.
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent

# Auto-load THIS project's .env (secrets) if python-dotenv is available. Scoped
# to PROJECT_ROOT/.env so we never pick up an unrelated .env from a parent dir.
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:  # pragma: no cover
    pass


class LLMSettings(BaseSettings):
    """LLM backend — provider-agnostic (the public DEFAULT).

    provider="openai_compatible" points the client at ANY OpenAI-compatible
    gateway (OpenAI, Azure OpenAI, vLLM, Ollama, LM Studio, TGI, …) via
    base_url + api_key + model. provider="argo" selects the bundled, optional
    Argo backend (institution-internal; see src/integrations/argo/).

    Per-role model overrides let you route, e.g., synthesis to a big model and
    routing to a small one; each falls back to `model` when unset.
    """
    provider: str = "openai_compatible"          # "openai_compatible" | "argo"
    base_url: str = "https://api.openai.com/v1"   # env: LLM_BASE_URL
    api_key: str = ""                             # env: LLM_API_KEY (secret)
    model: str = "gpt-4o-mini"                    # env: LLM_MODEL — default for all roles
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 240.0
    # Optional per-role model overrides (blank → use `model`)
    synthesis_model: str = ""
    routing_model: str = ""
    critique_model: str = ""
    judge_model: str = ""
    extraction_model: str = ""

    model_config = {"populate_by_name": True, "env_prefix": "LLM_"}


class ArgoSettings(BaseSettings):
    """Optional institution-internal LLM gateway (opt-in via llm.provider=argo)."""
    base_url: str = "https://your-argo-gateway.example.com/api/v1/resource"
    primary_model: str = "gpt-4o"
    fast_model: str = "gpt-4o-mini"
    critique_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o"
    username: str = Field(default="your-username", alias="ARGO_USERNAME")
    temperature: float = 0.0
    max_tokens: int = 100_000
    timeout: float = 240.0

    model_config = {"populate_by_name": True, "env_prefix": "ARGO_"}


class VectorStoreSettings(BaseSettings):
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    es_host: str = "localhost"
    es_port: int = 9200
    es_username: str = Field(default="elastic", alias="ELASTICSEARCH_USERNAME")
    es_password: str = Field(default="change-me", alias="ELASTICSEARCH_PASSWORD")
    embedding_model: str = "intfloat/e5-large-v2"
    embedding_device: str = "cuda:0"   # override to "cpu" on CPU-only hosts
    embedding_dim: int = 1024

    model_config = {"populate_by_name": True}


class GraphSettings(BaseSettings):
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="change-me", alias="NEO4J_PASSWORD")

    model_config = {"populate_by_name": True}


class AgentSettings(BaseSettings):
    max_react_iterations: int = 10
    quality_threshold: float = 0.6       # Below → corrective RAG triggers
    max_correction_rounds: int = 3
    irrelevant_doc_threshold: float = 0.5  # >50% irrelevant → re-retrieve
    tool_timeout_seconds: int = 30
    tool_chain_safety_cap: int = 8  # Hard upper bound on consecutive tool rounds
    use_create_react_agent: bool = False  # Opt-in to LangGraph prebuilt ReAct path
    tools_llm_model: str = ""             # blank → use llm.model
    planner_llm_model: str = ""
    enable_web_search: bool = False       # site-specific tool, off by default
    enable_code_execution: bool = False   # site-specific tool, off by default
    top_k: int = 100
    reranker_top_k: int = 50
    # "jina_v3" (cross-encoder, GPU-local, deterministic, default) | "llm"
    # (GraphProximityReranker that calls the LLM per query — for ablation).
    reranker_type: str = "jina_v3"
    reranker_model_name: str = "jinaai/jina-reranker-v3"

    model_config = {"populate_by_name": True}


class ChunkingSettings(BaseSettings):
    """Chunking strategy configuration for vectorstore building."""
    strategy: str = "granular-chunking"  # "whole-doc" or "granular-chunking"
    pre_chunk_size: int = 1000
    pre_chunk_overlap: int = 200
    max_chunk_size: int = 4000
    chunk_overlap: int = 200
    collection_prefix_gc: str = "agrag_gc_"  # prefix for granular-chunking collections
    collection_prefix_wd: str = "agrag_"     # prefix for whole-doc collections

    model_config = {"populate_by_name": True}


class PVSettings(BaseSettings):
    """Optional site-specific control-system (EPICS) tool config. Disabled by
    default; all hosts are placeholders — set to your own if you enable it."""
    enable_pv_tools: bool = False
    epics_gateway_address: str = "your-epics-gateway.example.com"
    epics_gateway_port: int = 5064
    epics_timeout: float = 5.0
    archiver_url: str = "https://your-archiver.example.com"
    archiver_timeout: int = 60
    channel_db_path: str = "data/channel_databases/hierarchical.json"
    in_context_db_path: str = "data/channel_databases/its_pv_in_context.json"
    max_pvs_per_read: int = 50

    model_config = {"populate_by_name": True}


class APISettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    openwebui_port: int = 3000
    title: str = "APS-RAG API"
    version: str = "0.1.0"
    request_timeout: int = 300

    model_config = {"populate_by_name": True}


class AppSettings(BaseSettings):
    environment: str = "development"

    llm: LLMSettings = Field(default_factory=LLMSettings)
    argo: ArgoSettings = Field(default_factory=ArgoSettings)
    vector: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    pv: PVSettings = Field(default_factory=PVSettings)
    api: APISettings = Field(default_factory=APISettings)

    model_config = {"populate_by_name": True, "env_nested_delimiter": "__"}

    @classmethod
    def _resolve_yaml(cls) -> Optional[Path]:
        for candidate in (
            CONFIG_DIR / "config.yaml",
            CONFIG_DIR / "config.example.yaml",
            CONFIG_DIR / "environments" / "base.yaml",
        ):
            if candidate.exists():
                return candidate
        return None

    @classmethod
    def from_yaml(cls, yaml_path: Optional[Path] = None) -> "AppSettings":
        """Load settings from YAML (config.yaml preferred), then env overrides."""
        path = yaml_path or cls._resolve_yaml()
        if path and path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Cached settings singleton."""
    return AppSettings.from_yaml()


def validate_settings(settings: Optional[AppSettings] = None, require_llm: bool = False) -> List[str]:
    """Validate config and return a list of human-readable problems.

    When `require_llm` is True (query / KG build / InPARS generation), the LLM
    backend must be usable. Ingestion of vectors + BM25 does NOT require an LLM
    (embeddings run locally), so `require_llm=False` there.
    """
    s = settings or get_settings()
    problems: List[str] = []

    if require_llm:
        prov = s.llm.provider
        if prov == "openai_compatible":
            if not s.llm.base_url:
                problems.append("LLM base URL is not set — set LLM_BASE_URL (e.g. https://api.openai.com/v1) or edit config llm.base_url.")
            is_local = any(h in s.llm.base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
            if not s.llm.api_key and not is_local:
                problems.append("LLM_API_KEY is not set — copy .env.example to .env and fill it in (local gateways like Ollama may leave it blank).")
            if not s.llm.model:
                problems.append("LLM model is not set — set LLM_MODEL or config llm.model.")
        elif prov == "argo":
            if not s.argo.base_url or "example.com" in s.argo.base_url:
                problems.append("llm.provider=argo but argo.base_url is a placeholder — set ARGO_BASE_URL / argo.base_url.")
        else:
            problems.append(f"Unknown llm.provider '{prov}' (expected 'openai_compatible' or 'argo').")

    return problems


def require_valid_settings(require_llm: bool = False) -> AppSettings:
    """Fail fast with an actionable message if config is invalid."""
    s = get_settings()
    problems = validate_settings(s, require_llm=require_llm)
    if problems:
        raise SystemExit("Configuration error(s):\n  - " + "\n  - ".join(problems))
    return s
