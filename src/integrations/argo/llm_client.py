"""
Argo LLM client (OPTIONAL, self-contained backend).

Provides ArgoLLMClientAG (a model-aware subclass of the vendored ArgoLLMClient
base) and LLMFactory. Argo is an institution-internal, OpenAI-INcompatible
gateway kept here as an opt-in backend. The public DEFAULT LLM backend is
OpenAI-compatible — see `src/integrations/llm/` and the `llm.provider` config
key. Select this backend with `llm.provider: argo`.

ArgoLLMClientAG overrides _generate() to send the correct API parameters per
model family:
  - gpt-/o-series : max_completion_tokens + temperature + top_p
  - claude-family : max_tokens + temperature  (no top_p together)
  - others        : max_tokens + temperature
"""
import inspect
import logging
from typing import List, Optional, Any

from src.integrations.argo.argo_base_client import ArgoLLMClient  # vendored base

logger = logging.getLogger(__name__)

# Model-family prefixes that use max_completion_tokens instead of max_tokens
_GPT_O_MODELS = {"gpt-4o", "gpt-4", "gpt-5", "o1", "o3", "o4"}

# Model families that reject temperature + top_p together (Claude via a gateway)
_CLAUDE_MODELS = {"claude"}


class ArgoLLMClientAG(ArgoLLMClient):
    """
    Model-aware ArgoLLMClient for use within agentic-graph-rag.

    Adds top_p field and overrides _generate() to send correct API parameters:
    - gpt-4o/gpt-4o/o-series: max_completion_tokens (NOT max_tokens), temperature, top_p
    - claude models:        max_tokens, temperature, top_p
    - others:               max_tokens, temperature

    This class must NOT modify entry_rag_app — it only exists in agentic-graph-rag.
    """

    top_p: float = 0.9

    def _build_payload(self, argo_messages, stop) -> dict:
        """Build model-specific API payload."""
        model_key = self.model.lower()
        data = {
            "user": self.username,
            "model": self.model,
            "messages": argo_messages,
            "stop": stop or [],
            "temperature": self.temperature,
        }
        # Claude models reject having both temperature and top_p
        is_claude = any(model_key.startswith(k) for k in ("claude",))
        if not is_claude:
            data["top_p"] = self.top_p
        # GPT / O-series use max_completion_tokens, not max_tokens
        if any(model_key.startswith(k) for k in ("gpt-4o", "gpt-4", "gpt-5", "o1", "o3", "o4")):
            data["max_completion_tokens"] = self.max_tokens
        else:
            # Claude models and others use max_tokens
            data["max_tokens"] = self.max_tokens
        return data

    # Soft preflight cap (~20k tokens). Above this, truncate the longest user
    # message so we don't hand Argo a request it will reject with 500.
    _PREFLIGHT_INPUT_LIMIT = 80_000
    _PREFLIGHT_TARGET = 60_000

    @staticmethod
    def _truncate_to_target(messages: List, target_chars: int) -> List:
        """Return a new message list with the longest message truncated so total
        content length is <= target_chars. Truncates from the head (keeps the
        tail, where instructions and the question typically sit)."""
        from langchain_core.messages import HumanMessage
        sizes = [len(getattr(m, "content", "") or "") for m in messages]
        total = sum(sizes)
        if total <= target_chars:
            return messages

        # Prefer truncating the latest HumanMessage (typically carries the bulk
        # of retrieval context) so system instructions remain intact.
        idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                idx = i
                break
        if idx == -1:
            # Fallback: truncate the largest message.
            idx = sizes.index(max(sizes))

        overflow = total - target_chars
        old = messages[idx].content or ""
        keep = max(1000, len(old) - overflow)
        truncated_content = (
            "(context truncated by Argo client to fit upstream limit)\n…\n"
            + old[-keep:]
        )
        new_messages = list(messages)
        new_messages[idx] = type(messages[idx])(content=truncated_content)
        return new_messages

    @staticmethod
    def _detect_caller() -> str:
        """Best-effort caller name for preflight diagnostics."""
        try:
            stack = inspect.stack()
            # Skip this helper and _generate frames
            for frame_info in stack[2:10]:
                module = inspect.getmodule(frame_info.frame)
                mod_name = module.__name__ if module else ""
                if "argo.llm_client" not in mod_name:
                    return f"{mod_name}.{frame_info.function}" if mod_name else frame_info.function
        except Exception:
            pass
        return "unknown"

    def _post_once(self, messages: List, stop, chat_url: str, headers: dict):
        """Single POST attempt. Returns (status_code, body_text)."""
        import httpx
        argo_messages = self._convert_messages_to_argo_format(messages)
        data = self._build_payload(argo_messages, stop)
        full_response = ""
        with httpx.stream(
            "POST", chat_url, json=data, headers=headers, timeout=self.timeout
        ) as response:
            status = response.status_code
            if status == 200:
                for chunk in response.iter_text():
                    if chunk:
                        full_response += chunk
            else:
                # Drain the body so the connection closes cleanly
                try:
                    full_response = "".join(response.iter_text())
                except Exception:
                    full_response = ""
        return status, full_response

    def _generate(
        self,
        messages: List,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ):
        import httpx
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatResult, ChatGeneration

        chat_url = self._get_chat_url()

        # Preflight: cap total input size before posting.
        total_chars = sum(len(getattr(m, "content", "") or "") for m in messages)
        if total_chars > self._PREFLIGHT_INPUT_LIMIT:
            caller = self._detect_caller()
            logger.warning(
                "ARGO PREFLIGHT TRUNCATE | caller=%s model=%s total_chars=%d > limit=%d, "
                "shrinking to %d", caller, self.model, total_chars,
                self._PREFLIGHT_INPUT_LIMIT, self._PREFLIGHT_TARGET,
            )
            messages = self._truncate_to_target(messages, self._PREFLIGHT_TARGET)

        logger.info("ARGO API CALL | model=%s url=%s msgs=%d input_chars=%d",
                    self.model, chat_url, len(messages),
                    sum(len(getattr(m, "content", "") or "") for m in messages))

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/plain",
            "Cache-Control": "no-cache",
            "Accept-Encoding": "identity",
        }

        try:
            status, body = self._post_once(messages, stop, chat_url, headers)

            if status == 500:
                # Retry once with input shrunk to half — handles upstream
                # token-limit 500s (Argo returns 500 not 413). Bounded: one
                # retry only, no exponential backoff.
                cur_total = sum(len(getattr(m, "content", "") or "") for m in messages)
                retry_target = max(8_000, cur_total // 2)
                logger.warning(
                    "ARGO 500 — retrying with input truncated to %d chars (was %d)",
                    retry_target, cur_total,
                )
                retry_messages = self._truncate_to_target(messages, retry_target)
                # Ensure user-visible marker for the retry path.
                for i in range(len(retry_messages) - 1, -1, -1):
                    content = getattr(retry_messages[i], "content", "") or ""
                    if content:
                        retry_messages[i] = type(retry_messages[i])(
                            content=(
                                "(context truncated due to upstream limit)\n\n"
                                + content
                            )
                        )
                        break
                status, body = self._post_once(
                    retry_messages, stop, chat_url, headers,
                )

            if status != 200:
                body_snippet = (body or "").strip()[:500].replace("\n", " ")
                error_msg = f"Argo API error: {status} | body: {body_snippet}"
                logger.error(error_msg)
                raise Exception(error_msg)

            logger.info("Response received: %d chars", len(body))
            message = AIMessage(content=body.strip())
            return ChatResult(generations=[ChatGeneration(message=message)])

        except httpx.TimeoutException as e:
            logger.error("Argo API timeout after %ss: %s", self.timeout, e)
            raise Exception(f"Argo API timeout: {e}")
        except Exception as e:
            logger.error("Argo API error: %s", e)
            raise

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, None, **kwargs)


class LLMFactory:
    """Provider-aware LLM factory.

    Returns a LangChain chat model for each use-case, dispatching on the
    configured backend (`llm.provider`):
      - "openai_compatible" (DEFAULT): a `langchain_openai.ChatOpenAI` pointed at
        the configured base_url / api_key / model (works with OpenAI, Azure,
        vLLM, Ollama, LM Studio, TGI, …). Per-role model overrides apply.
      - "argo" (opt-in): the bundled institution-internal Argo client.

    Call sites pass the Argo username/base_url (used only for the argo path);
    the openai path reads everything from `settings.llm`.
    """

    # ── internals ─────────────────────────────────────────────────────────
    @staticmethod
    def _cfg():
        from config.settings import get_settings
        return get_settings()

    @staticmethod
    def _openai_chat(role: str, max_tokens: int):
        """Build a ChatOpenAI for an OpenAI-compatible gateway."""
        from langchain_openai import ChatOpenAI
        llm = LLMFactory._cfg().llm
        model = getattr(llm, f"{role}_model", "") or llm.model
        return ChatOpenAI(
            base_url=llm.base_url,
            api_key=(llm.api_key or "not-needed"),   # local gateways ignore the key
            model=model,
            temperature=llm.temperature,
            max_tokens=max_tokens or llm.max_tokens,
            timeout=llm.timeout,
        )

    @staticmethod
    def _argo(model: str, username: str, base_url: str, max_tokens: int, tools: bool = False):
        if tools:
            from src.integrations.argo.native_tools_client import NativeToolsArgoLLM
            return NativeToolsArgoLLM(
                model=model, username=username, base_url=base_url,
                temperature=0.0, top_p=0.9, max_tokens=max_tokens,
            )
        return ArgoLLMClientAG(
            model=model, username=username, base_url=base_url,
            temperature=0.0, top_p=0.9, max_tokens=max_tokens,
        )

    # ── per-role clients ──────────────────────────────────────────────────
    @staticmethod
    def get_synthesis_llm(username: str, base_url: str):
        """Primary answer synthesis (tool-capable)."""
        cfg = LLMFactory._cfg()
        if cfg.llm.provider == "argo":
            return LLMFactory._argo(cfg.argo.primary_model, username, base_url, 8_000, tools=True)
        return LLMFactory._openai_chat("synthesis", 8_000)

    @staticmethod
    def get_answer_llm(username: str, base_url: str):
        """Optional high-capacity answer model."""
        cfg = LLMFactory._cfg()
        if cfg.llm.provider == "argo":
            return LLMFactory._argo(cfg.argo.primary_model, username, base_url, 12_000)
        return LLMFactory._openai_chat("synthesis", 12_000)

    @staticmethod
    def get_haiku_llm(username: str, base_url: str):
        """Fast, cheap model for light structured tasks / classification."""
        cfg = LLMFactory._cfg()
        if cfg.llm.provider == "argo":
            return LLMFactory._argo(cfg.argo.fast_model, username, base_url, 1024)
        return LLMFactory._openai_chat("routing", 1024)

    @staticmethod
    def get_routing_llm(username: str, base_url: str):
        """Routing, intent classification, entity extraction."""
        cfg = LLMFactory._cfg()
        if cfg.llm.provider == "argo":
            return LLMFactory._argo(cfg.argo.fast_model, username, base_url, 2048)
        return LLMFactory._openai_chat("routing", 2048)

    @staticmethod
    def get_critique_llm(username: str, base_url: str):
        """Self-critique quality gate."""
        cfg = LLMFactory._cfg()
        if cfg.llm.provider == "argo":
            return LLMFactory._argo(cfg.argo.critique_model, username, base_url, 2048)
        return LLMFactory._openai_chat("critique", 2048)

    @staticmethod
    def get_judge_llm(username: str, base_url: str):
        """LLM-as-judge evaluation."""
        cfg = LLMFactory._cfg()
        if cfg.llm.provider == "argo":
            return LLMFactory._argo(cfg.argo.judge_model, username, base_url, 4096)
        return LLMFactory._openai_chat("judge", 4096)

    @staticmethod
    def get_extraction_llm(username: str, base_url: str):
        """Entity + relation extraction for KG build."""
        cfg = LLMFactory._cfg()
        if cfg.llm.provider == "argo":
            return LLMFactory._argo(cfg.argo.primary_model, username, base_url, 4096)
        return LLMFactory._openai_chat("extraction", 4096)

    @staticmethod
    def get_tools_llm(username: str, base_url: str, model: str = ""):
        """Tool-calling client for the (opt-in) create_react_agent path."""
        cfg = LLMFactory._cfg()
        if cfg.llm.provider == "argo":
            return LLMFactory._argo(model or cfg.argo.primary_model, username, base_url, 4096, tools=True)
        return LLMFactory._openai_chat("synthesis", 4096)

    @staticmethod
    def get_planner_llm(username: str, base_url: str, model: str = ""):
        """Planner client (short JSON plans)."""
        cfg = LLMFactory._cfg()
        if cfg.llm.provider == "argo":
            return LLMFactory._argo(model or cfg.argo.primary_model, username, base_url, 4096, tools=True)
        return LLMFactory._openai_chat("routing", 4096)

    @staticmethod
    def create_embeddings(settings):
        """Create HuggingFace e5-large-v2 embeddings from AppSettings.

        Uses the local intfloat/e5-large-v2 model (1024-dim, CUDA-accelerated).
        No external API dependency — avoids transient 502/timeout failures.
        """
        import torch
        from langchain_huggingface import HuggingFaceEmbeddings

        device = settings.vector.embedding_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU")
            device = "cpu"

        model = HuggingFaceEmbeddings(
            model_name=settings.vector.embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32 if device.startswith("cuda") else 16,
            },
        )
        logger.info(
            "Embeddings initialized: model=%s, device=%s",
            settings.vector.embedding_model,
            device,
        )
        return model
