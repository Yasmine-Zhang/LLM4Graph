from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextlib
import os
import tempfile
import threading
from collections.abc import Coroutine
from typing import Any, Dict, List, Tuple

from copilot import CopilotClient
from copilot.session import ModelCapabilitiesOverride, ModelLimitsOverride
from copilot.session_events import AssistantMessageData

from src.llm_client.base_client import BaseClient


class _EventLoopRunner:
    """Run the asynchronous Copilot SDK behind BaseClient's synchronous API."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._started.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._started.set()
        self.loop.run_forever()

    def run(self, coroutine: Coroutine[Any, Any, Any], timeout: float) -> Any:
        if self._closed:
            coroutine.close()
            raise RuntimeError("GitHubCopilotClient is closed")
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"Copilot SDK operation timed out after {timeout}s"
            ) from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        self.loop.close()


_OPEN_CLIENTS: set[GitHubCopilotClient] = set()


def _close_open_clients() -> None:
    for client in list(_OPEN_CLIENTS):
        with contextlib.suppress(Exception):
            client.close()


atexit.register(_close_open_clients)


class GitHubCopilotClient(BaseClient):
    """GitHub Copilot SDK adapter for the LLM4Graph client contract.

    Set ``COPILOT_GITHUB_TOKEN`` (or ``github_token`` in the LLM config) to
    authenticate explicitly. Without a token, the SDK uses the locally logged-in
    GitHub Copilot user.
    """

    BASE_URL = "https://api.githubcopilot.com"

    def __init__(self, config: Dict):
        super().__init__(config)

        self.model = config.get("model", "gpt-4o")
        self.base_url = config.get("base_url", self.BASE_URL)
        self.github_token = str(
            config.get("github_token") or os.environ.get("COPILOT_GITHUB_TOKEN", "")
        ).strip()
        self.sdk_timeout = float(config.get("sdk_timeout", 180))
        self.sdk_log_level = config.get("sdk_log_level", "error")

        self._runner: _EventLoopRunner | None = None
        self._state_dir: tempfile.TemporaryDirectory[str] | None = None
        self._sdk_client: CopilotClient | None = None
        self._closed = False

    def _run_async(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        if self._closed:
            coroutine.close()
            raise RuntimeError("GitHubCopilotClient is closed")
        if self._runner is None:
            self._runner = _EventLoopRunner()
            _OPEN_CLIENTS.add(self)
        return self._runner.run(coroutine, self.sdk_timeout + 30)

    async def _ensure_runtime(self) -> None:
        if self._sdk_client is not None:
            return

        self._state_dir = tempfile.TemporaryDirectory(prefix="llm4graph-copilot-")
        client_options = {
            "use_logged_in_user": not bool(self.github_token),
            "mode": "empty",
            "base_directory": self._state_dir.name,
            "log_level": self.sdk_log_level,
        }
        if self.github_token:
            client_options["github_token"] = self.github_token
        if self.base_url:
            client_options["base_url"] = self.base_url

        self._sdk_client = CopilotClient(**client_options)
        try:
            await self._sdk_client.start()

            auth = await self._sdk_client.get_auth_status()
            if not auth.isAuthenticated:
                raise RuntimeError(
                    auth.statusMessage or "Copilot rejected the configured credentials"
                )

            available_models = {item.id for item in await self._sdk_client.list_models()}
            if self.model not in available_models:
                choices = ", ".join(sorted(available_models))
                raise RuntimeError(
                    f"Model {self.model!r} is unavailable for this account. "
                    f"Available: {choices}"
                )
        except Exception:
            with contextlib.suppress(Exception):
                await self._sdk_client.stop()
            self._sdk_client = None
            if self._state_dir is not None:
                self._state_dir.cleanup()
                self._state_dir = None
            raise

    @staticmethod
    def _split_messages(messages: List[Dict[str, str]]) -> tuple[str | None, str]:
        system_content = "\n\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "system" and message.get("content")
        )

        conversation = [message for message in messages if message.get("role") != "system"]
        if not conversation:
            raise ValueError("At least one non-system message is required")
        if len(conversation) == 1 and conversation[0].get("role") == "user":
            prompt = str(conversation[0].get("content", ""))
        else:
            transcript = []
            for message in conversation:
                role = str(message.get("role", "unknown")).upper()
                transcript.append(f"[{role}]\n{message.get('content') or ''}")
            prompt = "\n\n".join(transcript)

        return system_content or None, prompt

    async def _predict_async(self, messages: List[Dict[str, str]]) -> str:
        await self._ensure_runtime()
        assert self._sdk_client is not None

        system_content, prompt = self._split_messages(messages)
        session = await self._sdk_client.create_session(
            model=self.model,
            system_message=(
                {"mode": "replace", "content": system_content}
                if system_content
                else None
            ),
            infinite_sessions={"enabled": False},
            model_capabilities=ModelCapabilitiesOverride(
                limits=ModelLimitsOverride(max_output_tokens=self.max_tokens)
            ),
        )
        try:
            event = await session.send_and_wait(prompt, timeout=self.sdk_timeout)
            if event is None or not isinstance(event.data, AssistantMessageData):
                raise RuntimeError("Copilot returned no assistant message")
            content = str(event.data.content or "").strip()
            if not content:
                raise RuntimeError("Copilot returned an empty assistant message")
            return content
        finally:
            with contextlib.suppress(Exception):
                await session.disconnect()

    def predict_once(self, messages: List[Dict[str, str]]) -> Tuple[str, float]:
        """Return response text and a sentinel confidence (SDK has no logprobs)."""
        content = self._run_async(self._predict_async(messages))
        return content, -999.0

    async def _shutdown(self) -> None:
        if self._sdk_client is not None:
            await self._sdk_client.stop()
            self._sdk_client = None

    def close(self) -> None:
        if self._closed:
            return
        if self._runner is not None:
            with contextlib.suppress(Exception):
                self._runner.run(self._shutdown(), self.sdk_timeout)
            self._runner.close()
            self._runner = None
        if self._state_dir is not None:
            self._state_dir.cleanup()
            self._state_dir = None
        self._closed = True
        _OPEN_CLIENTS.discard(self)

    def __enter__(self) -> GitHubCopilotClient:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()