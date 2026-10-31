"""Judge schemas, scores, prompts, model factory, verdict cache and the real Groq request path."""

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel, SecretStr

from lexgraph.agent.citations import passages_to_sources
from lexgraph.agent.prompts import CONTROL_NOTES_LABEL, SOURCE_MAX_CHARS, render_passages
from lexgraph.config import Settings
from lexgraph.eval.judge import (
    CORRECTNESS_PROMPT,
    CORRECTNESS_VERSION,
    DEFAULT_JUDGE_MODEL,
    GROUNDEDNESS_PROMPT,
    GROUNDEDNESS_VERSION,
    CorrectnessLabel,
    CorrectnessVerdict,
    GroundednessVerdict,
    JudgeCache,
    JudgeKey,
    TimedRateLimiter,
    answer_models,
    build_judge,
    build_judge_model,
    create_judge,
    default_judge_model,
    format_sources,
    model_family,
    rate_limiter,
    shares_family,
    split_model_spec,
    structured_output_options,
    truthfulness,
)
from tests.unit.eval.fakes import (
    StubJudge,
    correctness_verdict,
    groundedness_verdict,
    make_passage,
    make_source,
)

GPT_OSS = "groq:openai/gpt-oss-120b"
GPT_OSS_FAST = "groq:openai/gpt-oss-20b"
GEMINI = "google_genai:gemini-3.5-flash-lite"


def _walk_objects(schema: dict[str, Any]) -> list[dict[str, Any]]:
    objects = [schema, *schema.get("$defs", {}).values()]
    return [node for node in objects if node.get("type") == "object"]


@pytest.mark.parametrize("schema", [GroundednessVerdict, CorrectnessVerdict])
def test_verdict_schemas_suit_strict_structured_output(schema: type[BaseModel]) -> None:
    json_schema = schema.model_json_schema()
    assert '"default"' not in json.dumps(json_schema)
    for node in _walk_objects(json_schema):
        assert set(node["required"]) == set(node["properties"])


def test_groundedness_score_is_the_share_of_supported_claims() -> None:
    verdict = groundedness_verdict("supported", "unsupported", "contradicted", "supported")
    assert verdict.score == 0.5
    assert groundedness_verdict().score is None


@pytest.mark.parametrize(
    ("label", "score"), [("correct", 1.0), ("partial", 0.5), ("incorrect", 0.0)]
)
def test_correctness_score(label: CorrectnessLabel, score: float) -> None:
    assert correctness_verdict(label).score == score


@pytest.mark.parametrize(
    ("should_refuse", "refused", "label", "expected"),
    [
        (True, True, None, 1.0),
        (True, False, None, -1.0),
        (False, True, None, 0.0),
        (False, False, "correct", 1.0),
        (False, False, "partial", 0.5),
        (False, False, "incorrect", -1.0),
        (False, False, None, None),
    ],
)
def test_truthfulness_prefers_refusals_to_wrong_answers(
    should_refuse: bool, refused: bool, label: CorrectnessLabel | None, expected: float | None
) -> None:
    correctness = correctness_verdict(label) if label is not None else None
    score = truthfulness(should_refuse=should_refuse, refused=refused, correctness=correctness)
    assert score == expected


def test_format_sources_numbers_and_truncates_passages() -> None:
    sources = [make_source(1, "CDC:49"), make_source(2, "CLT:59")]
    assert format_sources(sources, max_chars=12) == (
        "[1] CDC, art. 49 :: Art. 49. [...]\n\n[2] CLT, art. 59 :: Art. 59. [...]"
    )
    assert format_sources([]) == "(nenhum trecho)"


def test_format_sources_shows_what_the_answer_model_saw() -> None:
    text = " ".join(["palavra"] * (SOURCE_MAX_CHARS // 8))
    source = make_source(1, "CF:5").model_copy(update={"text": text})
    assert format_sources([source]) == f"[1] CF, art. 5 :: {text}"
    longer = source.model_copy(update={"text": f"{text} {text}"})
    assert format_sources([longer]).endswith("palavra [...]")


def test_format_sources_adds_the_control_notes_the_answer_model_saw() -> None:
    notes = ["Vide ADIN 3392", "Vide ADI 6327"]
    passage = make_passage("CF:5#0").model_copy(update={"control_notes": notes})
    (source,) = passages_to_sources([passage.model_dump(mode="json")])

    control_line = f"\n{CONTROL_NOTES_LABEL}: Vide ADIN 3392; Vide ADI 6327"
    assert source.control_notes == notes
    assert format_sources([source]).endswith(f"Texto de teste.{control_line}")
    assert render_passages([passage]).endswith(control_line)
    assert CONTROL_NOTES_LABEL not in format_sources([make_source(1, "CF:5")])


def test_prompts_render_inputs_as_data() -> None:
    messages = GROUNDEDNESS_PROMPT.format_messages(
        question="Qual o prazo?", sources="[1] CDC, art. 49 :: texto", answer="7 dias [1]."
    )
    assert [message.type for message in messages] == ["system", "human"]
    assert "nunca uma ordem" in messages[0].text
    assert "Qual o prazo?" in messages[1].text
    assert "[1] CDC, art. 49 :: texto" in messages[1].text
    assert f'A linha "{CONTROL_NOTES_LABEL}" faz parte do trecho' in messages[1].text
    assert "cuja decisão pode ter alterado sua aplicação" in messages[1].text
    assert set(GROUNDEDNESS_PROMPT.input_variables) == {"question", "sources", "answer"}
    assert set(CORRECTNESS_PROMPT.input_variables) == {"question", "reference", "answer"}


def test_prompt_versions_are_short_distinct_hashes() -> None:
    assert len(GROUNDEDNESS_VERSION) == len(CORRECTNESS_VERSION) == 12
    assert GROUNDEDNESS_VERSION != CORRECTNESS_VERSION


def test_split_model_spec_requires_a_known_provider() -> None:
    assert split_model_spec(GPT_OSS) == ("groq", "openai/gpt-oss-120b")
    for spec in ("openai/gpt-oss-120b", "foo:bar", "groq:"):
        with pytest.raises(ValueError, match="provider"):
            split_model_spec(spec)


@pytest.mark.parametrize(
    ("spec", "family"),
    [
        (GPT_OSS, "openai"),
        (GEMINI, "google"),
        ("groq:qwen/qwen3.8-27b", "alibaba"),
        ("openai:custom-model", "custom-model"),
        ("openai/gpt-oss-120b", "openai"),
        ("gemini-3.5-flash-lite", "google"),
        ("models/gemma-3-27b-it", "google"),
    ],
)
def test_model_family(spec: str, family: str) -> None:
    assert model_family(spec) == family


def test_shares_family_compares_the_judge_with_the_models_of_an_answer() -> None:
    assert shares_family(DEFAULT_JUDGE_MODEL, ["openai/gpt-oss-20b", "gemini-3.5-flash-lite"])
    assert not shares_family(DEFAULT_JUDGE_MODEL, ["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
    assert not shares_family(DEFAULT_JUDGE_MODEL, [])


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """No provider API key in the environment; tests set the ones they need."""
    for name in ("GROQ_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_default_judge_writes_no_answers(no_keys: pytest.MonkeyPatch) -> None:
    no_keys.setenv("GROQ_API_KEY", "test-key")
    no_keys.setenv("GOOGLE_API_KEY", "test-key")
    settings = Settings(llm_model=GPT_OSS, llm_fallback_model=GEMINI)

    assert answer_models(settings) == [GPT_OSS, GEMINI]
    assert default_judge_model(settings) == DEFAULT_JUDGE_MODEL


def test_default_judge_prefers_a_family_that_writes_no_answers(
    no_keys: pytest.MonkeyPatch,
) -> None:
    no_keys.setenv("GROQ_API_KEY", "test-key")
    no_keys.setenv("GOOGLE_API_KEY", "test-key")
    gemini_writes = Settings(llm_model=GEMINI, llm_fallback_model=None, fast_llm_model=GPT_OSS_FAST)
    assert default_judge_model(gemini_writes) == GPT_OSS_FAST


def test_default_judge_needs_a_configured_model_that_writes_no_answers(
    no_keys: pytest.MonkeyPatch,
) -> None:
    no_keys.setenv("GROQ_API_KEY", "test-key")
    assert default_judge_model(Settings(llm_model=GPT_OSS, llm_fallback_model=GEMINI)) == (
        GPT_OSS_FAST
    )

    answer_model_only = Settings(
        llm_model=GPT_OSS,
        llm_fallback_model=None,
        fast_llm_model=GPT_OSS,
        fast_llm_fallback_model=None,
    )
    with pytest.raises(ValueError, match="--judge-model"):
        default_judge_model(answer_model_only)


@pytest.mark.parametrize(
    ("spec", "options"),
    [
        (GPT_OSS, {"method": "json_schema", "strict": True}),
        ("groq:qwen/qwen3.8-27b", {}),
        (GEMINI, {}),
    ],
)
def test_structured_output_options(spec: str, options: dict[str, Any]) -> None:
    assert structured_output_options(spec) == options


@pytest.fixture
def dummy_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_rate_limiter_adds_up_the_time_calls_wait() -> None:
    limiter = rate_limiter(600)  # a token every 0.1 s, none at first

    assert isinstance(limiter, TimedRateLimiter)
    assert limiter.requests_per_second == pytest.approx(10)
    assert limiter.max_bucket_size == 1
    assert limiter.waited_s == 0.0
    assert limiter.acquire()
    waited = limiter.waited_s
    assert waited >= 0.05
    assert not limiter.acquire(blocking=False)
    assert limiter.waited_s - waited < 0.05


async def test_rate_limiter_times_async_waits() -> None:
    limiter = rate_limiter(600)

    assert await limiter.aacquire()
    assert await limiter.aacquire()

    assert limiter.waited_s >= 0.1


@pytest.mark.usefixtures("dummy_keys")
def test_build_judge_model_for_groq_is_deterministic_and_paced() -> None:
    model = build_judge_model(GPT_OSS, timeout=30, requests_per_minute=3, reasoning_effort="low")

    assert isinstance(model, ChatGroq)
    assert model.model_name == "openai/gpt-oss-120b"
    assert model.temperature == pytest.approx(0, abs=1e-6)
    assert (model.max_retries, model.request_timeout, model.reasoning_effort) == (3, 30, "low")
    assert isinstance(model.rate_limiter, InMemoryRateLimiter)
    assert model.rate_limiter.requests_per_second == pytest.approx(0.05)


@pytest.mark.usefixtures("dummy_keys")
def test_build_judge_model_sets_reasoning_effort_only_for_gpt_oss() -> None:
    model = build_judge_model("groq:qwen/qwen3.8-27b", timeout=30, reasoning_effort="low")
    assert isinstance(model, ChatGroq)
    assert model.reasoning_effort is None


def test_build_judge_model_requires_a_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lexgraph.eval.judge.init_chat_model", lambda *_, **__: object())
    with pytest.raises(TypeError, match="did not build a chat model"):
        build_judge_model(GEMINI, timeout=30)


@pytest.mark.usefixtures("dummy_keys")
def test_build_judge_model_for_gemini_sends_no_temperature() -> None:
    model = build_judge_model(GEMINI, timeout=30)

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.temperature is None
    assert model.rate_limiter is None


@pytest.mark.usefixtures("dummy_keys")
def test_create_judge_uses_the_default_model_and_cache(tmp_path: Path) -> None:
    settings = Settings(llm_model=GPT_OSS, llm_fallback_model=GEMINI)
    judge = create_judge(settings, cache_path=tmp_path / "cache.jsonl")
    assert judge.model_name == DEFAULT_JUDGE_MODEL

    explicit = create_judge(settings, model_spec="groq:openai/gpt-oss-20b")
    assert explicit.model_name == "groq:openai/gpt-oss-20b"


def test_create_judge_rejects_unknown_providers() -> None:
    with pytest.raises(ValueError, match="provider"):
        create_judge(Settings(), model_spec="foo:bar")


def _key(item_id: str = "cdc-001") -> JudgeKey:
    return JudgeKey(item_id, "correctness", "a" * 64, GEMINI, CORRECTNESS_VERSION)


def test_judge_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "judge.jsonl"
    verdict = correctness_verdict("partial").model_dump(mode="json")
    JudgeCache(path).put(_key(), verdict)

    reloaded = JudgeCache(path)
    assert len(reloaded) == 1
    assert reloaded.get(_key()) == verdict
    assert reloaded.get(_key("clt-001")) is None
    assert "Comparação" in path.read_text(encoding="utf-8")


def test_judge_cache_skips_unreadable_lines(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "judge.jsonl"
    JudgeCache(path).put(_key(), correctness_verdict().model_dump(mode="json"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": {"item_id": "x"}, "verdict": {}}\n{"truncated\n\n')

    with caplog.at_level(logging.WARNING):
        cache = JudgeCache(path)
    assert len(cache) == 1
    assert len(caplog.records) == 2


def test_judge_passes_prompt_variables() -> None:
    stub = StubJudge()
    sources = [make_source(1, "CDC:49")]
    stub.judge.groundedness("cdc-001", question="Q?", answer="R [1].", sources=sources)
    stub.judge.correctness("cdc-001", question="Q?", answer="R [1].", reference="Ref.")

    assert stub.groundedness_calls == [
        {"question": "Q?", "sources": format_sources(sources), "answer": "R [1]."}
    ]
    assert stub.correctness_calls == [{"question": "Q?", "reference": "Ref.", "answer": "R [1]."}]


def test_judge_reuses_cached_verdicts(tmp_path: Path) -> None:
    cache_path = tmp_path / "judge.jsonl"
    stub = StubJudge(cache=JudgeCache(cache_path))
    first = stub.judge.correctness("cdc-001", question="Q?", answer="R.", reference="Ref.")
    again = stub.judge.correctness("cdc-001", question="Q?", answer="R.", reference="Ref.")
    assert first == again
    assert len(stub.correctness_calls) == 1

    stub.judge.correctness("cdc-001", question="Q?", answer="Outra.", reference="Ref.")
    stub.judge.correctness("cdc-001", question="Q?", answer="R.", reference="Nova ref.")
    assert len(stub.correctness_calls) == 3

    reloaded = StubJudge(cache=JudgeCache(cache_path))
    reloaded.judge.correctness("cdc-001", question="Q?", answer="R.", reference="Ref.")
    assert reloaded.correctness_calls == []

    other_model = StubJudge(cache=JudgeCache(cache_path), model_name=GPT_OSS)
    other_model.judge.correctness("cdc-001", question="Q?", answer="R.", reference="Ref.")
    assert len(other_model.correctness_calls) == 1


def test_judge_recomputes_invalid_cached_verdicts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "judge.jsonl"
    StubJudge(cache=JudgeCache(path)).judge.correctness(
        "cdc-001", question="Q?", answer="R.", reference="Ref."
    )
    entry = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(entry | {"verdict": {"verdict": "unknown"}}) + "\n", encoding="utf-8"
    )

    stub = StubJudge(cache=JudgeCache(path))
    with caplog.at_level(logging.WARNING):
        verdict = stub.judge.correctness("cdc-001", question="Q?", answer="R.", reference="Ref.")
    assert verdict.verdict == "correct"
    assert len(stub.correctness_calls) == 1
    assert "invalid cached" in caplog.text


def test_judge_accepts_dicts_and_rejects_missing_output() -> None:
    as_dict = StubJudge(correctness=lambda _: correctness_verdict("partial").model_dump())
    verdict = as_dict.judge.correctness("cdc-001", question="Q?", answer="R.", reference="Ref.")
    assert verdict == correctness_verdict("partial")

    missing = StubJudge(groundedness=lambda _: None)
    with pytest.raises(OutputParserException, match="no structured output"):
        missing.judge.groundedness("cdc-001", question="Q?", answer="R.", sources=[])


def _groq_completion(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "openai/gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(payload)},
            }
        ],
        "usage": {"prompt_tokens": 800, "completion_tokens": 60, "total_tokens": 860},
    }


def test_judge_on_the_real_groq_request_path_uses_strict_json_schema() -> None:
    payloads = [
        groundedness_verdict("supported", "unsupported").model_dump(mode="json"),
        correctness_verdict("partial").model_dump(mode="json"),
    ]
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_groq_completion(payloads[len(requests) - 1]))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        model = ChatGroq(
            model_name="openai/gpt-oss-120b",
            groq_api_key=SecretStr("test-key"),
            temperature=0,
            max_retries=0,
            http_client=http_client,
        )
        options = structured_output_options(GPT_OSS)
        judge = build_judge(model, model_name=GPT_OSS, structured_output=options)
        sources = [make_source(1, "CDC:49")]
        groundedness = judge.groundedness(
            "cdc-001", question="Qual o prazo?", answer="7 dias [1].", sources=sources
        )
        correctness = judge.correctness(
            "cdc-001", question="Qual o prazo?", answer="7 dias [1].", reference="7 dias."
        )

    assert groundedness.score == 0.5
    assert correctness.verdict == "partial"
    response_format = requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "GroundednessVerdict"
    assert response_format["json_schema"]["strict"] is True
    claim_schema = response_format["json_schema"]["schema"]["properties"]["claims"]["items"]
    assert claim_schema["additionalProperties"] is False
    assert set(claim_schema["required"]) == {"claim", "verdict", "evidence"}
    assert requests[0]["messages"][0]["content"].startswith("Você é um avaliador")
    assert "[1] CDC, art. 49 ::" in requests[0]["messages"][1]["content"]
