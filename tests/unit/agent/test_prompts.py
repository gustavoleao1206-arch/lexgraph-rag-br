"""Prompt building: message structure, sources, history, truncation and fixed replies."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from lexgraph.agent import prompts
from lexgraph.catalog import LAWS
from lexgraph.domain import LawId
from tests.unit.agent.fakes import CDC_49, CF_7, CLT_58, make_passage


def test_law_guide_describes_every_law() -> None:
    guide = prompts.law_guide()

    for law in LAWS.values():
        assert f"- {law.law_id.value} ({law.name})" in guide


def test_answer_system_prompt_states_the_corpus_date() -> None:
    dated = prompts.answer_system_prompt("30/10/2026")
    undated = prompts.answer_system_prompt(None)

    assert "vigente em 30/10/2026" in dated
    assert "30/10/2026" not in undated
    assert "pode ter sido alterada" in undated
    assert "SOMENTE as fontes numeradas" in dated


def test_render_passages_numbers_sources_with_citation_and_location() -> None:
    rendered = prompts.render_passages([CDC_49, CLT_58])

    assert rendered.startswith(f"[1] CDC, art. 49 — {CDC_49.heading_path[0]}\n{CDC_49.text}")
    assert f"\n\n[2] CLT, art. 58 — {CLT_58.heading_path[0]}\n" in rendered


def test_render_passages_truncates_long_texts() -> None:
    long_passage = make_passage(LawId.CF, "5", "Art. 5º " + "palavra " * 400)

    rendered = prompts.render_passages([long_passage], max_chars=100)

    body = rendered.split("\n", 1)[1]
    assert body.endswith(" [...]")
    assert len(body) <= 100 + len(" [...]")


def test_truncate_cuts_at_a_word_boundary() -> None:
    assert prompts.truncate("curto", 10) == "curto"
    assert prompts.truncate("uma frase com várias palavras", 16) == "uma frase com [...]"
    assert prompts.truncate("x" * 30, 10) == "x" * 10 + " [...]"


def test_answer_messages_end_with_the_question_and_its_sources() -> None:
    history = [HumanMessage("Posso desistir?"), AIMessage("Sim, em 7 dias [1].")]

    messages = prompts.answer_messages(
        question="E se a compra foi na loja?",
        standalone_question="Posso desistir de uma compra feita na loja física?",
        passages=[CDC_49],
        history=history,
        corpus_date="30/10/2026",
    )

    assert isinstance(messages[0], SystemMessage)
    assert "30/10/2026" in messages[0].text
    assert [type(m) for m in messages[1:]] == [HumanMessage, AIMessage, HumanMessage]
    assert messages[2].text == "Sim, em 7 dias."  # old markers point to old sources
    final = messages[-1].text
    assert "<fontes>\n[1] CDC, art. 49" in final
    assert final.index("<fontes>") < final.index("<pergunta>")
    assert "Pergunta reformulada a partir do histórico: Posso desistir de uma compra" in final
    assert "revisão" not in final


def test_answer_messages_skip_the_standalone_question_when_identical() -> None:
    messages = prompts.answer_messages(
        question="Qual o prazo?", standalone_question="Qual o prazo?", passages=[], history=[]
    )

    assert len(messages) == 2
    assert "reformulada" not in messages[-1].text


def test_answer_messages_carry_the_review_feedback() -> None:
    messages = prompts.answer_messages(
        question="Qual o prazo?",
        standalone_question="Qual o prazo?",
        passages=[CDC_49],
        history=[],
        feedback=["O prazo de 30 dias não consta das fontes."],
    )

    assert "- O prazo de 30 dias não consta das fontes." in messages[-1].text


def test_history_messages_are_shortened() -> None:
    long_answer = AIMessage("palavra " * 200)

    messages = prompts.answer_messages(
        question="E agora?",
        standalone_question="E agora?",
        passages=[],
        history=[HumanMessage("Pergunta"), long_answer],
    )

    assert len(messages[2].text) <= prompts.HISTORY_MESSAGE_MAX_CHARS + len(" [...]")


def test_analyzer_messages_include_history_only_when_present() -> None:
    first = prompts.analyzer_messages("Oi, tudo bem?", [])
    follow_up = prompts.analyzer_messages(
        "E o prazo?", [HumanMessage("Posso desistir?"), AIMessage("Sim [1].")]
    )

    assert "<historico>" not in first[-1].text
    assert "<mensagem>\nOi, tudo bem?\n</mensagem>" in first[-1].text
    assert "Usuário: Posso desistir?\nAssistente: Sim." in follow_up[-1].text
    assert isinstance(follow_up[0], SystemMessage)
    assert "CLT (Consolidação das Leis do Trabalho)" in follow_up[0].text


def test_grader_messages_shorten_passages() -> None:
    long_passage = make_passage(LawId.CF, "5", "Art. 5º " + "palavra " * 400)

    messages = prompts.grader_messages("Pergunta?", [long_passage, *CF_7])

    text = messages[-1].text
    assert "[1] CF, art. 5" in text
    assert "[4] CF, art. 7" in text
    assert len(text) < 4 * (prompts.GRADER_PASSAGE_MAX_CHARS + 200)


def test_grader_sees_the_units_of_a_slice_rather_than_its_repeated_caput() -> None:
    caput = "Art. 5º Todos são iguais perante a lei, " + "sem distinção de qualquer natureza, " * 12
    filler = "\n".join(f"XXI{'I' * n} - inciso sobre outro assunto qualquer;" for n in range(6))
    units = f"{filler}\nXXXII - o Estado promoverá, na forma da lei, a defesa do consumidor;"
    first = make_passage(LawId.CF, "5", f"{caput}\nI - homens e mulheres são iguais", n_chunks=3)
    later = make_passage(LawId.CF, "5", f"{caput}\n{units}", chunk_index=2, n_chunks=3)
    assert len(later.text) > prompts.GRADER_PASSAGE_MAX_CHARS
    assert "XXXII" not in prompts.truncate(later.text, prompts.GRADER_PASSAGE_MAX_CHARS)

    text = prompts.grader_messages("Quem defende o consumidor?", [first, later])[-1].text

    assert "XXXII - o Estado promoverá" in text
    assert caput.strip() in text  # the first slice keeps its caput
    assert text.count("Art. 5º Todos são iguais perante a lei") == 2
    assert prompts.shorten_repeated_caput(first) is first


def test_rewriter_messages_describe_the_failed_search() -> None:
    searched = prompts.rewriter_messages("Pergunta?", "consulta antiga", ["CLT"], [CLT_58])
    unfiltered = prompts.rewriter_messages("Pergunta?", "consulta antiga", [], [])

    assert "Consulta anterior: consulta antiga" in searched[-1].text
    assert "Normas pesquisadas: CLT" in searched[-1].text
    assert "descartados: CLT, art. 58" in searched[-1].text
    assert "Normas pesquisadas: todas" in unfiltered[-1].text
    assert "descartados: nenhum" in unfiltered[-1].text


def test_verifier_messages_show_sources_then_answer() -> None:
    messages = prompts.verifier_messages("O prazo é de 7 dias [1].", [CDC_49])

    text = messages[-1].text
    assert text.index("<fontes>") < text.index("<resposta>")
    assert "O prazo é de 7 dias [1]." in text


def test_templated_replies_list_the_corpus() -> None:
    for route in ("smalltalk", "out_of_scope", "no_context"):
        reply = prompts.templated_reply(route)
        assert "CLT (Consolidação das Leis do Trabalho)" in reply
        assert "{" not in reply


def test_no_context_reply_mentions_the_user_filter() -> None:
    reply = prompts.templated_reply("no_context", user_laws=["CLT", "MCI"])

    assert "a busca ficou restrita a: CLT, Marco Civil." in reply
    assert "restrita" not in prompts.templated_reply("no_context")


def test_sources_show_constitutional_review_notes_and_the_answer_rule_explains_them() -> None:
    passage = make_passage().model_copy(update={"control_notes": ["Vide ADC 58"]})

    rendered = prompts.render_passages([passage])

    assert f"{prompts.CONTROL_NOTES_LABEL}: Vide ADC 58" in rendered
    assert prompts.CONTROL_NOTES_LABEL in prompts.answer_system_prompt("30/10/2026")
