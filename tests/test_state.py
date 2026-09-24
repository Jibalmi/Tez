# Adapted from laya v0.3.20 @23a1752, (c) Convai Innovations, Apache-2.0; modified.
# Source: tests/test_email.py (https://github.com/NandhaKishorM/laya, licence text in LICENSES/Apache-2.0.txt).
# Changes: rewritten as pytest parametrisations against tez.state.clean_email / email_state; the email_questions cases
# (a Laya preset Tez does not have) dropped; a test that the state is decided on the cleaned body added.
"""Email cleaning (tez/state.py): the new message survives, quoted history, signatures and footers go.

A disclaimer footer must not delete the sender's request, and a sign-off word inside the body is not a signature:
dropping the request is silent and severe, leaving one boilerplate line behind is neither, so the cleaning errs towards
keeping text."""
from __future__ import annotations

import pytest

from tez import FakeBackend, Tez
from tez.state import clean_email, email_state

DISCLAIMER = "This email is confidential and intended solely for the named addressee."


@pytest.mark.parametrize("body, want", [
    # the request survives the footer
    (f"My account is locked.\n{DISCLAIMER}\nPlease unlock it.", "My account is locked. Please unlock it."),
    (f"My account is locked\n{DISCLAIMER}\nPlease unlock it.", "My account is locked Please unlock it."),
    (f"Please unlock my account\n{DISCLAIMER}", "Please unlock my account"),
    (f"RMA 5521 is still pending\n{DISCLAIMER}\nPlease advise.", "RMA 5521 is still pending Please advise."),
    # keep-ward trade-off: a capitalised continuation of a boilerplate sentence stays
    ("If you have received this message in error,\nPlease delete it.", "Please delete it."),
    # documented boundary: an all-lowercase fused request is indistinguishable from a wrapped footer
    ("my account is locked\nthis email is confidential and intended solely for the named addressee.\nplease unlock it.",
     "please unlock it."),
    (f"My account is locked. {DISCLAIMER}", "My account is locked."),
    # a pure footer is still removed
    (f"My account is locked.\n\n{DISCLAIMER}", "My account is locked."),
    ("My account is locked.\n\nThis email and any files transmitted with it are\n"
     "confidential and intended solely for the named addressee.", "My account is locked."),
    ("Please reopen ticket 4411.\n\nIf you have received this message in error, delete it.", "Please reopen ticket 4411."),
    # quoted history and signatures
    ("Thanks for the update.\nOn Mon, Sep 20, Bob wrote:\n> original text", "Thanks for the update."),
    ("Hi team,\nCan you confirm the refund?\nRegards,\nAlice", "Hi team,\nCan you confirm the refund?"),
    ("", ""),
    (None, ""),
])
def test_footers_history_and_signatures(body, want):
    assert clean_email(body) == want


PT_EMAIL = """Olá equipe,

Fomos cobrados duas vezes na fatura de março. Por favor, estornem a cobrança duplicada hoje.

Atenciosamente,
João Silva
Financeiro - ACME Ltda

Enviado do meu iPhone

Esta mensagem pode conter informações confidenciais. Se você recebeu esta mensagem por engano, favor apagá-la.

Em seg., 22 de set. de 2026 às 10:14, Suporte <suporte@x.com> escreveu:
> Olá João, recebemos seu chamado de cancelamento do plano Enterprise.
"""


@pytest.mark.parametrize("body, want", [
    (PT_EMAIL, "Olá equipe,\n\nFomos cobrados duas vezes na fatura de março. Por favor, estornem a cobrança duplicada hoje."),
    ("Segue o comprovante do pagamento.\n\n-----Mensagem original-----\nDe: Maria <maria@acme.com>\n"
     "Assunto: cancelar contrato\nQueremos cancelar o contrato.", "Segue o comprovante do pagamento."),
    ("Segue o comprovante.\n\nDe: Maria <maria@acme.com>\nEnviado: segunda-feira\nAssunto: cancelar contrato\n"
     "Queremos cancelar o contrato.", "Segue o comprovante."),
    ("O acesso voltou, obrigado.\n\nEm seg., 22 de set. de 2026 às 10:14, Suporte Técnico <\n"
     "suporte@acme.com> escreveu:\n> texto antigo", "O acesso voltou, obrigado."),
    ("Bom dia,\nO boleto de março não chegou.\nObrigado,\nAna", "Bom dia,\nO boleto de março não chegou."),
    ("Hola,\nNo puedo acceder a mi cuenta desde ayer.\nSaludos,\nCarlos\n\n"
     "El lun, 22 sept 2026 a las 10:14, Soporte <soporte@x.com> escribió:\n> texto anterior",
     "Hola,\nNo puedo acceder a mi cuenta desde ayer."),
    ("Necesito la factura de marzo.\n\nSi usted ha recibido este mensaje por error, bórrelo.", "Necesito la factura de marzo."),
    # ... without eating the request
    ("Preciso do contrato confidencial assinado até sexta.", "Preciso do contrato confidencial assinado até sexta."),
    ("Oi,\nRecebi a resposta.\nObrigado pelo retorno, mas continua\nsem funcionar.",
     "Oi,\nRecebi a resposta.\nObrigado pelo retorno, mas continua\nsem funcionar."),
    ("Favor reenviar a nota fiscal.\n\nCaso tenha recebido esta mensagem por engano, notifique o remetente.",
     "Favor reenviar a nota fiscal."),
    ("Resolvido, pode fechar.\n\nEm qua., 24 de set. de 2026 às 09:02, Suporte\nTécnico <suporte@acme.com> escreveu:\n"
     "> texto antigo", "Resolvido, pode fechar."),
    ("Fixed, thanks.\n\nOn Wed, Sep 24, 2026 at 9:02 AM Support Team <\nsupport@acme.com> wrote:\n> old text", "Fixed, thanks."),
    ("Oi,\nEm resposta ao que você escreveu:\no pedido 4411 ainda não chegou.",
     "Oi,\nEm resposta ao que você escreveu:\no pedido 4411 ainda não chegou."),
    ("O valor é destinado exclusivamente ao pagamento do boleto. Podem confirmar?",
     "O valor é destinado exclusivamente ao pagamento do boleto. Podem confirmar?"),
    ("Preciso das férias.\nDe: 10/09 a 15/09\nPode aprovar?", "Preciso das férias.\nDe: 10/09 a 15/09\nPode aprovar?"),
])
def test_portuguese_and_spanish_mail(body, want):
    assert clean_email(body) == want


BOLETO = "Preciso da segunda via do boleto."


@pytest.mark.parametrize("footer", [
    "Enviado do meu Galaxy",
    "Enviado do meu smartphone Samsung Galaxy.",
    "Enviado do Outlook para iOS",
    "Obter o Outlook para Android",
    "Enviado do Email para Windows",
    "Enviado do Yahoo Mail no Android",
    "Antes de imprimir, pense em sua responsabilidade e compromisso com o MEIO AMBIENTE.",
    "Pense no meio ambiente antes de imprimir este e-mail.",
])
def test_brazilian_footers_are_removed(footer):
    assert clean_email(BOLETO + "\n\n" + footer) == BOLETO


def test_outlook_lines():
    assert clean_email("Please resend the invoice.\n\nGet Outlook for iOS") == "Please resend the invoice."
    for second in ("Enviado: sexta-feira, 19 de setembro de 2026 10:02", "Data: sexta-feira, 19 de setembro de 2026 10:02"):
        body = ("Segue o comprovante.\n\nDe: Maria Souza\n%s\nPara: Suporte\nAssunto: cancelar contrato\n\n"
                "Queremos cancelar o contrato." % second)
        assert clean_email(body) == "Segue o comprovante."


@pytest.mark.parametrize("body", [
    "Oi,\nSegue o pedido.\nEnviado do meu celular o comprovante ontem.",
    "Preciso das férias.\nDe: 10/09\nPara: 15/09\nPode aprovar?",
    "Relatório do evento.\nDe: João\nData: amanhã cedo\nPode confirmar?",
    "Antes de imprimir o boleto, confira o valor. Está errado.",
    "Hi,\nThe box is at the front desk.\nGet mail",
    # a sign-off word inside the body is not a sign-off
    "Hi,\n\nThanks for the quick reply.\nCould you refund invoice 4411 as well?",
    "Hello,\n\nWe were billed twice in March.\nThanks for looking into it.\nThe duplicate is 49 EUR on invoice 4411.",
    "Hi team,\n\nOur account is locked.\nBest practice would be a manual unlock.\nPlease unlock account 88213 today.",
    "Hi,\n\nPlease refund 4411.\nThanks and the team will confirm it today.",
    "Hi,\n\nThe room is cold.\nWarmest setting still reads 18 degrees.",
    # the word, without the disclaimer
    "Is this confidential?",
    "What is your confidentiality policy?",
    "Please keep this confidential but process my refund.",
    "Please treat this as confidential.",
    "This is confidential - can you help?",
    "Is the attached document confidential?",
    "Confidential: I need a refund.",
    "What is the information policy for contractors?",
])
def test_requests_are_kept(body):
    assert clean_email(body) == body


BODY = "Hi,\n\nPlease refund invoice 4411."


@pytest.mark.parametrize("tail", [
    "Thanks,\nAnna", "Thanks!", "Best regards,\nAnna", "Kind regards", "Cheers, Anna", "Many thanks,\nAnna Meier",
    "Thank you,", "Thanks in advance,", "Sincerely,\nA. Meier", "Sent from my iPhone", "--\nAnna Meier\nSupport",
    "Thanks and regards,\nAnna", "Thanks & Regards,\nAnna", "Warmest regards,\nAnna", "Warmest wishes,",
    "Regards, Łukasz", "Thanks, José", "Regards, Дмитрий",
])
def test_real_sign_offs_are_cut(tail):
    assert clean_email(f"{BODY}\n\n{tail}") == BODY


@pytest.mark.parametrize("body", [
    DISCLAIMER,
    "This message is confidential and intended solely for the use of the individual to whom it is addressed.",
    "The information in this email is confidential and may be privileged.",
    "This email and any files transmitted with it are\nconfidential and intended solely for the named addressee.",
])
def test_real_footers_are_dropped(body):
    assert not clean_email(body).strip()


def test_mixed_footer_and_request():
    assert clean_email(f"My account is locked.\n{DISCLAIMER}\nPlease unlock it.") == "My account is locked. Please unlock it."
    assert clean_email(f"Please unlock it. {DISCLAIMER}") == "Please unlock it."


def test_max_chars_and_line_endings():
    assert clean_email("a" * 5000, max_chars=100) == "a" * 100
    assert clean_email("Line one\r\nLine two\\nLine three") == "Line one\nLine two\nLine three"


def test_email_state():
    short = "Hi,\n\nThanks for the quick reply.\nCould you refund invoice 4411 as well?"
    assert email_state("Duplicate charge", short)["body"] == short
    assert email_state("Locked out", f"My account is locked. {DISCLAIMER}")["body"] == "My account is locked."
    assert email_state("Question", "Is this confidential?")["body"] == "Is this confidential?"
    s = email_state("  Re: invoice  ", "Please refund.\nRegards,\nAna", sender="ana@example.com", ticket=12, tag=None)
    assert s == {"subject": "Re: invoice", "body": "Please refund.", "from": "ana@example.com", "ticket": 12}
    assert email_state(None, "raw\nRegards,\nAna", clean=False) == {"subject": "", "body": "raw\nRegards,\nAna"}


def test_the_model_reads_the_cleaned_email():
    fb = FakeBackend()
    raw = "Please refund invoice 4411.\n\nOn Mon, Sep 20, Bob wrote:\n> cancel my whole account"
    Tez(backend=fb).decide(email_state("Refund", raw), questions={"q": {"type": "noul", "instructions": "Refund?"}})
    assert "Please refund invoice 4411." in fb.prompts[0] and "cancel my whole account" not in fb.prompts[0]
