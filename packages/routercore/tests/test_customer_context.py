from routercore.application.contracts import InboundMessage
from routercore.application.customer_context import SECOND_NS, register

START = 1_000 * SECOND_NS


def message(text, at_s, customer="c1"):
    return InboundMessage(message_id=f"m{at_s}", customer_id=customer, channel="chat", text=text,
                          ts_sent_ns=START + at_s * SECOND_NS)


def test_first_message_is_new_with_empty_context():
    state = {}
    duplicate, context = register(state, message("my card is lost", 0))
    assert (duplicate, context) == (False, [])


def test_same_text_within_60s_is_a_duplicate_ignoring_case_and_spacing():
    state = {}
    register(state, message("My card  is lost", 0))
    duplicate, _ = register(state, message("my card is lost", 30))
    assert duplicate is True


def test_same_text_after_60s_is_not_a_duplicate():
    state = {}
    register(state, message("my card is lost", 0))
    duplicate, _ = register(state, message("my card is lost", 90))
    assert duplicate is False


def test_context_holds_recent_messages_of_the_last_5_minutes():
    state = {}
    register(state, message("first", 0))
    register(state, message("second", 100))
    _, context = register(state, message("third", 350))
    assert context == ["second"]          # "first" is 350 s old


def test_duplicates_are_not_added_to_context():
    state = {}
    register(state, message("hello there", 0))
    register(state, message("hello there", 10))
    _, context = register(state, message("other", 20))
    assert context == ["hello there"]
