"""Hard cases for the escalation policy (PLAN §4.2): the messages System 1 should NOT resolve alone.

Each case: text, lang, route (the right destination), urgency (0-3), category, should_escalate.
`should_escalate` is the policy's expected behaviour: True when the model must hand the message to System 2.
Hand-written categories live here; multi-intent cases are composed from real Banking77 rows.
"""
import random

# (text, route, urgency, should_escalate)
GENERAL = {
    "en": ["hello", "hi there", "good morning", "good evening", "hey", "thanks", "thank you so much",
           "thanks for your help", "great, thanks!", "ok thanks", "perfect, that's all", "bye", "goodbye",
           "have a nice day", "you've been very helpful", "that solved it, thank you", "cool", "got it",
           "no more questions", "talk later"],
    "es": ["hola", "buenos días", "buenas tardes", "buenas noches", "qué tal", "gracias", "muchas gracias",
           "gracias por tu ayuda", "perfecto, gracias", "ok gracias", "listo, eso es todo", "chao", "adiós",
           "que tengas buen día", "fuiste de gran ayuda", "eso lo resolvió, gracias", "genial", "entendido",
           "ya no tengo más preguntas", "hablamos luego"],
}
SHORT = {
    "en": ["??", "help", "help me", "urgent", "hello?", "anyone there?", "my money", "problem", "card",
           "it doesn't work", "why", "please", "error", "not working", "???? help", "money gone"],
    "es": ["??", "ayuda", "ayúdame", "urgente", "¿hola?", "¿hay alguien?", "mi plata", "problema", "tarjeta",
           "no funciona", "por qué", "por favor", "error", "no sirve", "???? ayuda", "se fue la plata"],
}
OUT_OF_DOMAIN = {
    "en": ["who won the football match yesterday?", "what's the weather like in Madrid?",
           "give me a recipe for lasagna", "write a python function to sort a list",
           "how tall is Mount Everest?", "tell me a joke", "what is the capital of Australia?",
           "recommend a good movie for tonight", "translate 'good night' into French",
           "how do I fix my bicycle chain?", "what is the meaning of life?",
           "who is the president of Brazil?",
           "explain quantum computing to me", "what time is it in Tokyo?", "best way to learn guitar?",
           "my dog keeps barking at night, what do I do?", "can you book me a flight to Paris?",
           "what's the score of the basketball game?"],
    "es": ["¿quién ganó el partido de fútbol ayer?", "¿cómo está el clima en Madrid?",
           "dame una receta de lasaña", "escribe una función en python para ordenar una lista",
           "¿cuánto mide el Everest?", "cuéntame un chiste", "¿cuál es la capital de Australia?",
           "recomiéndame una película para esta noche", "traduce 'buenas noches' al francés",
           "¿cómo arreglo la cadena de mi bicicleta?", "¿cuál es el sentido de la vida?",
           "¿quién es el presidente de Brasil?", "explícame la computación cuántica",
           "¿qué hora es en Tokio?", "¿la mejor forma de aprender guitarra?",
           "mi perro ladra toda la noche, ¿qué hago?", "¿puedes reservarme un vuelo a París?",
           "¿cómo quedó el partido de baloncesto?"],
}
# Negations and corrections: keyword rules fire on "stolen"/"fraud" but the right answer is calmer.
NEGATION = {
    "en": [("my card was not stolen, I just want to change my PIN", "cards", 0),
           ("I'm not saying it's fraud, I only want to know the exchange rate", "fees", 0),
           ("nobody hacked my account, I just forgot my passcode", "account", 0),
           ("my card isn't lost, it just hasn't arrived yet", "cards", 0),
           ("this is not a scam, I simply want to verify my identity", "kyc", 0),
           ("I wasn't robbed, I only need a spare card", "cards", 0),
           ("no fraud here, I just want to know the top up limits", "topup", 0),
           ("I didn't lose my phone, I want to change my personal details", "account", 0),
           ("the payment wasn't unauthorized, I made it but want a refund", "disputes", 1),
           ("no, my card isn't blocked, I just want to activate it", "cards", 0)],
    "es": [("no me robaron la tarjeta, solo quiero cambiar mi PIN", "cards", 0),
           ("no digo que sea fraude, solo quiero saber el tipo de cambio", "fees", 0),
           ("nadie hackeó mi cuenta, solo olvidé mi clave", "account", 0),
           ("mi tarjeta no está perdida, solo no ha llegado", "cards", 0),
           ("esto no es una estafa, solo quiero verificar mi identidad", "kyc", 0),
           ("no me asaltaron, solo necesito una tarjeta de repuesto", "cards", 0),
           ("no hay fraude, solo quiero saber los límites de recarga", "topup", 0),
           ("no perdí el celular, quiero cambiar mis datos personales", "account", 0),
           ("el pago no fue no autorizado, lo hice yo pero quiero un reembolso", "disputes", 1),
           ("no, mi tarjeta no está bloqueada, solo quiero activarla", "cards", 0)],
}
# Sarcasm and ambiguous complaints: no clear single intent, so System 2 should look at them.
SARCASM = {
    "en": ["great, another fee, I love paying to use my own money", "wow, five days for a transfer, so fast",
           "amazing service, my card just stopped working again", "sure, charge me twice, why not",
           "thanks for nothing, the money is still missing", "fantastic, an app that never loads",
           "oh perfect, declined again at the supermarket", "love how my refund 'arrives soon' every week"],
    "es": ["genial, otra comisión, me encanta pagar por usar mi propia plata",
           "guau, cinco días para una transferencia, qué rápido",
           "excelente servicio, la tarjeta dejó de funcionar otra vez",
           "claro, cóbrenme doble, por qué no", "gracias por nada, la plata sigue sin aparecer",
           "fantástico, una app que nunca carga", "qué bien, rechazada otra vez en el supermercado",
           "me encanta que mi reembolso 'llega pronto' cada semana"],
}


def hand_written() -> list[dict]:
    cases = []

    def add(lang, text, route, urgency, category, should_escalate):
        cases.append({"text": text, "lang": lang, "route": route, "urgency": urgency, "category": category,
                      "should_escalate": should_escalate})

    for lang, texts in GENERAL.items():
        for text in texts:
            add(lang, text, "general", 0, "general", False)
    for lang, texts in SHORT.items():
        for text in texts:
            add(lang, text, "general", 0, "short", True)
    for lang, texts in OUT_OF_DOMAIN.items():
        for text in texts:
            add(lang, text, "general", 0, "out_of_domain", True)
    for lang, rows in NEGATION.items():
        for text, route, urgency in rows:
            add(lang, text, route, urgency, "negation", False)
    for lang, texts in SARCASM.items():
        for text in texts:
            add(lang, text, "general", 0, "sarcasm", True)
    return cases


def multi_intent(rows: list[dict], route_map, n: int, seed: int) -> list[dict]:
    """Glue two English messages from different routes: the first sets the route, urgency is the max."""
    rng = random.Random(seed)
    cases = []
    while len(cases) < n:
        a, b = rng.sample(rows, 2)
        route_a, route_b = route_map.route_of(a["intent"]), route_map.route_of(b["intent"])
        if route_a == route_b:
            continue
        joiner = rng.choice(["Also, ", "And ", "Another thing: ", "On top of that, "])
        text = f"{a['text'].rstrip()} {joiner}{b['text'][0].lower()}{b['text'][1:]}"
        urgency = max(route_map.urgency_of(a["intent"]), route_map.urgency_of(b["intent"]))
        cases.append({"text": text, "lang": "en", "route": route_a, "urgency": urgency,
                      "category": "multi_intent", "should_escalate": True, "secondary_route": route_b})
    return cases


def hard_records(cases: list[dict]) -> list[dict]:
    """Give hard cases the same shape as dataset records (split='hard')."""
    return [{"id": f"hc-{i}:{c['lang']}", "origin_id": f"hc-{i}", "intent": None, "split": "hard",
             "needs_human": c["urgency"] >= 1 or c["route"] in {"security", "disputes", "kyc"},
             "needs_kb": False, **c} for i, c in enumerate(cases)]
