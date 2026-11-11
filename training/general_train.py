"""Training examples for the `general` route: greetings, thanks, tiny messages and out-of-domain text.

Banking77 has none, so without these Laya could never answer `general`. Disjoint from the evaluation hard
cases in hard_cases.py (a test enforces it): never train on the examples you evaluate on.
"""
GREETINGS = {
    "en": ["hey, what's up", "hello, is anyone available?", "hi, good afternoon", "good day", "greetings",
           "hi team", "howdy", "hello again", "morning!", "yo", "hi, quick question",
           "hello, can you hear me?"],
    "es": ["hola, ¿qué tal?", "hola, ¿hay alguien disponible?", "hola, buenas tardes", "buen día", "saludos",
           "hola equipo", "qué onda", "hola de nuevo", "¡buenas!", "ey", "hola, una consulta rápida",
           "hola, ¿me escuchan?"],
}
THANKS_AND_BYE = {
    "en": ["many thanks", "thanks a lot, bye", "thank you, that is all for now", "appreciate it", "cheers",
           "thanks, you're great", "ok, thank you very much", "see you", "take care", "have a great day",
           "that's all I needed", "nice, thanks", "awesome, thank you", "no problem, thanks anyway"],
    "es": ["muchísimas gracias", "mil gracias, chao", "gracias, eso es todo por ahora", "te lo agradezco",
           "gracias, eres genial", "ok, muchas gracias", "nos vemos", "cuídate", "que tengas un gran día",
           "eso era todo lo que necesitaba", "bien, gracias", "excelente, gracias",
           "no hay problema, gracias igual", "vale, gracias"],
}
TINY = {
    "en": ["uh", "hmm", "ok", "test", "okay then", "yes", "no", "maybe later", "sorry, wrong chat",
           "never mind", "forget it", "nothing", "just checking", "sorry"],
    "es": ["eh", "mmm", "ok", "prueba", "bueno pues", "sí", "no", "tal vez luego", "perdón, chat equivocado",
           "olvídalo", "déjalo así", "nada", "solo revisaba", "perdón"],
}
OUT_OF_DOMAIN = {
    "en": ["what's the best pizza topping?", "who invented the telephone?", "how many moons does Mars have?",
           "suggest a name for my cat", "what should I cook tonight?", "is it going to rain tomorrow?",
           "who is your favorite football team?", "how do I learn to swim?", "what year did the war end?",
           "can you write me a poem?", "what's the tallest building in the world?",
           "explain how airplanes fly", "recommend a book about history", "what is the speed of light?"],
    "es": ["¿cuál es el mejor ingrediente para una pizza?", "¿quién inventó el teléfono?",
           "¿cuántas lunas tiene Marte?", "sugiere un nombre para mi gato", "¿qué cocino esta noche?",
           "¿va a llover mañana?", "¿cuál es tu equipo de fútbol favorito?", "¿cómo aprendo a nadar?",
           "¿en qué año terminó la guerra?", "¿puedes escribirme un poema?",
           "¿cuál es el edificio más alto del mundo?", "explícame cómo vuelan los aviones",
           "recomiéndame un libro de historia", "¿cuál es la velocidad de la luz?"],
}


def general_examples() -> list[dict]:
    examples = []
    for group in (GREETINGS, THANKS_AND_BYE, TINY, OUT_OF_DOMAIN):
        for lang, texts in group.items():
            examples += [{"text": t, "lang": lang, "route": "general", "urgency": 0} for t in texts]
    return examples
