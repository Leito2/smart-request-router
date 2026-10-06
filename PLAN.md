# 🧭 P2 — Intelligent Request Router (System 1 / System 2)

> **Plan de proyecto** (vive en este repo como `PLAN.md`). **Estado:** ✅ Plan completo (11/11 módulos), listo para implementar en la fase F5 del plan de Learning.
> Stack: Laya (decision model, CPU) · Quix Streams · Kafka/Redpanda · LangGraph (agente System 2) · LLM local vía LLM Gateway (`llm-gateway`, Python) · Hugging Face + PyTorch (fine-tune) · Redis · MLflow · Langfuse · FastAPI · Docker Compose
> **v2 (evaluación industrial, §12):** `judgekit` (asyncio · aiohttp · Pandas · HF Evaluate/Tokenizers) · Gemma 4 31B *golden evaluator* · Langfuse (datasets, experiments, prompt management) · SageMaker Processing (modo local) + MinIO/S3 · KFP v2 (`kfp.local` → Vertex AI Pipelines) · Evidently · Presidio
> Gasto: **$0** (todo local; Langfuse Cloud free opcional).

## Módulos del plan
| # | Sección | Estado |
|---|---|---|
| 1 | Visión, problema y frase del CV | ✅ |
| 2 | Arquitectura macro y flujo de datos | ✅ |
| 3 | Componentes (uno por pieza, de lo conceptual a lo técnico) | ✅ |
| 4 | Modelo: dataset, fine-tune de Laya, calibración, umbrales System 1 → System 2 | ✅ |
| 5 | Harness de evaluación (set dorado, LLM-as-a-Judge, ECE, drift) | ✅ |
| 6 | Observabilidad | ✅ |
| 7 | Presupuesto de recursos (8 GB / `⏳ 16GB`) | ✅ |
| 8 | Ejecución local y alternativas de despliegue | ✅ |
| 9 | Estructura del repo y del README | ✅ |
| 10 | Hitos de implementación y criterios de aceptación | ✅ |
| 11 | Riesgos y pendientes | ✅ |
| 12 | **v2 — Evaluación industrial: `judgekit`** (absorbe la *LLM Evaluation Suite* del CV) | ✅ v2 |

## Regla del README
README progresivo: **contexto teórico, conceptual y macro primero**; en cada componente, el detalle técnico al final. Incluye cómo funciona, los pasos para ejecutarlo y las alternativas de ejecución o despliegue (local primero).

---

## 1. Visión, problema y frase del CV

### 1.1 El problema (contexto de negocio)
Una fintech recibe **miles de mensajes por hora** por chat, email y la app: "no me llegó la transferencia", "¿cómo subo mi límite?", "me cobraron dos veces", "creo que me robaron la tarjeta". Cada mensaje debe llegar al **lugar correcto**:

| Destino | Ejemplo |
|---|---|
| Respuesta automática con la base de conocimiento (**→ P3, RAG**) | "¿Cuánto cuesta una transferencia internacional?" |
| Consulta de una decisión de fraude (**→ P1**) | "Me bloquearon un pago, ¿por qué?" |
| Cola humana del equipo correcto, con prioridad | "Me cobraron dos veces" → Disputas, prioridad alta |
| Escalamiento urgente | "Me robaron la tarjeta" → Seguridad, **inmediato** |
| Respuesta de plantilla | "¿Cuál es el horario de atención?" |

Hoy hay dos formas típicas de hacerlo, y ambas fallan:
- **Reglas y palabras clave:** baratas y rápidas, pero frágiles. "No me robaron, solo perdí la tarjeta" contiene "robaron".
- **Un LLM para todo:** entiende bien, pero cuesta tokens en **cada** mensaje, tarda de cientos de ms a segundos y su salida es texto que hay que parsear, sin una probabilidad confiable.

### 1.2 La tesis: System 1 / System 2
Así como en el pensamiento humano (Kahneman), **la mayoría de las decisiones son rápidas e intuitivas, y solo las difíciles requieren razonar**:
- **System 1 · Laya (decision model):** evalúa **en una sola pasada y en CPU** varias decisiones a la vez (intención, urgencia, ¿necesita a un humano?, ¿necesita RAG?) y devuelve **probabilidades calibradas**. Resuelve la mayoría de los mensajes en milisegundos y a costo casi cero.
- **System 2 · Agente LangGraph:** solo cuando Laya **no está segura** (confianza baja, mensaje con varias intenciones, contexto raro). El agente razona, consulta herramientas (el historial del cliente, la API de P1, la base de P3) y decide.

La pregunta que responde el proyecto con datos es: **¿cuánto costo y latencia se ahorran sin perder calidad frente a un router que usa solo un LLM?**

### 1.3 Por qué streaming (Quix Streams)
Un router que clasifica mensaje por mensaje no ve el **patrón**. El streaming agrega visión de conjunto:
- **Detección de oleadas (*surge*):** si los mensajes de "tarjeta rechazada" se multiplican por 5 en 10 minutos, probablemente hay una **caída del procesador de pagos o una ola de fraude**. Se dispara una alerta de incidente antes de que lo note un humano. Es el puente natural con P1.
- **Contexto por cliente:** si un cliente envía 4 mensajes en 3 minutos, se agrupan en una sola conversación y la prioridad sube por frustración.
- **Deduplicación y anti-spam** por ventana de tiempo.

Quix Streams permite hacerlo en **Python nativo**: el mismo proceso que corre Laya mantiene las ventanas (estado en RocksDB respaldado por changelog topics), sin JVM. Es el tercer motor del curso comparativo (C2). Se eligió sobre Bytewax porque Bytewax no publica releases desde nov-2024 (v0.21.1), mientras Quix Streams está activo (v3.27.0, sep-2026) y ofrece exactly-once opcional.

### 1.4 Qué construimos
1. **Ingesta:** los mensajes (simulados, a partir de un dataset real) entran por Kafka/Redpanda y por una API FastAPI.
2. **Dataflow en Quix Streams:** normaliza, deduplica, agrupa por cliente, llama a Laya y detecta oleadas por ventana.
3. **Laya (System 1):** una pasada multi-pregunta, con umbrales de confianza calibrados.
4. **Agente LangGraph (System 2):** para los casos de baja confianza, con herramientas y un LLM local vía el **LLM Gateway** propio (`llm-gateway`).
5. **Destinos:** colas por equipo, integración con P3 (RAG) y P1 (decisiones de fraude).
6. **Harness de evaluación:** set dorado, LLM-as-a-Judge local, métricas de calibración y drift, todo en MLflow.
7. **Comparativa de routers:** reglas vs SetFit vs Laya base vs Laya fine-tuneada vs LLM-only.
8. **Observabilidad:** Langfuse (trazas del agente) + Prometheus/Grafana.

### 1.5 Datos
- **Base: Banking77**, un dataset público de 13k consultas bancarias en inglés etiquetadas en 77 intenciones. Es justo el dominio fintech. Las 77 intenciones se **mapean** a nuestros destinos y prioridades (verificar la licencia al implementar).
- **Español:** con traducción y paráfrasis generadas **localmente** (LLM vía gateway, $0), más un set de prueba revisado a mano. Laya multilingüe (322M) cubre ambos idiomas.
- **Casos difíciles agregados:** varias intenciones en un mensaje, negaciones ("no me robaron…"), sarcasmo, mensajes muy cortos ("??", "ayuda") y fuera de dominio. Son los que deberían escalar a System 2.

### 1.6 Qué demuestra (para quién)
| Audiencia | Lo que ve |
|---|---|
| Recruiter | Una tecnología de sept. 2026 (decision models) aplicada con números de costo y latencia |
| Entrevistador técnico | Calibración, umbrales de escalamiento, evaluación rigurosa, el trade-off costo/calidad medido y el patrón agéntico usado **solo donde aporta** |
| Tú | Fine-tuning de modelos encoder, evaluación de LLMs y agentes, streaming en Python, arquitectura de costo eficiente |

### 1.7 Métricas de éxito
| Tipo | Métrica | Meta inicial |
|---|---|---|
| Calidad | Macro-F1 del destino final (System 1 + 2) en el test set | ≥ LLM-only, o a ≤ 2 puntos |
| Calidad crítica | **Recall de urgentes** (robo, fraude, seguridad) | ≥ 98%: un urgente mal enrutado es el peor error |
| Calibración | ECE de Laya antes y después del temperature scaling | < 0.05 después |
| Cobertura | % resuelto por System 1 | Se reporta la curva cobertura vs precisión según el umbral |
| Latencia | p50/p95 de System 1 en CPU; p95 de System 2 | System 1: p95 < 300 ms en el i5-10300H (a validar) |
| Costo | **% de llamadas al LLM evitadas** y costo estimado vs LLM-only | Tokens contados y valorados con un precio de referencia (p. ej., Haiku) **sin gastar** |
| Streaming | Tiempo de detección de una oleada simulada | < 2 min desde que empieza |
| Agente | Acierto de System 2 en los casos escalados (evaluado por set dorado + juez) | Mayor que el de Laya en esos mismos casos (si no, el agente sobra) |

### 1.8 Alcance
**Dentro:** todo lo anterior, local con Docker Compose, $0. Langfuse Cloud free es opcional.
**Fuera:** canales reales (WhatsApp, email), una UI de agentes humanos (las colas se ven en Grafana y en la API), respuestas automáticas largas (eso lo hace P3) y Jev real por API (no corre local; queda solo en la comparativa del curso C7).

### 1.9 Frase del CV (plantilla)
> **Intelligent Request Router (System 1 / System 2)** · Laya · Quix Streams · LangGraph · Hugging Face · MLflow · Langfuse
> Decision-model router resolving **{X}% of fintech support requests on CPU at p95 {Y} ms**, escalating low-confidence cases to a LangGraph agent; macro-F1 **{Z}** (vs {Z'} LLM-only) with **{C}% fewer LLM calls**, calibrated probabilities (ECE {E}), and real-time surge detection over Quix Streams windows.

Versión corta: *"System 1/2 router: {X}% resolved by a CPU decision model, {C}% fewer LLM calls, same accuracy as LLM-only."*

### 1.10 Narrativa para entrevista (30 segundos)
"Construí un router de mensajes de soporte con un patrón System 1 / System 2. Un decision model, Laya, clasifica en una sola pasada en CPU varias decisiones con probabilidades calibradas. Solo cuando la confianza es baja escala a un agente LangGraph. Comparé contra un router que usa solo un LLM: con {C}% menos llamadas al LLM logré una calidad de {Z} frente a {Z'}, y nunca bajé el recall de urgentes del 98%. Además, las ventanas de Quix Streams detectan oleadas de quejas en menos de 2 minutos, lo que puede anticipar caídas o fraudes masivos."

### 1.11 Aporte al README
Secciones **Overview**, **The Problem**, **The System 1 / System 2 Thesis**, **What This Demonstrates** y **Results** (con placeholders hasta medir).

---

## 2. Arquitectura macro y flujo de datos

### 2.1 Vista general
```
 Canales simulados            ┌──────────────────────── QUIX STREAMS DATAFLOW (System 1) ──────────────────┐
┌──────────────┐ inbound-     │ parse/validate → dedupe → contexto por cliente → micro-lote → LAYA (ONNX)   │
│ Replayer     │ messages     │      │                                                    │                │
│ (dataset)    │─────────────►│      └─► DLQ                      política de decisión ◄──┘                │
└──────────────┘  key=cliente │                                   │ confianza ≥ τ       │ confianza < τ   │
┌──────────────┐              │ ventanas por intención ──► surge  │                     │ o multi-intent  │
│ FastAPI      │─────────────►│ (tumbling 1 min + EWMA)  detector │                     │ o urgente dudoso│
│ POST /messages              └──────────────────────────│────────│─────────────────────│─────────────────┘
│ POST /route (sync, solo S1)                             ▼        ▼                     ▼
└──────────────┘                                     alerts    routed              escalations
                                                                  ▲                     │
                                         ┌────────────────────────┘                     ▼
                                         │                          ┌──────────────────────────────┐
                                         │                          │ System 2 Worker (LangGraph)  │
                                         │◄──── routed (S2) ────────│ tools: historial (Redis),    │
                                         │                          │ P1 /decisions, P3 /ask (KB)  │
                                         ▼                          │ LLM local vía llm-gateway    │
                                ┌─────────────────┐                 └──────────────────────────────┘
                                │   Dispatcher    │──► Redis Streams: queue:disputes, queue:security, …
                                │                 │──► P3 /ask (RAG)   ──► P1 /decisions/{id}
                                └────────┬────────┘
                                         ▼
                              PostgreSQL: decision log + feedback humano ──► dataset de reentrenamiento
   Observabilidad: Langfuse (agente/juez) · Prometheus + Grafana (latencia, cobertura, oleadas) · MLflow (evals)
```

### 2.2 Decisiones de diseño (ADRs resumidos)
**ADR-1 · El System 2 vive fuera del dataflow.** Un agente tarda segundos. Si corriera dentro de Quix Streams, **bloquearía** el flujo de todos los demás mensajes (head-of-line blocking). El dataflow solo **publica** en `escalations`, y un worker aparte, con su propia concurrencia, lo resuelve.

**ADR-2 · Contexto sin esperar.** Una *session window* por cliente obligaría a esperar a que la sesión "cierre" antes de enrutar. En su lugar, cada mensaje se enruta **de inmediato**, adjuntando el estado de los mensajes recientes del cliente (5 min) que mantiene un operador con estado. Las sesiones se usan solo para analítica, nunca para frenar el enrutamiento. Es el mismo principio que las ventanas `OVER` de P1.

**ADR-3 · Micro-lotes hacia Laya.** Un encoder rinde mucho más por mensaje en lote. El dataflow acumula hasta 16 mensajes o 20 ms y clasifica en una sola llamada. Es un trade-off latencia/throughput que se mide.

**ADR-4 · Costo asimétrico de errores.** Mandar un robo de tarjeta a "FAQ" es mucho peor que mandar una FAQ a un humano. Por eso hay una **red de seguridad de urgencia**: si P(urgente) supera un umbral **bajo** (p. ej., 0.15), se escala aunque la intención principal tenga confianza alta.

**ADR-5 · Una pasada, varias preguntas.** Laya responde en **la misma pasada** cuatro preguntas: `intent` (las 77 de Banking77 mapeadas a ~12 rutas), `urgency` (low/medium/high/critical), `needs_human` (sí/no) y `needs_kb` (sí/no). Es la ventaja distintiva de los decision models frente a un clasificador clásico, que necesitaría cuatro modelos o cuatro cabezas.

**ADR-6 · Redpanda en lugar de Kafka.** Es compatible con la API de Kafka, no tiene JVM y usa ~500 MB de RAM. P1 ya demuestra Kafka; P2 demuestra la alternativa ligera, y el código no cambia.

**ADR-7 · Colas con Redis Streams.** Los destinos humanos son *consumer groups* de Redis Streams (`queue:<equipo>`), con acks y reintentos. Es más ligero que otro topic de Kafka por equipo y conecta con el curso C3.

### 2.3 Topics y colas
| Nombre | Tipo | Clave | Contenido |
|---|---|---|---|
| `inbound-messages` | Topic (6 particiones) | `customer_id` | Mensaje crudo + canal + timestamps |
| `routed` | Topic (6) | `message_id` | Decisión final (`decided_by: system1 \| system2 \| safety_net`) |
| `escalations` | Topic (3) | `message_id` | Mensaje + probabilidades de Laya + razón del escalamiento |
| `alerts` | Topic (1) | `intent` | Oleadas detectadas |
| `dlq` | Topic (1) | — | Inválidos |
| `queue:<equipo>` | Redis Stream | — | Trabajo para humanos (disputes, security, cards, transfers, kyc, general) |

### 2.4 Contratos (v1)
```jsonc
// inbound-messages
{ "message_id": "uuid", "customer_id": "c_0042", "channel": "chat", "lang": "es?",
  "text": "me cobraron dos veces la suscripción", "ts_sent_ns": 0 }

// routed
{ "message_id": "…", "route": "disputes", "priority": "high",
  "decided_by": "system1", "confidence": 0.93,
  "laya": { "intent": {"duplicate_charge": 0.93, "…": 0.02},
            "urgency": {"high": 0.71, "…": 0.1}, "needs_human": {"yes": 0.88},
            "needs_kb": {"no": 0.9} },
  "model_version": "laya-ft-3", "threshold_set": "v2", "ts_routed_ns": 0 }

// escalations
{ …inbound, "laya": {…}, "reason": "low_confidence | multi_intent | urgency_safety_net | ood" }
```

### 2.5 El viaje de un mensaje
1. Entra por el Replayer (a una tasa controlada) o por `POST /messages`.
2. Quix Streams lo valida, lo deduplica (mismo cliente y texto en 60 s) y le adjunta el contexto reciente del cliente.
3. Se acumula en un micro-lote y Laya responde las 4 preguntas.
4. La **política de decisión** compara cada distribución con sus umbrales calibrados y aplica la red de seguridad de urgencia.
5. Si hay confianza, va a `routed` (System 1). Si no, va a `escalations` con la razón.
6. El System 2 Worker corre el agente, que razona, usa herramientas y devuelve una decisión estructurada a `routed`.
7. El Dispatcher la ejecuta: encola en Redis Streams, llama a P3 o consulta P1. Todo queda en Postgres.
8. En paralelo, las ventanas por intención alimentan el detector de oleadas y emiten en `alerts`.

### 2.6 Aporte al README
**Architecture**, **Design Decisions**, **Data Contracts** y **The Journey of a Message**.

---

## 3. Componentes

> Patrón de cada componente: **🧠 Concepto → ⚙️ Cómo funciona aquí → 🔧 Detalle técnico → 🔁 Alternativas.**

### 3.1 Replayer (`services/replayer/`)
**🧠** Reproduce mensajes reales (del dataset) a una tasa controlada, con escenarios que imitan la vida real.
**⚙️** Lee el test set y el set de casos difíciles, asigna clientes y emite en lazo abierto (la misma técnica que en P1, para medir sin sesgo). Tiene **escenarios de oleada**: `--surge card_declined x5 at 120s`.
**🔧** `confluent-kafka` contra Redpanda; `--rate`, `--duration`, `--lang-mix es:0.6,en:0.4`, `--surge`.
**🔁** k6 por HTTP contra `POST /messages`.

### 3.2 Redpanda (`infra/redpanda`)
**🧠** Un log de eventos con la API de Kafka, escrito en C++ y sin JVM.
**⚙️** Un nodo; un contenedor `init` crea los topics de §2.3.
**🔧** `--smp 1 --memory 512M --overprovisioned` (el modo de desarrollo para laptops).
**🔁** Kafka KRaft (como en P1, mismo código).

### 3.3 Dataflow de Quix Streams (`services/dataflow/`)
**🧠** Procesamiento de streams **en Python nativo**, con operadores con estado y ventanas. Corre la misma librería que el modelo, sin JVM.
**⚙️** Operadores en orden: `KafkaSource` → `parse` (inválidos a la DLQ) → `dedupe` (estado por cliente, TTL de 60 s) → `attach_context` (últimos mensajes en 5 min) → `collect` (micro-lote) → `classify` (Laya) → `decide` (política) → bifurcación hacia `routed` o `escalations`. En paralelo, `count_window` por intención (tumbling de 1 min) → `surge_detect`.
**🔧** Quix Streams (`Application` + `StreamingDataFrame`): estado por cliente en RocksDB local respaldado por changelog topics en Redpanda; ventanas `tumbling_window`/`sliding_window` con `.current()` para emitir por evento. Uno o dos procesos en el mismo consumer group según la CPU (≤ particiones). El modelo se carga **una vez por proceso**. Métricas de Prometheus expuestas desde el proceso.
**🔁** Quix Streams o Faust (Python). Flink (P1) o Spark (P3). Se comparan en el curso C2.

### 3.4 Laya — System 1 (`packages/routercore/laya.py`)
**🧠** Un **decision model**: recibe un estado (el texto y su contexto) y un conjunto de **preguntas con opciones cerradas**, y devuelve una **distribución de probabilidad por pregunta** en una sola pasada, sin generar texto. Está basado en encoders ModernBERT/mmBERT (322M multilingüe), así que es rápido en CPU.
**⚙️** Se usa la variante **multilingüe (322M)** fine-tuneada con nuestras rutas (§4). En inferencia se exporta a **ONNX con cuantización int8 dinámica** para la CPU, y se mide la pérdida de calidad.
**🔧** Hugging Face + Optimum para el export. ONNX Runtime con `intra_op_num_threads` ajustado al i5 (4C/8T). Longitud máxima de 128 tokens (los mensajes de soporte son cortos).
**🔁** Von (395M, OpenVINO en CPU), Kev 0.8B (GPU, API compatible con Jev), SetFit y ModernBERT fine-tuneado como clasificador clásico. Están en la comparativa del curso C7, y algunas en los baselines de §4.

### 3.5 Política de decisión (`packages/routercore/policy.py`)
**🧠** Convierte probabilidades en acciones. Es **clasificación selectiva**: el modelo puede **abstenerse** cuando duda, y eso es una ventaja, no un fallo.
**⚙️** Reglas en orden:
1. Red de seguridad de urgencia (ADR-4).
2. Si `max P(intent) < τ_intent` → `low_confidence`.
3. Si las dos intenciones más probables están cerca (margen < m) → `multi_intent`.
4. Si se detecta un mensaje fuera de dominio (entropía alta) → `ood`.
5. Si no se cumple nada de lo anterior, la decisión es de System 1.

**🔧** Los umbrales (`τ`, `m`) se eligen en validación con la **curva riesgo-cobertura** (§4.5) y se guardan en `policy.yaml` (versionado).
**🔁** Un solo umbral global (más simple y peor). Conformal prediction (garantías estadísticas de cobertura); queda como mejora documentada.

### 3.6 System 2 Worker — agente LangGraph (`services/system2/`)
**🧠** Un agente **acotado**, no una conversación libre. Su trabajo es tomar **una** decisión de enrutamiento mejor que la de Laya en los casos difíciles, usando información que Laya no tiene.
**⚙️** Grafo:
```
analyze ──► (¿necesita datos?) ──► tools ──► decide ──► validate ──► emit
   ▲                                  │                   │ (inválido)
   └──────────────────────────────────┘◄──────────────────┘   máx. 2 reintentos → fallback: queue:triage
```
- **Herramientas:** `get_customer_history` (Redis), `lookup_payment_decision` (API de P1; mock si P1 no corre), `search_kb` (API de P3; mock si no corre) y `get_laya_scores` (las probabilidades ya calculadas).
- **Salida estructurada** (Pydantic): `route`, `priority`, `rationale` y `evidence`.

**🔧** LLM local (Qwen3 1.7B en Q4 vía Ollama) **a través del LLM Gateway (`llm-gateway`)**. Concurrencia de 2, timeout de 20 s por caso, máximo 6 pasos. Cada ejecución queda trazada en **Langfuse**. Si algo falla, va a `queue:triage` (humano), nunca se pierde.
**🔁** Un LLM sin agente (una sola llamada con salida estructurada). Es el baseline "S2 sin herramientas" y se mide si las herramientas aportan.

### 3.7 Dispatcher (`services/dispatcher/`)
**🧠** Separa **decidir** de **ejecutar**. El router dice "a dónde"; el dispatcher lo hace.
**⚙️** Consume `routed` y, según la ruta: hace `XADD` al stream del equipo, llama a `P3 /ask` (y guarda la respuesta) o a `P1 /decisions/{id}`, o responde con una plantilla. Registra todo en Postgres.
**🔧** Clientes HTTP con timeout y circuit breaker. Los **contratos de P1 y P3 están mockeados** (servidores falsos en Compose), así P2 funciona solo.
**🔁** Que los equipos consuman directamente de `routed` filtrando (más acoplado).

### 3.8 API (`services/api/`)
**🧠** La puerta HTTP para demos, integraciones y feedback humano.
**⚙️** Endpoints:
- `POST /messages`: encola en el stream (asíncrono).
- `POST /route`: solo System 1, síncrono; devuelve las probabilidades. Útil para demos y para P3.
- `GET /messages/{id}`: devuelve la decisión, quién decidió y el rationale.
- `POST /feedback`: un humano corrige la ruta. Alimenta el reentrenamiento (§4.7).
- `GET /queues`: tamaño de cada cola.

**🔧** FastAPI con el mismo `routercore` que el dataflow (una sola política, sin skew).
**🔁** gRPC para `POST /route`.

### 3.9 Detector de oleadas (operador en el dataflow)
**🧠** Detectar **anomalías de volumen** por intención, no por mensaje.
**⚙️** Cuenta por intención en ventanas de 1 min. Compara con una línea base **EWMA** y su desviación, y si el z-score supera 4 durante 2 ventanas seguidas, emite en `alerts` con contexto (intención, volumen, ejemplos).
**🔧** El estado de la EWMA vive en el operador, con un período de calentamiento. Los parámetros están en `surge.yaml`.
**🔁** ADWIN o Page-Hinkley (curso `09/40/04`), o Prophet (demasiado pesado aquí).

### 3.10 Redis (`infra/redis`)
**⚙️** Tres usos: (1) **caché de decisiones** por hash del texto normalizado, para las plantillas repetidas ("hola", "gracias"), con TTL de 1 h; (2) **historial de cliente** para el agente; (3) **colas** con Redis Streams.
**🔧** `maxmemory 128mb`, `allkeys-lru` **solo** para la caché (en un DB lógico aparte de las colas, que no se pueden expulsar).

### 3.11 PostgreSQL (`infra/postgres`)
**⚙️** Tablas `messages`, `decisions`, `escalations`, `feedback` y `alerts`. Vistas: `v_coverage` (System 1 vs System 2), `v_accuracy_with_feedback` y `v_retrain_dataset`.

### 3.12 MLflow, Langfuse y Prometheus/Grafana
- **MLflow:** entrenamiento y fine-tune de Laya, cada corrida del harness de evaluación (§5) y registry con aliases (`production`, `candidate`).
- **Langfuse (Cloud free por defecto):** trazas del agente System 2 y del juez, con *datasets* para el set dorado. Self-hosted en `⏳ 16GB`.
- **Prometheus + Grafana:** latencia por etapa, cobertura de System 1, tasa de escalamiento por razón, profundidad de las colas y oleadas (§6).

### 3.13 LLM Gateway (`llm-gateway`, proyecto P0 en Python)
Lo usan el agente System 2 (alias `smart`), el juez (alias `judge`, apuntando a otra familia de modelos) y el baseline LLM-only. Proveedores: `mock` (tests y CI) y `ollama` (local). **Nunca Haiku en P2** (los aliases de P2 no incluyen `anthropic`). La salida estructurada del agente usa el `response_format` del gateway (ver `../llm-gateway/PLAN.md` §6).

### 3.14 Aporte al README
Cada componente es una subsección de **Components**, con el detalle técnico plegado en `<details>`.

---

## 4. Modelo: datos, fine-tune, calibración y umbrales

### 4.1 Por qué un decision model (concepto)
| Opción | Qué devuelve | Latencia (CPU) | Costo | Problema |
|---|---|---|---|---|
| Reglas | Sí/no | µs | 0 | Frágiles: negaciones, sinónimos |
| Clasificador clásico (SetFit/BERT) | Probabilidades de **una** pregunta | ms | ~0 | Hace falta un modelo por pregunta |
| **Decision model (Laya)** | Probabilidades de **varias** preguntas en una pasada | decenas–cientos de ms | ~0 | Ecosistema nuevo (sept. 2026) |
| LLM | Texto (hay que parsearlo) | cientos de ms–s | Por token | Caro y lento a escala; sin probabilidades confiables |

### 4.2 Datos
| Conjunto | Origen | Tamaño aprox. | Uso |
|---|---|---|---|
| Banking77 train/test | Público (inglés) | 10k / 3k | Base; las 77 intenciones se mapean a ~12 rutas (`config/route_map.yaml`) |
| Banking77-ES | Traducción + paráfrasis con LLM local (gateway, $0) | ~6k | Entrenamiento en español |
| Etiquetas auxiliares | `urgency`, `needs_human` y `needs_kb` por reglas a partir de la intención, más revisión manual de una muestra | — | Las 3 preguntas extra de Laya |
| **Casos difíciles** | Escritos y generados a propósito: varias intenciones, negaciones, sarcasmo, mensajes muy cortos, fuera de dominio | ~400 | Evaluar la política de escalamiento |
| **Set dorado** | **Revisado a mano**, estratificado por ruta e idioma, incluye casos difíciles | ~500 | La verdad para todas las comparaciones (§5) |

- Split por **paráfrasis de origen**: si una frase está en train, sus traducciones y paráfrasis **no** van al test (evita leakage).
- **Verificar la licencia de Banking77** al implementar y citarla en el README.

### 4.3 Baselines (todos evaluados igual)
| # | Router | Notas |
|---|---|---|
| B0 | Reglas y palabras clave | El piso |
| B1 | SetFit (few-shot) | El clasificador clásico moderno, en CPU |
| B2 | Laya zero-shot | Solo con las preguntas, sin fine-tune |
| B3 | **Laya fine-tuneada** (+ ONNX int8) | El System 1 propuesto |
| B4 | LLM-only (Qwen3 1.7B/4B local, salida estructurada vía gateway) | El "todo con LLM" |
| B5 | **B3 + System 2** | El sistema completo |

⚠️ **Nota de honestidad:** un LLM local pequeño es más débil que uno de frontera. Por eso B4 se reporta con dos columnas: calidad medida (local) y **costo estimado si fuera Haiku** (tokens reales × precio de referencia, sin llamarlo). En el README se declara que la comparación de calidad favorece al System 1 frente a un LLM pequeño.

### 4.4 Fine-tune de Laya
- **Local (4 GB de VRAM):** el full fine-tune de 322M con AdamW necesita ~5 GB solo en estados del optimizador, así que **no cabe**. Opciones: **LoRA** (PEFT) sobre el encoder + fp16 + gradient checkpointing. Es la opción por defecto.
- **Full fine-tune:** en **Kaggle o Colab gratis** (GPU T4/P100), con un notebook reproducible en `training/notebooks/`.
- Se compara LoRA local vs full fine-tune, y el resultado se registra en MLflow.
- Los hiperparámetros, las semillas y el hash del dataset van a MLflow.

### 4.5 Calibración y umbrales
1. **Temperature scaling** por pregunta en validación. Reporte de **ECE** y *reliability diagrams*, antes y después.
2. **Curva riesgo-cobertura:** para cada umbral τ, qué % resuelve System 1 (cobertura) y con qué error. Se elige τ para una **precisión objetivo** de System 1 (p. ej., 97%).
3. Umbral de la red de seguridad de urgencia elegido para un **recall de urgentes ≥ 98%** (§1.7).
4. Todo va a `policy.yaml` y a MLflow, junto con las curvas para el README.

### 4.6 Export e inferencia
Export a ONNX con Optimum + **cuantización int8 dinámica**. Test de paridad: el acuerdo de la clase top entre fp32 e int8 debe ser ≥ 99.5%, con un delta de ECE pequeño. Benchmark en el i5 con lotes de 1, 8 y 16: p50/p95 por mensaje.

### 4.7 Lazo de mejora continua
Las correcciones humanas (`POST /feedback`) y las decisiones del System 2 **validadas** alimentan `v_retrain_dataset`. Al reentrenar, el nuevo modelo entra como `candidate` en MLflow y solo se promueve si **pasa las compuertas del harness** (§5.4). Con el tiempo, System 2 "le enseña" a System 1 y la cobertura sube; eso se grafica.

### 4.8 Aporte al README
**Why a Decision Model**, **Data**, **Baselines**, **Fine-tuning on a 4 GB GPU**, **Calibration & Selective Classification** y **Continuous Improvement Loop**.

---

## 5. Harness de evaluación

### 5.1 Concepto
"Funciona en mis ejemplos" no es evidencia. El harness es un **banco de pruebas reproducible**: mismos datos, mismas métricas y el mismo código para todos los routers, versionado en MLflow. Es *Harness Engineering* aplicado a la calidad (tu skill del CV).

### 5.2 Métricas
| Nivel | Métrica |
|---|---|
| Por pregunta (Laya) | Macro-F1, accuracy, ECE, Brier |
| Ruta final | Macro-F1, **recall de urgentes**, matriz de confusión **ponderada por costo** (§ADR-4) |
| Política | Cobertura de System 1, precisión de System 1, curva riesgo-cobertura, **tasa de escalamientos útiles** (casos donde System 2 corrigió a Laya) |
| System 2 | Acierto sobre los escalados, latencia, pasos, uso de herramientas, tasa de fallback |
| Costo | Llamadas al LLM por cada 1.000 mensajes, tokens y costo estimado a precio de referencia |
| Latencia | p50/p95 de System 1 (in-process y de punta a punta) |

### 5.3 LLM-as-a-Judge (validado)
- **Uso:** el set dorado da la ruta correcta, pero **no** evalúa el `rationale` del agente ni los casos genuinamente ambiguos. El juez califica la calidad del razonamiento y si la evidencia citada es coherente.
- **Juez local de otra familia que el agente:** Gemma 3 4B (cabe solo en 4 GB de VRAM y se corre **separado** del agente) frente al agente Qwen3. Así se evita el *self-preference bias*.
- **Validación del juez:** 100 casos etiquetados a mano y se mide el acuerdo juez↔humano (Cohen's kappa). **Si el kappa es bajo, el juez no se usa** para conclusiones, solo como señal. El kappa se reporta en el README.

### 5.4 Compuertas de regresión (CI)
En cada PR que toca el modelo, la política o los prompts, se corre en la CI un subconjunto del set dorado con **Laya ONNX en CPU**. El LLM va en `mock`, porque la CI no corre LLMs reales.
- ❌ Falla si la macro-F1 cae más de 1 punto frente al `production` actual.
- ❌ Falla si el recall de urgentes queda por debajo de 98%.
- ❌ Falla si el ECE sube más de 0.02.
La evaluación completa (con System 2 y el juez) corre local con `make eval-full`.

### 5.5 Drift en producción
Se monitorea la distribución de intenciones (PSI frente al entrenamiento), la **distribución de confianza** (si baja la confianza media, llegan mensajes que el modelo no conoce) y la tasa de `ood`. Una alerta sugiere reentrenar (§4.7).

### 5.6 Tabla de resultados (plantilla)
| Router | Macro-F1 | Recall urg. | ECE | Cobertura S1 | p95 (ms) | Llamadas LLM/1k | Costo est./1k msgs |
|---|---|---|---|---|---|---|---|
| B0 Reglas | | | — | 100% | | 0 | $0 |
| B1 SetFit | | | | 100% | | 0 | $0 |
| B3 Laya FT | | | | 100% | | 0 | $0 |
| B4 LLM-only | | | — | 0% | | 1000 | ${H} (ref. Haiku) |
| **B5 Laya + S2** | | | | {X}% | | {N} | ${h} |

### 5.7 Aporte al README
**Evaluation Harness**, **LLM-as-a-Judge (and how we validated it)**, **Results** y **Regression Gates**.

---

## 6. Observabilidad
| Señal | Herramienta | Qué se ve |
|---|---|---|
| Métricas | Prometheus + Grafana | Throughput, p50/p95 por etapa, cobertura de System 1, escalamientos por razón, profundidad de colas, lag, oleadas |
| Trazas del agente | **Langfuse** | Cada ejecución de System 2: pasos, herramientas, prompts, tokens y latencia; puntajes del juez enlazados |
| Logs | JSON estructurado | `message_id` y `trace_id`; **el texto del mensaje no se loguea completo** (PII en un caso real; aquí se hace por disciplina) |

**Dashboards (como código):**
- **R1 · Overview:** volumen, p95, cobertura de System 1 y tasa de escalamiento.
- **R2 · Routing Quality:** distribución de rutas, confianza media, PSI, `ood` y feedback humano.
- **R3 · Surges:** volumen por intención, línea base EWMA y alertas.
- **R4 · System 2:** latencia, pasos, fallback y tokens.

**Alertas:**
- Cobertura de System 1 por debajo de lo esperado (drift).
- Oleada detectada.
- Cola de un equipo por encima de su capacidad.
- Latencia o fallback de System 2 alto.
- DLQ con mensajes.

**Aporte al README:** sección **Observability** con capturas de R1 y R3 y una traza de Langfuse.

---

## 7. Presupuesto de recursos (8 GB / `⏳ 16GB`)
**Hardware:** i5-10300H (4C/8T), 8 GB de RAM y 4 GB de VRAM. Usa el mismo `.wslconfig` que P1 (`memory=5GB`, `processors=6`). Ollama corre nativo en Windows.

| Perfil | Servicios | RAM en la VM | VRAM |
|---|---|---|---|
| **`core`** (System 1 + streaming) | Redpanda 600 MB · Dataflow con Laya int8 800 MB · API 300 MB · Dispatcher 150 MB · Redis 150 MB · Postgres 250 MB · Prometheus 300 MB · mocks P1/P3 100 MB | **≈ 2.7 GB** ✅ | 0 |
| **`agent`** (core + System 2) | + System 2 Worker 250 MB | **≈ 2.9 GB** ✅ | Qwen3 1.7B Q4 ~1.3 GB |
| **`eval-judge`** (sin streaming) | Harness + Gemma 3 4B en Ollama, **solo** | ~1 GB | ~3 GB (el agente no está cargado) |
| **`train`** | Fine-tune con LoRA (Laya 322M) | ~2 GB | ~3–3.5 GB (fp16 + checkpointing) |
| **`ops`** | core/agent + Grafana 150 MB + MLflow 200 MB | ≈ 3.3 GB ✅ | — |
| **`full`** `⏳ 16GB` | + Langfuse self-hosted (Postgres + ClickHouse + workers) + LLM local de 4B para el baseline B4 junto con el agente | +3–4 GB | — |

P2 es el más liviano de los 3 proyectos. La VRAM es el recurso que se reparte por turnos (agente vs juez vs entrenamiento): `make doctor` verifica que Ollama no tenga otro modelo cargado.

**Aporte al README:** **Hardware Requirements & Profiles**.

---

## 8. Ejecución local y alternativas de despliegue

### 8.1 Inicio rápido
```bash
git clone https://github.com/Leito2/<repo-p2> && cd <repo-p2>
cp .env.example .env              # LLM_PROVIDER=mock por defecto
make doctor && make setup         # uv sync + imágenes
make data                         # Banking77 + mapeo de rutas + splits (la versión ES se descarga de los releases del repo o se regenera con make data-es)
make model                        # descarga Laya y aplica el checkpoint fine-tuneado (o make train)
make up PROFILE=core
make smoke                        # 50 msg/s durante 2 min → cobertura de S1, p95 y distribución de rutas
```

### 8.2 Flujos
| Flujo | Comandos |
|---|---|
| Fine-tune local (LoRA) | `make train` |
| Full fine-tune en Kaggle/Colab | Abrir `training/notebooks/laya_full_ft.ipynb` → `make import-checkpoint` |
| Calibrar y elegir umbrales | `make calibrate` → `policy.yaml` + curvas |
| Evaluación rápida (como la CI) | `make eval` |
| Evaluación completa (S2 + juez) | `ollama pull qwen3:1.7b gemma3:4b` → `make eval-full` |
| Sistema completo con agente | `make up PROFILE=agent` + `LLM_PROVIDER=ollama` |
| Demo de oleada | `make surge-demo` (inyecta "card_declined ×5" y muestra la alerta en R3) |
| Probar un mensaje | `curl -X POST localhost:8000/route -d '{"text":"me robaron la tarjeta"}'` |

### 8.3 Alternativas
| Opción | Uso | Costo |
|---|---|---|
| **Compose local** | Principal | $0 |
| **Kaggle/Colab** | Full fine-tune de Laya | $0 (cuotas gratis; verificar límites vigentes) |
| **Langfuse Cloud (free)** | Trazas sin consumir RAM local | $0 en el free tier |
| **Hugging Face Hub** | Publicar el checkpoint fine-tuneado y la versión ES del dataset (si la licencia lo permite) | $0 |
| **HF Spaces (CPU básico gratis)** | Demo pública de `POST /route` (solo System 1) | $0; opcional |
| Kubernetes / cloud | Documentado como camino a producción (por regla, solo P3 usa GCP) | — |

**Mapeo a producción:** Redpanda → Redpanda Cloud o MSK; Quix Streams → contenedores en K8s (escala con procesos del consumer group, ≤ particiones); Laya → servicio ONNX en CPU con autoescalado (barato, sin GPU); System 2 → LLM gestionado (Haiku) detrás del gateway; Langfuse Cloud.

### 8.4 Troubleshooting específico
- Laya lenta en CPU → revisar los threads de ONNX y el tamaño de lote; confirmar que se usa el modelo int8.
- Estado lento al reiniciar → Quix restaura el estado reproduciendo el changelog topic; mantener ventanas acotadas y changelogs compactados; documentar el tiempo de restauración en la demo.
- Ollama sin VRAM → `ollama ps` y descargar el modelo que no se usa (`ollama stop <modelo>`).
- Más los mismos problemas de Windows y Docker que P1 (CRLF, listeners, `host.docker.internal`).

**Aporte al README:** **Quickstart**, **Running Experiments & Demos**, **Deployment Options** y **Troubleshooting**.

---

## 9. Estructura del repo y del README

### 9.1 Repo: `smart-request-router`
```
smart-request-router/
├── README.md · LICENSE · Makefile · docker-compose.yml (profiles) · .env.example · pyproject.toml (uv)
├── packages/routercore/          ← compartido: laya.py, policy.py, contracts.py, route_map, telemetry
├── services/
│   ├── replayer/ · dataflow/ · system2/ · dispatcher/ · api/
│   └── mocks/ (p1_decisions, p3_ask)    ← contratos falsos para correr P2 solo
├── training/
│   ├── data_prep.py · translate_es.py · hard_cases.py · golden_set/
│   ├── train_lora.py · calibrate.py · export_onnx.py · baselines/ (rules, setfit, llm_only)
│   └── notebooks/laya_full_ft.ipynb
├── eval/                         ← el harness: runners, métricas, juez, validación del juez, gates
├── config/ (route_map.yaml, policy.yaml, surge.yaml, prompts/)
├── infra/ (redpanda, redis, postgres/init.sql)
├── observability/ (prometheus, grafana dashboards R1–R4, alerts)
├── tests/ (unit, contract, integration)
└── docs/ (adr/, results/, eval-reports/, images/)
```

### 9.2 Esqueleto del README (en inglés)
```markdown
# 🧭 Smart Request Router — System 1 / System 2
> headline with real numbers · badges · GIF (surge demo + routing live)
## TL;DR — Results at a Glance
## Part I — The Big Picture
  1. The Problem: routing support at fintech scale
  2. Core Concepts Primer: decision models vs classifiers vs LLMs · calibration & ECE ·
     selective classification · System 1/System 2 · agents with tools · stream processing in Python
  3. The System 1 / System 2 Thesis
  4. Architecture · 5. Design Decisions · 6. Journey of a Message
## Part II — Components (Concept → How → Technical details)
## Part III — The Model (data, baselines, fine-tuning on 4 GB, calibration, thresholds, improvement loop)
## Part IV — Proof (evaluation harness, judge validation, results, cost analysis, observability)
## Part V — Run It Yourself (hardware, quickstart, demos, deployment options, troubleshooting, structure)
## Part VI — Reflection (lessons, limitations, future work, glossary, references, license)
```

---

## 10. Hitos de implementación y criterios de aceptación
**Definición de terminado** (todos los hitos): CI en verde · cabe en 8 GB · secciones del README escritas · $0 · tag del hito.

| Hito | Objetivo | Criterios de aceptación | README | Curso | Tamaño |
|---|---|---|---|---|---|
| **M0 · Bootstrap** | Repo, Compose, CI, esqueleto del README | `make doctor` pasa; CI en verde | Problem, Core Concepts (borrador) | — | S |
| **M1 · Datos** | Banking77 + mapeo de rutas + ES + casos difíciles + **set dorado** | Splits sin leakage; set dorado revisado (500); licencia verificada | Data | — | M |
| **M2 · Baselines + harness v1** | B0, B1 y B4 evaluados con el mismo harness en MLflow; **`judgekit` v1** (métricas deterministas, runner async con aiohttp, reportes en Pandas) | Tabla con 3 routers; harness reproducible con `make eval`; `judgekit` instalable desde otro repo | Evaluation Harness v1 | C7 (00–02) | M |
| **M3 · Laya** | B2, fine-tune LoRA (+ full en Kaggle), calibración, ONNX int8 | B3 ≥ B1 en macro-F1; ECE < 0.05 después de calibrar; paridad int8 ≥ 99.5% | Why a Decision Model, Fine-tuning, Calibration | C7 (03–04) | L |
| **M4 · Streaming System 1** | Redpanda + Quix Streams + política + API + Dispatcher (con mocks) | `make smoke` da cobertura y p95; **primera cifra del CV** (cobertura de S1 @ precisión objetivo, p95) | Architecture, Components (dataflow, policy, API) | C2 (03), C3 (02) | L |
| **M5 · System 2** | Agente LangGraph + herramientas + guardrails + Langfuse | B5 evaluado; System 2 **supera** a Laya en los escalados (si no, se documenta y se simplifica) | System 2 Agent | C7 (05) | L |
| **M6 · Oleadas + observabilidad** | Detector EWMA + `alerts` + dashboards R1–R4 + alertas | `make surge-demo` detecta la oleada en < 2 min; GIF grabado | Surge Detection, Observability | C6 | M |
| **M7 · Calidad continua** | Juez validado (kappa), compuertas en la CI, drift, lazo de feedback | Kappa reportado; un PR con regresión falla en la CI (demostrado); reentrenamiento con feedback | LLM-as-a-Judge, Regression Gates, Improvement Loop | — | M |
| **M7b · Evaluación industrial** (§12) | Jueces de `judgekit` (rúbrica, pairwise, claims, seguridad y marca), panel con `judge_golden`, Langfuse (datasets, experiments, prompts, anotación), SageMaker Processing local + MinIO versionado, pipeline KFP `quality-loop` local, Evidently | Kappa de ambos jueces reportado; un *processing job* local produce el reporte desde MinIO; el pipeline reentrena o re-promptea ante una degradación inyectada; % de QA ahorrado medido | LLM-as-a-Judge, Industrial Evaluation, Quality Loop | — | L |
| **M8 · Integraciones** | Contratos reales con P1 (`/decisions`) y P3 (`/ask`) | Con P1/P3 levantados, el dispatcher los usa; sin ellos, los mocks | Integrations | — | S |
| **M9 · Pulido y publicación** | README completo, checkpoint en HF Hub, Space opcional, `v1.0` | Una persona ajena lo corre desde el README | Todo | — | M |
| M10 · `⏳ 16GB` | Langfuse self-hosted, B4 con un modelo de 4B, todo simultáneo | Resultados `v1.1` | Results | — | S |

**MVP para el CV:** M0 → M5 (la cifra de cobertura y costo con System 2 incluido) + M7b parcial (`judgekit` con kappa y Langfuse).
**Orden de recorte:** M8 → M6 (se conserva el detector sin dashboards completos) → M7 parcial (se conservan las compuertas de la CI).

---

## 11. Riesgos y pendientes

### 11.1 Riesgos
| ID | Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|---|
| **R1** | Laya es un proyecto nuevo (sept. 2026): API, documentación o checkpoints inmaduros o cambiantes | Media | Alto | Fijar versiones; aislar Laya detrás de una interfaz en `routercore/laya.py`; **plan B: Von o ModernBERT con cabezas multi-tarea** (misma interfaz) |
| **R1b** | **Latencia de Laya en CPU** (hallazgo del curso C7): el fabricante reporta 193–464 ms por request y un reporte de la comunidad mide varios segundos en una laptop sin optimizar | Media | Alto | En M3: preload, ONNX int8, micro-lotes y entradas cortas; medir el p95 en el i5. Si el p95 supera 300 ms, usar **Von (OpenVINO CPU)** |
| **R1c** | **Colapso con muchas opciones** (Banking77 con 77 opciones: 0.425 en Laya vs 0.870 en Jev) | Alta si se usan las 77 | Alto | Ya mitigado en el diseño: **~12 rutas**; sub-intención como segunda pregunta opcional |
| **R1d** | **El checkpoint multilingüe viene sin calibrar** (ECE 0.314) y las preguntas `score` son las más débiles | Alta | Medio | Temperatura por idioma y por pregunta, con compuerta de ECE en la CI; la red de seguridad de urgencia no depende solo del `score` |
| **R2** | La licencia de Banking77 no permite algún uso (p. ej., redistribuir la versión ES) | Baja | Medio | Verificar en M1; si no se puede redistribuir, publicar solo el script que la genera |
| **R3** | El español sintético es de baja calidad (traducción local con un modelo pequeño) | Media | Medio | Revisión manual del set dorado ES; reportar las métricas por idioma |
| **R4** | Comparación injusta: LLM-only local débil vs System 1 | Alta | Medio | Nota de honestidad (§4.3) + costo estimado a precio de Haiku; la tesis se apoya en **costo y latencia**, no solo en calidad |
| **R5** | El System 2 con un LLM de 1.7B no supera a Laya en los casos difíciles | Media | Alto | Probar el modelo de 4B en `⏳ 16GB` o por turnos; documentar el hallazgo con honestidad ("en este hardware, S2 aporta X"). También es un resultado válido |
| **R6** | Sesgo o baja calidad del juez | Media | Medio | Juez de otra familia + kappa con humanos (§5.3) |
| **R7** | Quix Streams: cambios de API entre versiones 3.x (helpers de ventanas renombrados) | Media | Bajo | Fijar la versión en M0; aislar el dataflow detrás de funciones puras testeadas sin Quix. Bytewax queda documentado como alternativa (sin releases desde nov-2024) |
| **R8** | La VRAM de 4 GB obliga a turnar agente, juez y entrenamiento | Alta | Bajo | Perfiles separados + `doctor`; `⏳ 16GB` no ayuda en VRAM, solo en RAM |
| **R9** | Scope creep | Media | Alto | MVP M0 → M5 y orden de recorte |

### 11.2 Verificar al implementar
API y formato de entrada de Laya (preguntas y opciones) · licencia de Banking77 · versión fijada de Quix Streams y forma de salida de sus ventanas · soporte de Optimum para exportar la arquitectura mmBERT · cuotas de Kaggle y Colab · límites del free tier de Langfuse Cloud.

### 11.3 Pendientes `⏳ 16GB`
Langfuse self-hosted · baseline B4 con un modelo de 4B en simultáneo · todos los perfiles a la vez para la demo integrada con P1 y P3.

### 11.4 Preguntas abiertas
1. ~~¿El gateway soporta salida estructurada?~~ → **resuelto:** `llm-gateway` implementa `response_format` con JSON Schema, validación y un reintento (su hito M7).
2. ~~¿Nombre del repo `smart-request-router`?~~ → **resuelto:** repo público `Leito2/smart-request-router`.
3. ¿Una corrida opcional en AWS real (SageMaker Processing + S3, centavos) como prueba final de `judgekit`? (ver §12.6)


---

## 12. v2 — Evaluación industrial: `judgekit` (absorbe el proyecto "Automated LLM Evaluation Suite" del CV)

> **Decisión (2026-10-06):** el proyecto del CV *Enterprise MLOps: Automated LLM Evaluation Suite* no tiene repo y queda aparte. **Todo** lo que describe (asyncio + aiohttp + Pandas, SageMaker Processing sobre S3, Vertex AI Pipelines, Model Monitor, Gemma 4 31B como *golden evaluator*, Hugging Face Evaluate y tokenizers, algoritmos de token matching, auditoría de alucinaciones, compuertas de calidad) se implementa aquí como componente real, mejorado con **Langfuse**. P2 es su casa porque ya tiene el harness más completo; P1, P3 y P4 lo consumen como librería.

### 12.1 Mapa de absorción
| Elemento del proyecto del CV | Cómo existe en P2 | Costo |
|---|---|---|
| Framework asíncrono de LLM-as-a-Judge (asyncio) | `packages/judgekit`: runner con `asyncio.TaskGroup`, semáforo de concurrencia, reintentos, reanudación desde checkpoint y estimación de costo **antes** de correr | $0 |
| aiohttp | Cliente HTTP del runner (`aiohttp.ClientSession` con pool de conexiones) contra el gateway P0; P0 usa httpx: así ambos clientes quedan demostrados y se comparan en throughput | $0 |
| Pandas | Resultados como DataFrame: cortes por idioma, ruta y dificultad; intervalos de confianza por *bootstrap*; export a Parquet | $0 |
| Gemma 4 31B como *Golden Evaluator* (contexto de 256K) | Alias `judge_golden` del gateway → Google AI Studio (free tier, solo datos sintéticos). Valida al juez local y juzga **trazas completas** del agente, que no caben en el contexto de un modelo pequeño | $0 |
| Hugging Face Transformers, Tokenizers y Evaluate | Métricas deterministas (`evaluate`: exact match, ROUGE-L, BLEU/chrF, BERTScore opcional) y conteo de tokens con el tokenizer de cada modelo | $0 |
| "Algoritmos especializados de token matching" | Exact match normalizado, F1 a nivel de token (estilo SQuAD), *fuzzy matching* (`rapidfuzz`) para nombres de rutas y entidades, y verificación de IDs de cita | $0 |
| Auditoría de alucinaciones | El `rationale` del agente System 2 se descompone en afirmaciones y cada una se verifica contra las salidas de sus herramientas (historial, decisiones de P1, KB de P3); métrica `unsupported_claim_rate` | $0 |
| Benchmarking de alineamiento, seguridad y marca | Rúbricas versionadas: tono de marca (guía de estilo de soporte fintech), seguridad (sin asesoría financiera, sin filtrar PII, sin prometer reembolsos), cumplimiento de la política de escalamiento | $0 |
| Gating "listo para producción" | `judgekit gate`: umbrales en YAML → código de salida; corre en la CI (subset con `mock`) y antes de promover un modelo o un prompt | $0 |
| **SageMaker Processing Jobs** sobre datasets versionados en **S3** | El runner se empaqueta como contenedor y se ejecuta como *Processing Job* en **modo local** del SDK de SageMaker (corre en Docker) leyendo de **MinIO** (API S3, *object versioning* activado). El mismo job corre en SageMaker real cambiando la sesión (opcional, ver §12.6) | $0 local |
| **Vertex AI Pipelines** que disparan reentrenamiento o *prompt tuning* | Pipeline KFP v2 `quality-loop`: evaluar → comparar con umbrales → si se degrada, **reentrenar Laya** (LoRA con el feedback) u **optimizar el prompt** del System 2 (búsqueda de variantes evaluadas por `judgekit`) → registrar en MLflow → compuerta. Corre local con `kfp.local` (`DockerRunner`); el YAML compilado es el mismo que se ejecuta en Vertex AI Pipelines durante la prueba final de P3 (único proyecto en GCP) | $0 local |
| **SageMaker Model Monitor** (drift semántico, observabilidad) | **Evidently**: drift de la distribución de intenciones, de la confianza y **drift semántico de embeddings** de los mensajes; reportes y *test suites*; se documenta la equivalencia con Model Monitor (baseline → constraints → schedule → violations) | $0 |
| Gobernanza de datos | *Dataset cards*, manifiesto con hash de contenido por versión, linaje en MLflow (digest del dataset en cada corrida), escaneo de PII con Presidio antes de que un dato entre a un dataset, retención documentada | $0 |
| "Reduce el QA manual hasta 90%" | Se **mide**: tiempo humano por caso (cronometrado al etiquetar el set de kappa) × casos vs tiempo del runner; se publica el % real | $0 |

### 12.2 Langfuse como columna vertebral de la evaluación
| Función de Langfuse | Uso en P2 |
|---|---|
| **Datasets** | El set dorado y el de casos difíciles, versionados |
| **Experiments** (dataset runs) | Cada versión del router (B0–B5) y cada variante de prompt es un experimento comparable |
| **Scores** | Métricas deterministas y del juez adjuntas a cada traza del agente |
| **LLM-as-a-Judge gestionado** | Evaluadores sobre una muestra del tráfico en vivo (online eval), además del batch de `judgekit` (offline eval) |
| **Prompt management** | Prompts del agente y de los jueces versionados; el pipeline de *prompt tuning* publica la nueva versión con etiqueta `candidate` |
| **Annotation queues** | Etiquetado humano para el kappa juez↔humano |

Despliegue: **Langfuse Cloud (plan Hobby, gratis)** con datos sintéticos ahora; self-hosted en `⏳ 16GB` (la v3 pide ClickHouse, Postgres, Redis y blob storage: no entra en 8 GB junto con el stack).

### 12.3 Diseño de `judgekit` (librería compartida)
```
packages/judgekit/
├── datasets.py    # carga desde local, MinIO/S3 o Langfuse; manifiesto con hashes
├── metrics/       # deterministas: exact, token_f1, rouge, chrf, fuzzy, citation_ids, json_valid
├── judges/        # rubric (1–5 con CoT), pairwise (con intercambio de posición), reference-guided,
│                  # claim-level faithfulness, safety/brand; salida validada con JSON Schema vía P0
├── runner.py      # asyncio + aiohttp, concurrencia acotada, reintentos, checkpoint/reanudación, costo estimado
├── calibration.py # kappa de Cohen, Spearman, acuerdo entre jueces (panel), acuerdo por umbral
├── report.py      # Pandas: cortes, bootstrap CIs, tablas Markdown para el README
├── gate.py        # umbrales YAML → pass/fail (código de salida)
├── sinks/         # MLflow, Langfuse, Parquet en MinIO/S3
└── cli.py         # judgekit run | report | gate | calibrate
```
- **Mitigación de sesgos del juez:** juez de otra familia que el agente (*self-preference*), intercambio de posición en pairwise (*position bias*), control de longitud (*verbosity bias*), CoT antes del puntaje, referencia cuando existe, y **panel de jueces** (Gemma 3 4B local + Gemma 4 31B golden) con voto o promedio.
- **Consumo desde otros repos:** `judgekit @ git+https://github.com/Leito2/smart-request-router#subdirectory=packages/judgekit`. P1 la usa para la fidelidad de las explicaciones, P3 para faithfulness y citas del RAG, P4 para los reportes de investigación.

### 12.4 Métricas nuevas
| Métrica | Meta |
|---|---|
| Throughput del runner (evaluaciones por minuto) con `judge` local y con `judge_golden` | Se reporta |
| Kappa juez local ↔ humano y juez golden ↔ humano | ≥ 0.6 para usar el juez en conclusiones |
| Reducción del tiempo de QA frente a revisión manual | Se reporta (el CV dice hasta 90%) |
| `unsupported_claim_rate` del System 2 | Se reporta y tiene compuerta |
| Tiempo desde que se degrada una métrica hasta que el pipeline propone un candidato | Se reporta |

### 12.5 Frase del CV (agregado)
> … with an async **LLM-as-a-Judge** framework (asyncio · aiohttp · Pandas · HF Evaluate) validated against human labels (κ {k}), a **Gemma 4 31B golden evaluator**, Langfuse datasets/experiments, SageMaker Processing jobs over versioned S3 datasets and a KFP/Vertex AI quality loop that retrains or re-prompts on degradation — **{q}% less manual QA time**.

### 12.6 Decisiones pendientes y riesgos
| Tema | Decisión / mitigación |
|---|---|
| ¿Correr una vez en **AWS real** (SageMaker Processing + S3)? | **Pendiente del usuario.** Por defecto todo es local ($0). Una corrida real de pocos minutos en una instancia pequeña cuesta centavos, pero rompe la regla de "un solo proyecto en la nube"; se propone como prueba final opcional |
| El modo local de SageMaker en Windows | Correrlo dentro de WSL2; verificar al implementar el soporte de `LocalSession` con un endpoint S3 personalizado (MinIO) |
| Cuotas del free tier de Google AI Studio para `judge_golden` | Solo subsets de validación (cientos de casos, no miles); el gateway respeta las cuotas (P0 §4) |
| Términos de uso de los free tiers | Solo datos sintéticos o públicos (Banking77) |
| Scope creep | `judgekit` v1 = métricas deterministas + runner + reporte; los jueces, Langfuse y los pipelines llegan en M7b |
