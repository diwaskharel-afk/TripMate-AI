# Predictive Maintenance Multi-Agent System — Full Build Guide

This is a complete, step-by-step build plan for the pipeline you described:

```
Sensor Stream → Anomaly Agent → [decision] → Diagnosis Agent (RAG) → Scheduling Agent (optimizer) → Report Agent (LLM) → Work Order
```

The core engineering idea to hold onto throughout: **you're building a pipeline where cheap/fast/deterministic logic filters and structures the problem, and expensive/slow LLM calls are only invoked at the two points where language understanding is actually the bottleneck** (diagnosis and reporting). Everything else is plumbing. That's what makes it a real "agentic system" rather than a chatbot with extra steps.

---

## 0. Decide your tech stack (and why)

You don't need heavy frameworks to make this work. Recommended stack, with reasoning:

| Piece | Tool | Why |
|---|---|---|
| Language | Python | Best ecosystem for data, ML, RAG, and LLM SDKs |
| Data simulation | `numpy` / `pandas` | Generate believable sensor time series |
| Orchestration | Plain Python functions + a simple event loop, OR `LangGraph` | Start simple; you can always graduate to a framework once the logic is proven. A framework *before* you understand the control flow just hides bugs. |
| Anomaly detection | `numpy`/`scipy` (z-score, EWMA, or `river` for streaming stats) | This is classic statistics, not ML in the "learned model" sense — no LLM, no training needed for v1 |
| Vector store (RAG) | `chromadb` (local, zero setup) | Lets you do the manual-search step without standing up infrastructure |
| Embeddings | Any embedding API (OpenAI, Voyage, or local `sentence-transformers`) | Needed to turn manual text into searchable vectors |
| LLM calls | Anthropic API (`claude-sonnet-4-6` or similar) | Powers steps 4 and 6 only |
| Scheduling/optimization | `python-constraint` or `ortools` (Google OR-Tools) for real optimization; plain rule-based logic is fine for a v1 | This must be *real* logic, not an LLM guessing a date — that's the whole point of the design |
| Output | Just print/log a JSON "work order" object, or write to a file / a fake "Maximo" JSON endpoint | Simulates a work-order system without needing real integration |

Why this matters: a common mistake in "agent" projects is routing everything through an LLM because it's easy. Your project's *design lesson* is proving that you shouldn't. Keep that discipline as you build.

---

## 1. Build the sensor stream (fake "live" data)

**What it is:** not an agent, just a generator producing rows like:

```python
{"timestamp": ..., "equipment_id": "Pump-14", "temperature": 71.2, "vibration": 0.42, "rpm": 1180, "torque": 55.3}
```

**Why you need this first:** every downstream agent depends on having a believable, controllable data source. If you start with a "real" data feed, you can't easily inject the anomaly you need to test the rest of the pipeline. Build the simulator so you're in control of when things go wrong.

**How to build it:**

1. Model each sensor as noise around a baseline: `value = baseline + gaussian_noise`.
2. Add a function to **inject a fault** on demand — e.g., after N ticks, ramp vibration up gradually (real bearing wear looks like a slow climb, not a step function) or spike it suddenly (simulating a different fault type).
3. Emit one row every X seconds using `time.sleep()` in a loop, or if you want to "replay a dataset," pull rows from a public dataset like NASA's CMAPSS turbofan degradation dataset or a bearing vibration dataset from Kaggle, and stream them out row-by-row with a delay.

```python
import random, time, itertools

def sensor_stream(equipment_id="Pump-14", fault_after=50):
    baseline = {"temperature": 70, "vibration": 0.05, "rpm": 1200, "torque": 50}
    for i in itertools.count():
        drift = max(0, i - fault_after) * 0.01  # slow degradation after fault_after ticks
        reading = {
            "timestamp": time.time(),
            "equipment_id": equipment_id,
            "temperature": baseline["temperature"] + random.gauss(0, 0.5) + drift * 2,
            "vibration": baseline["vibration"] + random.gauss(0, 0.01) + drift,
            "rpm": baseline["rpm"] + random.gauss(0, 5),
            "torque": baseline["torque"] + random.gauss(0, 1),
        }
        yield reading
        time.sleep(1)  # simulate "live" cadence
```

**Why this design:** the `fault_after` + `drift` pattern lets you *demo* the system reliably — you can show a judge/reviewer/yourself the exact moment anomaly detection fires, instead of hoping random noise happens to cross a threshold during a live demo.

---

## 2. Build the anomaly agent (pure statistics, no LLM)

**What it is:** a rolling statistical check that watches each new reading and asks "is this normal for this piece of equipment?"

**Why no LLM here:** LLMs are slow (hundreds of ms–seconds), costly per call, and non-deterministic. Running one per sensor tick (every few seconds, forever) is wasteful and pointless — a threshold check answers the same question instantly and for free. This step is where you *prove* you understand when NOT to use an LLM, which is the actual point of the assignment/project.

**How to build it — pick one method, in order of sophistication:**

- **Simplest: static threshold.** `if vibration > 0.3: anomaly = True`. Fine for a first pass but brittle.
- **Better: rolling z-score.** Keep a rolling window (e.g., last 30 readings) per sensor. Compute mean/std of the window, flag if the new value is more than *k* standard deviations away.
- **Better still: EWMA (exponentially weighted moving average).** Reacts faster to sustained drift than a flat window, which matters for slow degradation like bearing wear.

```python
from collections import deque
import statistics

class AnomalyAgent:
    def __init__(self, window=30, z_thresh=3.5):
        self.history = deque(maxlen=window)
        self.z_thresh = z_thresh

    def check(self, value):
        self.history.append(value)
        if len(self.history) < 10:
            return False, 0.0  # not enough data yet to judge "normal"
        mean = statistics.mean(self.history)
        std = statistics.pstdev(self.history) or 1e-6
        z = (value - mean) / std
        return abs(z) > self.z_thresh, z
```

Run one `AnomalyAgent` instance **per sensor per equipment ID** (a dict keyed by `(equipment_id, sensor_name)`), because "normal" vibration for Pump-14 is not the same as "normal" for Pump-22.

**The decision point (step 3 in your diagram):** this is just an `if` statement in your main loop, not a separate agent:

```python
is_anomaly, z_score = vibration_agent.check(reading["vibration"])
if is_anomaly:
    trigger_diagnosis(reading, z_score)
else:
    continue  # loop back, keep watching
```

**Why this is "the whole reason it's multi-agent":** everything below this line only runs on the rare anomalous tick. If you skip this gate and pipe every reading into an LLM, you've built a slow, expensive, non-deterministic monitoring system for no benefit over the math check above.

---

## 3. Build the diagnosis agent (RAG over manuals)

**What it is:** given a structured fact ("Pump-14, vibration z-score 4.2, sustained rise over 20 ticks"), retrieve relevant text from equipment manuals/repair logs and ask an LLM to interpret it.

**Why an LLM is actually needed here:** matching a symptom description to the right paragraph in a 200-page PDF manual, and turning that into a plain-English diagnosis, is language understanding — exactly what classic code is bad at and LLMs are good at.

**How to build it:**

1. **Get manual text.** Use real equipment manuals if you have them (PDF → text extraction), or fabricate 5–10 short "manual excerpt" documents covering common fault patterns (bearing wear, misalignment, cavitation, imbalance, lubrication failure). Fabricated data is completely fine for a demo project — the pipeline architecture is the point, not manual authenticity.

2. **Chunk the documents.** Split into paragraph-sized chunks (200–500 tokens) so retrieval returns focused, relevant text rather than whole documents.

3. **Embed and store.** Use `chromadb` for a zero-setup local vector store:

```python
import chromadb
client = chromadb.Client()
collection = client.create_collection("equipment_manuals")

collection.add(
    documents=[chunk1, chunk2, chunk3, ...],
    ids=[f"chunk-{i}" for i in range(len(chunks))],
    metadatas=[{"source": "pump_manual.pdf", "section": "vibration"} for _ in chunks]
)
```

4. **Retrieve on trigger.** When the anomaly agent fires, build a query from the structured anomaly data and search:

```python
results = collection.query(
    query_texts=["vibration spike sustained rise pump bearing"],
    n_results=3
)
```

5. **Call the LLM with retrieved context.** This is the actual "RAG" step — you're grounding the LLM's answer in real (or fabricated) manual text instead of letting it hallucinate a diagnosis from general knowledge:

```python
prompt = f"""You are a maintenance diagnosis assistant.

Anomaly detected:
Equipment: {equipment_id}
Sensor: vibration
Z-score: {z_score:.2f}
Pattern: sustained rise over last {n} readings

Relevant manual excerpts:
{chr(10).join(results['documents'][0])}

Based only on the excerpts above, state the most likely cause and the manual's recommended action and timeframe. Be concise."""

response = call_llm(prompt)  # your Anthropic API call
```

**Why retrieval before generation matters:** without it, the LLM is guessing based on general training knowledge about "vibration = bearing wear," which might be right by coincidence but isn't grounded in *your* equipment's actual manual, isn't verifiable, and won't reflect equipment-specific quirks documented in your real repair logs.

---

## 4. Build the scheduling agent (real optimization, not LLM guessing)

**What it is:** given urgency + diagnosis, find the best time to actually do the repair without needlessly halting production.

**Why this must be real logic, not an LLM:** scheduling is a constraint satisfaction problem — you're checking hard facts (is the technician actually free at that time? is the part actually in stock?) against a calendar. An LLM has no access to ground truth here and will confidently invent a plausible-sounding but wrong answer. This is the second half of your project's core lesson: LLMs are bad at things with objectively correct answers derivable from structured data.

**How to build it — v1 (rule-based, still "real logic"):**

```python
def find_earliest_window(urgency_hours, planned_downtimes, parts_in_stock, technician_calendar):
    for window in planned_downtimes:
        if window.start <= now() + timedelta(hours=urgency_hours):
            if parts_in_stock and technician_available(technician_calendar, window):
                return window
    # no planned window soon enough -> may need to force an unplanned stop
    return find_next_technician_slot(technician_calendar, urgency_hours)
```

**v2 (proper optimizer, if you want to go further):** model it as a constraint problem with `ortools.sat` — variables for "which window," constraints for parts availability, technician availability, and equipment mutual exclusion (can't schedule two big jobs in the same downtime window), and an objective to minimize production disruption or maximize slack before the urgency deadline. This is worth doing if you want to demonstrate genuine optimization skill rather than an if/else chain.

**Fake the supporting data** for the demo: a small JSON/dict of planned downtime windows, a parts inventory dict, and a technician calendar dict. This is exactly the same "believable fake data" principle as step 1 — the scheduling logic is real, the data behind it is simulated.

**Output:** a structured object, not prose:

```python
{"equipment_id": "Pump-14", "window_start": "2026-08-28T02:00", "window_end": "2026-08-28T04:00", "reason": "planned changeover", "technician": "J. Aalto"}
```

Keep this structured (not natural language) — the report agent's job in the next step is specifically to turn structured data like this into prose. Don't do that translation twice.

---

## 5. Build the report agent (LLM again, deliberately)

**What it is:** collect the anomaly facts + diagnosis text + schedule object and produce one clear, human-readable maintenance ticket.

**Why an LLM here too:** this is a synthesis/writing task — turning three separate structured/technical outputs into one coherent paragraph with a checklist, in plain language a technician can act on without reading raw JSON. That's a genuine language task, same category as step 4.

```python
prompt = f"""Write a maintenance work order in plain, professional language for a technician.

Equipment: {equipment_id}
Anomaly: {anomaly_summary}
Diagnosis: {diagnosis_text}
Scheduled window: {schedule['window_start']} to {schedule['window_end']} ({schedule['reason']})
Assigned technician: {schedule['technician']}

Include: a one-paragraph summary, the likely cause, and a short numbered checklist of steps to take during the repair window."""

ticket_text = call_llm(prompt)
```

**Why not just template-string this instead of using an LLM:** you could — a Jinja template would produce readable output too. The LLM version is worth keeping if you want the wording to adapt naturally (e.g., folding in extra context, adjusting tone, handling cases where a field is missing gracefully). If you want to keep the project cheap and deterministic, this step is your best candidate to swap for a plain template — mention that trade-off if you're explaining your design decisions.

---

## 6. Emit the work order (the "human" endpoint)

Simplest version: write the final ticket to a JSON file or print it, simulating a work-order system like Maximo:

```python
import json
with open("work_orders.jsonl", "a") as f:
    f.write(json.dumps({
        "equipment_id": equipment_id,
        "ticket_text": ticket_text,
        "schedule": schedule,
        "created_at": time.time()
    }) + "\n")
```

If you want it to feel more real, build a tiny Flask/FastAPI endpoint that accepts the ticket and displays it in a simple web page — this also gives you a natural demo UI without much extra work.

---

## 7. Wire it all together (the orchestration loop)

This is the part that makes it a "system" rather than four separate scripts. Keep it as plain Python control flow first:

```python
def main():
    agents = {}  # keyed by (equipment_id, sensor)

    for reading in sensor_stream("Pump-14"):
        for sensor in ["vibration", "temperature"]:
            key = (reading["equipment_id"], sensor)
            agents.setdefault(key, AnomalyAgent())
            is_anomaly, z = agents[key].check(reading[sensor])

            if is_anomaly:
                diagnosis = run_diagnosis_agent(reading, sensor, z)
                schedule = run_scheduling_agent(diagnosis)
                ticket = run_report_agent(reading, diagnosis, schedule)
                emit_work_order(reading["equipment_id"], ticket, schedule)
```

**Why start here instead of a framework like LangGraph/CrewAI:** frameworks add value once you have multiple conditional branches, retries, and parallel agents to manage. For a 4-step linear pipeline with one decision gate, a plain function chain is easier to debug, and it forces *you* (not the framework) to understand exactly what data flows between steps — which is the actual skill you're demonstrating. Once this works, porting it into LangGraph as a state machine is a good "v2" if you want the project to look more like a formal agent framework.

---

## 8. Test it end-to-end

1. Run the stream with `fault_after` set low (e.g., 5) so you don't wait long to see the pipeline fire.
2. Confirm: anomaly agent stays silent for normal readings, fires once drift crosses threshold.
3. Confirm: diagnosis agent retrieves a *relevant* manual chunk (spot-check this manually — bad retrieval quietly ruins the whole downstream chain).
4. Confirm: scheduling agent respects your fake constraints (try making all technicians "busy" and confirm it correctly reports no available window, rather than inventing one).
5. Confirm: final ticket text is coherent and includes all three inputs (anomaly, diagnosis, schedule).
6. Try 2+ pieces of equipment concurrently to confirm your per-`(equipment_id, sensor)` state tracking doesn't cross-contaminate.

---

## 9. Where to go further (optional extensions)

- **Multiple fault types:** add manual excerpts for 3–4 distinct fault patterns and have the anomaly agent pass along enough pattern detail (rate of change, which sensors moved together) for the diagnosis agent to distinguish between them.
- **Confidence/urgency scoring:** have the diagnosis agent output a structured urgency level (not just prose) so the scheduling agent has a real numeric input instead of parsing free text.
- **Streaming dashboard:** a simple live chart (sensor value over time, with anomaly points marked) makes the demo much more compelling than console logs.
- **Real optimizer:** swap the rule-based scheduler for OR-Tools if you want to show genuine constraint optimization skills.
- **Human-in-the-loop:** let a "technician" approve/reject the suggested window before the ticket is finalized — closer to how real predictive maintenance systems are deployed.

---

## The one-sentence summary to remember

Steps 2 and 5 are cheap, deterministic, and must stay that way — they're what makes the system fast and trustworthy at scale. Steps 4 and 6 are the only places an LLM's language ability is actually load-bearing. Building the project in that order (data → deterministic gate → LLM diagnosis → deterministic scheduling → LLM writing → output) is what turns this from "four API calls" into an actual multi-agent architecture with a defensible design.
