"""Probe dataset: schema, (de)serialization, and a synthetic generator.

An AFP probe instance is an ``Episode`` = (start frame o0, instruction, action
chunk a). For the real pipeline these come from a held-out split of the model's
training domain (e.g. Bridge for Cosmos-SAC); ``afp/data/`` ships a *synthetic*
set so the harness + mock world model run with zero downloads.

Instruction embeddings
----------------------
``embed_instruction`` maps text to a small unit vector. It is engineered so that
semantics-preserving paraphrases land *close* and semantics-changing
counterfactuals land *far* -- which is exactly the floor/ceiling structure the
instruction channel of the metric needs (see ``afp.perturb`` / ``afp.metrics``).
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from afp.rollout import Episode


def _stable_seed(*parts) -> int:
    """Process-stable 32-bit seed from arbitrary parts.

    Uses blake2b (NOT the builtin ``hash``, which is PYTHONHASHSEED-randomized)
    so embeddings -- and therefore the whole probe set and every verdict -- are
    bit-reproducible across processes and machines.
    """
    h = hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & 0x7FFFFFFF

EMB_DIM = 8

# A tiny task vocabulary: (verb-phrase, object, location). Object+location fix the
# semantics (and thus the embedding / drift direction); verb-phrase varies for
# paraphrases without changing semantics.
_VERBS = ["pick up", "grab", "lift", "take"]
_PLACE = ["and place it on", "and put it on", "and set it down on", "and move it to"]
_OBJECTS = ["the red block", "the blue bowl", "the green cup", "the yellow sponge",
            "the metal can", "the wooden spoon"]
_LOCATIONS = ["the left plate", "the right tray", "the top shelf", "the sink",
              "the cutting board", "the stove"]


def embed_instruction(text: str, dim: int = EMB_DIM) -> np.ndarray:
    """Deterministic unit embedding keyed on the semantic content (object+location).

    We hash the *semantic tokens* present in the text (objects/locations), so
    rewording the verb leaves the embedding (nearly) unchanged while swapping the
    object/location moves it.
    """
    toks = []
    low = text.lower()
    for i, o in enumerate(_OBJECTS):
        if o in low:
            toks.append(("o", i))
    for i, l in enumerate(_LOCATIONS):
        if l in low:
            toks.append(("l", i))
    if not toks:  # fall back to a stable hash of the whole string
        rng = np.random.default_rng(_stable_seed("free", low))
        return _unit(rng.normal(size=dim))
    vec = np.zeros(dim, dtype=np.float64)
    for kind, idx in toks:
        rng = np.random.default_rng(_stable_seed(kind, idx))
        vec += rng.normal(size=dim)
    return _unit(vec)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float64) if n > 1e-8 else v.astype(np.float64)


# -- (de)serialization -----------------------------------------------------
def _b64(arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    return {"b64": base64.b64encode(arr.tobytes()).decode("ascii"),
            "dtype": str(arr.dtype), "shape": list(arr.shape)}


def _unb64(d: dict) -> np.ndarray:
    raw = base64.b64decode(d["b64"])
    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()


def episode_to_json(ep: Episode) -> dict:
    return {
        "episode_id": ep.episode_id,
        "instruction": ep.instruction,
        "instruction_emb": ep.instruction_emb.astype(np.float64).tolist(),
        "actions": ep.actions.astype(np.float64).tolist(),
        "o0": _b64(ep.o0),
        "true_future": _b64(ep.true_future) if ep.true_future is not None else None,
        "meta": ep.meta,
    }


def episode_from_json(d: dict) -> Episode:
    return Episode(
        episode_id=d["episode_id"],
        o0=_unb64(d["o0"]),
        instruction=d["instruction"],
        instruction_emb=np.asarray(d["instruction_emb"], dtype=np.float64),
        actions=np.asarray(d["actions"], dtype=np.float64),
        true_future=_unb64(d["true_future"]) if d.get("true_future") else None,
        meta=d.get("meta", {}),
    ).validate()


def save_episodes(episodes: List[Episode], path: str) -> None:
    with open(path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(episode_to_json(ep)) + "\n")


def load_episodes(path: str) -> List[Episode]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(episode_from_json(json.loads(line)))
    return out


@dataclass
class InstructionPerturbationSpec:
    """One row of paraphrases.jsonl: a paraphrase (floor) + counterfactual (ceiling).

    ``dose_texts`` (optional) is a graded ladder of instruction REWORDINGS of
    increasing counterfactual-ness, aligned index-for-index with the scorer's
    ``instruction_fracs``. It only matters for TEXT-conditioned models
    (``adapter.instruction_input == "text"``): embeddings can be interpolated to
    any fraction, but text cannot, so intermediate dose rungs on a real
    checkpoint require human-written graded texts. When absent, the scorer
    probes a text-conditioned model with the paraphrase (floor) and the full
    counterfactual (dose 1.0) only -- it never wastes a GPU rollout feeding the
    unchanged instruction and calling it an intermediate dose.
    """

    episode_id: str
    paraphrase: str
    paraphrase_emb: List[float]
    counterfactual: str
    counterfactual_emb: List[float]
    dose_texts: Optional[List[str]] = None


def save_paraphrases(specs: List[InstructionPerturbationSpec], path: str) -> None:
    with open(path, "w") as f:
        for s in specs:
            f.write(json.dumps(s.__dict__) + "\n")


def load_paraphrases(path: str) -> List[InstructionPerturbationSpec]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(InstructionPerturbationSpec(**json.loads(line)))
    return out


# -- synthetic generator ---------------------------------------------------
def _random_o0(rng: np.random.Generator, size: int) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    base = np.zeros((size, size, 3), dtype=np.float64)
    # a couple of soft static "objects" so episodes look distinct
    for _ in range(3):
        cx, cy = rng.uniform(0, size, size=2)
        sig = rng.uniform(5, 12)
        col = rng.uniform(40, 200, size=3)
        blob = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sig ** 2)))
        for c in range(3):
            base[..., c] += col[c] * blob
    return np.clip(base, 0, 255).astype(np.uint8)


def _random_actions(rng: np.random.Generator, T: int) -> np.ndarray:
    """Bridge-like 7-DoF chunk: a coherent principal direction + noise + gripper toggles."""
    direction = _unit(rng.normal(size=2))
    speed = rng.uniform(0.2, 0.7)
    a = np.zeros((T, 7), dtype=np.float64)
    a[:, 0:2] = direction[None, :] * speed + rng.normal(0, 0.08, size=(T, 2))
    a[:, 2] = rng.normal(0, 0.15, size=T)          # dz
    a[:, 3:6] = rng.normal(0, 0.06, size=(T, 3))   # rotation
    g = np.zeros(T)
    if rng.random() < 0.7:                          # a grasp partway through
        g[rng.integers(T // 3, T):] = 1.0
    a[:, 6] = g
    return a


def make_synthetic_dataset(
    n_episodes: int = 24,
    horizon: int = 12,
    size: int = 64,
    seed: int = 0,
    with_true_future: bool = False,
) -> Tuple[List[Episode], List[InstructionPerturbationSpec]]:
    """Generate a synthetic probe set + matched instruction-perturbation specs."""
    rng = np.random.default_rng(seed)
    episodes: List[Episode] = []
    specs: List[InstructionPerturbationSpec] = []
    for i in range(n_episodes):
        oi = rng.integers(len(_OBJECTS))
        li = rng.integers(len(_LOCATIONS))
        verb = _VERBS[rng.integers(len(_VERBS))]
        place = _PLACE[rng.integers(len(_PLACE))]
        instr = f"{verb} {_OBJECTS[oi]} {place} {_LOCATIONS[li]}"
        emb = embed_instruction(instr)
        actions = _random_actions(rng, horizon)
        o0 = _random_o0(rng, size)
        ep = Episode(
            episode_id=f"syn-{i:04d}",
            o0=o0,
            instruction=instr,
            instruction_emb=emb,
            actions=actions,
            meta={"object": _OBJECTS[oi], "location": _LOCATIONS[li]},
        ).validate()
        episodes.append(ep)

        # paraphrase: same object+location, different verb phrasing (semantics preserved)
        v2 = _VERBS[(int(rng.integers(len(_VERBS))) + 1) % len(_VERBS)]
        p2 = _PLACE[(int(rng.integers(len(_PLACE))) + 1) % len(_PLACE)]
        para = f"{v2} {_OBJECTS[oi]} {p2} {_LOCATIONS[li]}"
        # counterfactual: different object AND location (semantics changed)
        oj = (oi + 1 + int(rng.integers(len(_OBJECTS) - 1))) % len(_OBJECTS)
        lj = (li + 1 + int(rng.integers(len(_LOCATIONS) - 1))) % len(_LOCATIONS)
        cf = f"{verb} {_OBJECTS[oj]} {place} {_LOCATIONS[lj]}"
        specs.append(InstructionPerturbationSpec(
            episode_id=ep.episode_id,
            paraphrase=para, paraphrase_emb=embed_instruction(para).tolist(),
            counterfactual=cf, counterfactual_emb=embed_instruction(cf).tolist(),
        ))
    return episodes, specs
