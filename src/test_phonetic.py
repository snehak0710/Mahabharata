"""Checks for the phonetic name key and the resolver step built on it.

Runs without build/, without the corpus and without an API key. The resolver
checks use a stand-in Retriever built from out/entities.ttl, where every label
counts as frequent. That is a superset of what the real Retriever indexes
(it floors names at MIN_MENTIONS), so a false match that cannot happen here
cannot happen there either. The real Retriever is covered by src/test_chat.py.

    python src/test_phonetic.py
"""
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat import Retriever, entity_labels
from indic_phonetic import CORPUS_WORDS, GUARD, derive_corpus_words, indic_key, phonetic_key
from phase2_entities import norm_surface

ROOT = Path(__file__).resolve().parents[1]

SAME = [  # (what a reader types, what the corpus writes)
    ("Arjun", "Arjuna"), ("Orjun", "Arjuna"), ("Orjuno", "Arjuna"), ("arjoon", "Arjuna"),
    ("Korno", "Karna"), ("Bheem", "Bhima"), ("Bhim", "Bhima"), ("Kṛṣṇa", "Krishna"),
    ("Krsna", "Krishna"), ("Judhisthir", "Yudhishthira"), ("Yudhistir", "Yudhishthira"),
    ("Durjodhon", "Duryodhana"), ("Obhimonyu", "Abhimanyu"), ("Bidur", "Vidura"),
    ("Vishma", "Bhishma"), ("Bhisma", "Bhishma"), ("Sokuni", "Shakuni"),
    ("Dropadi", "Droupadi"), ("Draupadi", "Droupadi"), ("Gandhary", "Gandhari"),
    ("Ashwathama", "Ashvatthama"), ("Dhritorashtro", "Dhritarashtra"),
]
DIFFERENT = [  # different people who must keep different keys
    ("Drupada", "Droupadi"), ("Karna", "Kunti"), ("Bhima", "Bhishma"),
    ("Arjuna", "Abhimanyu"), ("Nakula", "Sahadeva"), ("Pandu", "Pandya"),
]


def test_variants_share_a_key():
    for typed, written in SAME:
        assert indic_key(typed) == indic_key(written), (typed, written, indic_key(typed), indic_key(written))


def test_different_people_keep_different_keys():
    for a, b in DIFFERENT:
        assert indic_key(a) != indic_key(b), (a, b, indic_key(a))


def test_guard_refuses_english_and_short_tokens():
    for w in ["part", "born", "many", "your", "fourth", "what", "tell", "son", "war"]:
        assert phonetic_key(w) is None, w
    assert phonetic_key("Orjun") == "arjun"


def labels():
    names = entity_labels(ROOT / "out/entities.ttl")
    with open(ROOT / "annotations/lineage/mahabharata_lineage.csv", newline="", encoding="utf-8") as f:
        for row in list(csv.reader(f))[1:]:
            names |= {re.sub(r"\s*\(.*?\)", "", x).strip() for x in row if x.strip()}
    return names


def test_guard_is_current():
    """CORPUS_WORDS is derived. If a new label collides with a common word, say so."""
    text = (ROOT / "out/semantic.trig").read_text(encoding="utf-8")
    passages = re.findall(r'"((?:[^"\\]|\\.){40,})"', text)
    missing = derive_corpus_words(passages, labels()) - CORPUS_WORDS
    assert not missing, "add to CORPUS_WORDS in indic_phonetic.py and indic-phonetic.js: %s" % sorted(missing)


def test_python_and_javascript_agree():
    node = shutil.which("node")
    if not node:
        print("  skipped: node not installed")
        return
    names = sorted(labels()) + [w for w, _ in SAME] + ["part", "born", "son", "Kṛṣṇa"]
    script = ("const p=require(%s);const names=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "console.log(JSON.stringify({keys:names.map(p.key),ph:names.map(p.phoneticKey),"
              "guard:[...p.GUARD].sort()}));") % json.dumps(str(ROOT / "out/indic-phonetic.js"))
    out = json.loads(subprocess.run([node, "-e", script], input=json.dumps(names), text=True,
                                    capture_output=True, check=True).stdout)
    bad = [(n, k, indic_key(n)) for n, k in zip(names, out["keys"]) if k != indic_key(n)]
    assert not bad, bad[:10]
    assert out["ph"] == [phonetic_key(n) for n in names]
    assert out["guard"] == sorted(GUARD)


def stand_in():
    """The Retriever's name tables without its parquet inputs."""
    r = Retriever.__new__(Retriever)
    r.names = sorted(labels())
    r.counts = {n: 100 for n in r.names}
    r.by_norm = {}
    for n in r.names:
        r.by_norm.setdefault(norm_surface(n), n)
    r.app = pd.DataFrame(columns=["appellation", "candidate_entity", "confidence",
                                  "evidence_count", "status", "prior_says"])
    r.appellations, r.appellations_by_key, r.epithets_of = {}, {}, {}
    r.by_key = {}
    for n in r.by_norm.values():
        r.by_key.setdefault(indic_key(n), []).append(n)
    return r


def test_resolver_reaches_spelling_variants():
    r = stand_in()
    for typed, want in [("Orjun", "Arjuna"), ("Korno", "Karna"), ("Bheem", "Bhima"),
                        ("Durjodhon", "Duryodhana"), ("Obhimonyu", "Abhimanyu"),
                        ("Judhisthir", "Yudhishthira"), ("Joydrath", "Jayadratha")]:
        got = [h["name"] for h in r.resolve("who killed " + typed)]
        assert want in got, (typed, got)


def test_resolver_drops_the_wrong_fuzzy_hits():
    """WRatio sends these to the wrong person at 90. The phonetic step answers first."""
    r = stand_in()
    for typed, want, wrong in [("sokuni", "Shakuni", "Kuni"), ("vishma", "Bhishma", "Havishmat"),
                               ("bhim", "Bhima", "Abhimanyu")]:
        got = {h["name"] for h in r.resolve("tell me about " + typed)}
        assert want in got and wrong not in got, (typed, got)


def heard(resolutions):
    """Only what the phonetic step added. The stand-in has no mention floor, so
    the existing WRatio step finds rare labels ("did" reaches Adidaya at 90)
    that the real Retriever never indexes. That is not what is tested here."""
    return [h for h in resolutions if h["via"] == "phonetic"]


def test_resolver_leaves_english_alone():
    r = stand_in()
    got = r.resolve("who was born to Kunti and what part did many of your sons play in the war")
    assert not heard(got), heard(got)


def test_no_guess():
    got = heard(stand_in().resolve("what did Napoleon Bonaparte do"))
    assert not got, got


def test_variants_are_not_merged():
    names = {h["name"] for h in stand_in().resolve("tell me about Judhisthir")}
    assert {"Yudhisthira", "Yudhishthira"} <= names, names


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok", name)
    print("%d passed" % len(tests))
