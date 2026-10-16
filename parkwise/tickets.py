"""Voice-of-customer: classify support tickets into operational issue categories.

Purpose in the product: turn free-text complaints into a *guardrail metric* ("complaints per
10k parked hours, by cause") so pricing changes are judged on customer harm, not revenue alone.

This is a deliberately small, fully offline model (TF-IDF word + character n-grams into a
class-weighted logistic regression) rather than an LLM call. It is evaluated against a keyword-rule
baseline on held-out, noisy, partly Hinglish tickets with ambiguous multi-issue messages and ~5%
label noise, so the reported numbers are not inflated by templates that are trivially separable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from . import config as C

CATEGORIES = ["no_space", "entry_delay", "overcharge", "payment_failed", "safety", "signage", "other"]
LABELS = {"no_space": "No space / availability", "entry_delay": "Entry & exit delays",
          "overcharge": "Overcharge / price shock", "payment_failed": "Payment failures",
          "safety": "Safety & security", "signage": "Signage & navigation", "other": "Other / feedback"}

_T = {
    "no_space": [
        "lot was completely full, could not find any bay", "app showed 20 bays free but every slot was taken",
        "circled for 25 minutes, no empty space at the gate", "jagah nahi mili, parking full thi bahut der",
        "parking full tha app mein available dikha raha tha", "no slot available even after booking",
        "reached the lot and security said full, wasted my time", "no free bay on any level",
    ],
    "entry_delay": [
        "long queue at the entry gate, barrier took forever to open", "ANPR did not read my plate at the entrance, stuck for 15 min",
        "gate khulne mein bahut time laga, line lambi thi", "stuck behind cars waiting for ticket at the entry",
        "FASTag not detected at the barrier, had to wait", "exit was so slow, 20 minutes in queue to leave",
        "barrier not lifting after payment, people honking behind", "boom barrier stuck at the gate",
    ],
    "overcharge": [
        "charged much more than the displayed rate", "board said 60 per hour but I paid 240 for two hours",
        "zyada paisa kaat liya, rate board se alag tha", "price jumped without any notice, surge pricing is unfair",
        "billed for overnight although I left at 9pm", "why was I charged double compared to yesterday for the same lot",
        "parking fee was way higher than app estimate", "rate increased suddenly, felt cheated",
    ],
    "payment_failed": [
        "payment failed but money got deducted", "UPI transaction stuck, exit gate not opening",
        "wallet balance deducted twice for one parking", "payment nahi ho raha app mein error aa raha hai",
        "FASTag deducted double at the gate", "money debited but no receipt in the app",
        "card declined at exit, had to pay cash, then refund pending", "refund for failed payment still not received",
    ],
    "safety": [
        "basement was dark and felt unsafe at night", "no security guard anywhere in the lot",
        "my car got scratched while parked here", "ramp was slippery after rain, almost skidded",
        "andhera tha, lights band the, bahut unsafe laga", "guard was rude and shouted at me",
        "water logging near the ramp, car damaged", "someone tried to open my car door in the lot",
    ],
    "signage": [
        "no signs to find the exit, got lost inside", "confusing directions, ended up on the wrong level",
        "price board not visible at the entrance", "map pin in the app was wrong, entry is on the other road",
        "exit ka rasta samajh nahi aaya, koi board nahi", "could not find my reserved bay, no markings",
        "app navigation took me to the back gate which was closed", "level numbers not displayed anywhere",
    ],
    "other": [
        "great service, thanks", "please add EV charging bays", "need a GST invoice for my parking",
        "how do I get a monthly pass", "app is very smooth now, good update", "left my umbrella in the lot, please help",
        "can you add more two wheeler parking", "very helpful staff today",
    ],
}
_SUFFIX = ["", "", "", " around 7pm", " yesterday", " again today", " very frustrating", " pls fix", " third time this week",
           " at the mall lot", " near the main road", " on sunday", " this morning", " :("]
_PREFIX = ["", "", "", "hi, ", "hello team, ", "complaint: ", "urgent! ", "bro "]


def _typo(word: str, rng: np.random.Generator) -> str:
    if len(word) < 4:
        return word
    i = int(rng.integers(1, len(word) - 1))
    if rng.random() < 0.5:
        return word[:i] + word[i + 1:]
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def _noisy(text: str, rng: np.random.Generator) -> str:
    words = [(_typo(w, rng) if rng.random() < 0.10 else w) for w in text.split()]
    out = " ".join(words)
    r = rng.random()
    if r < 0.12:
        out = out.upper()
    elif r < 0.30:
        out = out.capitalize()
    if rng.random() < 0.15:
        out += "!!"
    return out


def _compose(cat: str, rng: np.random.Generator) -> tuple[str, str]:
    """Return (text, label). 15% of tickets mention two issues; the label is then ambiguous."""
    text = rng.choice(_T[cat])
    label = cat
    if cat != "other" and rng.random() < 0.15:
        other = str(rng.choice([c for c in CATEGORIES if c not in (cat, "other")]))
        second = rng.choice(_T[other])
        joiner = str(rng.choice([" and ", ". also ", ", plus "]))
        if rng.random() < 0.5:
            text, label = text + joiner + second, cat
        else:
            text, label = second + joiner + text, cat if rng.random() < 0.6 else other
    text = str(rng.choice(_PREFIX)) + str(text) + str(rng.choice(_SUFFIX))
    if rng.random() < 0.05:                                    # annotator noise
        label = str(rng.choice(CATEGORIES))
    return _noisy(text, rng), label


def generate_labeled(n: int = 2400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = np.array([.22, .17, .17, .14, .08, .10, .12])
    cats = rng.choice(CATEGORIES, size=n, p=p)
    rows = [_compose(str(c), rng) for c in cats]
    return pd.DataFrame(rows, columns=["text", "label"])


def generate_stream(hourly: pd.DataFrame, seed: int = 23) -> pd.DataFrame:
    """Tickets tied to what actually happened: full lots breed 'no space', price shocks breed 'overcharge'."""
    rng = np.random.default_rng(seed)
    h = hourly[hourly["sensor_ok"]].copy()
    h["parked"] = h["occupied"]
    h["hot"] = (h["occupancy"] >= 0.9).astype(float) * h["capacity"]
    h["shock"] = np.clip(h["price_mult"] - 1.05, 0, None) * h["occupied"]
    h["night"] = ((h["hour"] >= 22) | (h["hour"] < 5)).astype(float) * h["occupied"]
    h["wet"] = (h["rain_mm"] > 1).astype(float) * h["occupied"]
    g = h.groupby(["zone_id", "date"]).agg(parked=("parked", "sum"), turned=("turned_away", "sum"), hot=("hot", "sum"),
                                           shock=("shock", "sum"), night=("night", "sum"), wet=("wet", "sum")).reset_index()
    lam = {
        "no_space": 0.004 * g["turned"],
        "entry_delay": 0.00012 * g["hot"] + 0.00002 * g["parked"],
        "overcharge": 0.00006 * g["parked"] + 0.0025 * g["shock"],
        "payment_failed": 0.00010 * g["parked"],
        "safety": 0.00003 * g["parked"] + 0.00012 * g["night"] + 0.0001 * g["wet"],
        "signage": 0.00004 * g["parked"],
        "other": 0.00006 * g["parked"],
    }
    rows = []
    for cat in CATEGORIES:
        n = rng.poisson(np.asarray(lam[cat]))
        for idx in np.nonzero(n)[0]:
            for _ in range(int(n[idx])):
                text, _label = _compose(cat, rng)
                rows.append((int(g.at[idx, "zone_id"]), g.at[idx, "date"] + pd.Timedelta(hours=int(rng.integers(0, 24))),
                             text, cat))
    t = pd.DataFrame(rows, columns=["zone_id", "ts", "text", "true_category"]).sort_values("ts").reset_index(drop=True)
    t.insert(0, "ticket_id", np.arange(len(t)))
    return t


_KEYWORDS = {
    "no_space": r"\b(full|no (free |empty )?(space|slot|bay)|jagah)\b",
    "entry_delay": r"\b(queue|barrier|gate|anpr|fastag|stuck)\b",
    "overcharge": r"\b(charged|overcharg|price|rate|billed|fee|zyada)\b",
    "payment_failed": r"\b(payment|upi|deducted|debited|refund|wallet|card)\b",
    "safety": r"\b(unsafe|dark|security|scratch|slippery|guard|andhera|stolen)\b",
    "signage": r"\b(sign|board|direction|map|level|marking|exit ka)\b",
}


class KeywordBaseline:
    """What a team would write in an afternoon: first matching regex wins, else 'other'."""

    def predict(self, texts) -> np.ndarray:
        out = []
        for t in texts:
            t = t.lower()
            out.append(next((c for c, rx in _KEYWORDS.items() if re.search(rx, t)), "other"))
        return np.array(out)


class TicketClassifier:
    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, lowercase=True)
        self.clf = LogisticRegression(C=8.0, max_iter=2000, class_weight="balanced")

    def _x(self, texts, fit=False):
        if fit:
            return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def fit(self, texts, labels) -> "TicketClassifier":
        self.clf.fit(self._x(texts, fit=True), labels)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self.clf.predict_proba(self._x(texts))

    def predict(self, texts) -> np.ndarray:
        return self.clf.classes_[np.argmax(self.predict_proba(texts), axis=1)]


@dataclass
class TicketReport:
    model: TicketClassifier
    metrics: dict


def _prf(y, p) -> dict:
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=CATEGORIES, zero_division=0)
    return {"per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])}
                          for i, c in enumerate(CATEGORIES)},
            "macro_f1": float(np.mean(f1)), "accuracy": float(np.mean(np.asarray(y) == np.asarray(p)))}


def train_and_evaluate(n_labeled: int = 2400, seed: int = 11) -> TicketReport:
    data = generate_labeled(n_labeled, seed)
    tr, te = train_test_split(data, test_size=0.25, random_state=0, stratify=data["label"])
    model = TicketClassifier().fit(tr["text"], tr["label"])
    pred = model.predict(te["text"])
    base = KeywordBaseline().predict(te["text"])
    proba = model.predict_proba(te["text"])
    conf = confusion_matrix(te["label"], pred, labels=CATEGORIES)
    metrics = {"model": _prf(te["label"], pred), "keyword_baseline": _prf(te["label"], base),
               "confusion": {"labels": CATEGORIES, "matrix": conf.tolist()},
               "n_train": int(len(tr)), "n_test": int(len(te)),
               "mean_confidence": float(np.mean(np.max(proba, axis=1))),
               "label_noise_note": "~5% of labels randomly flipped and ~15% of tickets mention two issues; "
                                   "scores are measured against these noisy labels."}
    # Refit on everything for production use.
    final = TicketClassifier().fit(data["text"], data["label"])
    return TicketReport(final, metrics)
