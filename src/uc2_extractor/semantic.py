"""Optional model layer: match unusual wording to labels with sentence embeddings.

The rules run first. This layer only fills the label fields the rules left
empty (interests, occasion, accommodation, dietary, special requests), and
only when the match is at or above a similarity threshold. Every value it adds
is marked source="model" with its similarity as the confidence.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(Apache-2.0 according to its Hugging Face model card), quantised ONNX file
onnx/model_quint8_avx2.onnx, run with onnxruntime. The files are downloaded
once into models/uc2_minilm/ (git-ignored) with download_model().

Known limits:
- similarity models handle negation badly, so any piece of text that contains
  a negation word ("no", "pas", "kein", ...) is skipped completely
- the similarity is a matching score, not a calibrated probability
- the example phrases per label were written by hand for this project
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from . import lexicon as lx
from .schema import ClientProfile, Evidence, Provenance

REPO_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ONNX_FILE = "onnx/model_quint8_avx2.onnx"
TOKENIZER_FILE = "tokenizer.json"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "uc2_minilm"
MAX_TOKENS = 128

# Chosen on the gold set on 2026-11-03, never on a held-out set: the lowest
# threshold at which the model added no false positive beyond rules only on the
# gold set (sweep 0.40-0.93). The model added no correct value on the gold set
# at any threshold, so this setting makes it cautious. See docs/uc2_results.md.
DEFAULT_THRESHOLD = 0.88
MAX_LABELS_PER_FIELD = 3

MODEL_FIELDS = ("interests", "occasion", "accommodation", "dietary", "special_requests")

# Hand-written example phrases per label, in English, French and German. They
# describe each label in ordinary words; they are not copied from any test set.
PROTOTYPES: dict[str, dict[str, list[str]]] = {
    "interests": {
        "wildlife": ["seeing wild animals and birds in nature", "a safari to spot animals",
                     "observer les animaux sauvages et les oiseaux", "wilde Tiere und Vögel beobachten"],
        "culture and heritage": ["visiting historic monuments and old buildings", "history and architecture",
                                 "visiter des monuments historiques", "historische Bauwerke besichtigen"],
        "wellness and yoga": ["relaxing with massages, spa treatments and yoga", "being pampered at a spa",
                              "soins de bien-être et massages", "Entspannung mit Massagen und Yoga"],
        "beach": ["relaxing on a sandy beach by the sea", "lazing by the ocean",
                  "se détendre sur une plage au bord de la mer", "faulenzen am Sandstrand"],
        "food and cooking": ["tasting the local food and learning to cook", "trying the local cuisine",
                             "découvrir la cuisine locale", "die lokale Küche probieren und kochen lernen"],
        "trekking and hiking": ["walking and hiking in the hills and mountains", "long walks in the countryside",
                                "faire des randonnées en montagne", "Wandern in den Bergen"],
        "photography": ["taking photographs of landscapes and wildlife", "prendre des photos",
                        "fotografieren und Fotos machen"],
        "diving and snorkelling": ["snorkelling and diving among coral reefs", "plongée et snorkeling",
                                   "Tauchen und Schnorcheln am Riff"],
        "shopping": ["shopping in local markets and bazaars", "faire du shopping dans les marchés",
                     "auf Märkten und Basaren einkaufen"],
        "golf": ["playing a round of golf", "jouer au golf", "Golf spielen"],
        "tea plantations": ["visiting tea estates and tea gardens", "visiter des plantations de thé",
                            "Teeplantagen besuchen"],
        "luxury train": ["a journey on a luxury train", "un voyage en train de luxe", "eine Fahrt im Luxuszug"],
        "festivals": ["experiencing local festivals and celebrations", "assister à des fêtes locales",
                      "lokale Feste und Festivals erleben"],
        "spirituality": ["visiting monasteries and spiritual places", "lieux spirituels et monastères",
                         "Klöster und spirituelle Orte besuchen"],
    },
    "occasion": {
        "honeymoon": ["we have just got married and this is our honeymoon", "newly married couple",
                      "nous venons de nous marier", "wir haben gerade geheiratet"],
        "anniversary": ["celebrating our wedding anniversary", "many years of marriage",
                        "fêter notre anniversaire de mariage", "unseren Hochzeitstag feiern"],
        "birthday": ["celebrating a birthday", "a big birthday, turning fifty",
                     "fêter un anniversaire", "einen runden Geburtstag feiern"],
        "retirement": ["celebrating my retirement after years of work", "fêter mon départ à la retraite",
                       "meinen Ruhestand feiern"],
        "proposal": ["I want to ask my partner to marry me", "faire ma demande en mariage",
                     "einen Heiratsantrag machen"],
        "graduation": ["celebrating finishing university", "fêter la fin des études", "den Studienabschluss feiern"],
        "family reunion": ["a reunion of the whole family", "réunir toute la famille", "ein großes Familientreffen"],
        "babymoon": ["a last trip before our baby is born", "un dernier voyage avant la naissance du bébé",
                     "eine letzte Reise vor der Geburt unseres Babys"],
        "Christmas or New Year": ["spending Christmas and New Year away", "passer les fêtes de fin d'année",
                                  "Weihnachten und Silvester verbringen"],
    },
    "accommodation": {
        "heritage or palace hotel": ["staying in a historic palace or old fort turned into a hotel",
                                     "un hôtel dans un ancien palais", "ein Hotel in einem alten Palast"],
        "boutique hotel": ["a small, charming hotel with character", "un petit hôtel de charme",
                           "ein kleines, persönliches Hotel mit Charme"],
        "5-star": ["a five-star luxury hotel", "un hôtel de grand luxe", "ein Luxushotel mit fünf Sternen"],
        "villa": ["a private villa just for us", "une villa privée", "eine private Villa"],
        "private pool": ["a room with its own private pool", "une piscine privée", "ein eigener privater Pool"],
        "overwater villa": ["a bungalow built over the lagoon water", "un bungalow sur l'eau",
                            "ein Bungalow über dem Wasser"],
        "beachfront": ["a room right on the beach", "les pieds dans le sable", "direkt am Strand wohnen"],
        "tented camp": ["sleeping in a luxury tent at a camp", "dormir sous une tente de luxe",
                        "in einem Luxuszelt übernachten"],
        "safari lodge": ["a lodge in the jungle near a national park", "un lodge dans la jungle",
                         "eine Lodge im Dschungel"],
        "connecting rooms": ["two rooms joined by a door", "deux chambres reliées",
                             "zwei Zimmer mit Verbindungstür"],
        "family room": ["one large room for the whole family", "une chambre pour toute la famille",
                        "ein Zimmer für die ganze Familie"],
        "eco-lodge": ["an eco-friendly lodge in nature", "un hébergement écologique",
                      "eine umweltfreundliche Unterkunft"],
        "houseboat": ["staying overnight on a boat on the backwaters", "dormir sur un bateau",
                      "auf einem Boot übernachten"],
        "suite": ["a spacious suite", "une grande suite", "eine geräumige Suite"],
    },
    "dietary": {
        "vegetarian": ["vegetarian meals only", "des repas végétariens", "vegetarisches Essen"],
        "vegan": ["fully plant-based vegan food", "une alimentation végane", "rein pflanzliches veganes Essen"],
        "gluten-free": ["food that is free of gluten and wheat", "une alimentation sans blé",
                        "Essen ohne Weizen"],
        "lactose-free": ["food that is free of milk and dairy products", "une alimentation sans lait",
                         "Essen ohne Milchprodukte"],
        "nut allergy": ["a serious allergy to peanuts and nuts", "une allergie aux arachides",
                        "eine Allergie gegen Erdnüsse"],
        "shellfish allergy": ["an allergy to prawns and shellfish", "une allergie aux crevettes",
                              "eine Allergie gegen Garnelen"],
        "halal": ["halal food", "nourriture halal", "Halal-Essen"],
        "kosher": ["kosher food", "nourriture casher", "koscheres Essen"],
        "jain": ["Jain vegetarian food", "nourriture jaïne", "jainistisches Essen"],
        "diabetic": ["a diabetic diet with little sugar", "un régime pour diabétique", "Diabetikerkost"],
    },
    "special_requests": {
        "private guide": ["a personal guide just for our group", "un guide rien que pour nous",
                          "ein eigener Reiseleiter nur für uns"],
        "private driver": ["a car with our own driver", "une voiture avec chauffeur",
                           "ein Auto mit eigenem Fahrer"],
        "airport transfers": ["being picked up and met at the airport", "accueil et transfert à l'aéroport",
                              "Abholung vom Flughafen"],
        "accessibility needs": ["help for someone with difficulty walking or using a wheelchair",
                                "une personne qui marche difficilement", "eine gehbehinderte Person im Rollstuhl"],
        "childcare": ["someone to look after the children", "une garde pour les enfants",
                      "Betreuung für die Kinder"],
        "butler service": ["a personal butler", "un majordome", "ein persönlicher Butler"],
        "late checkout": ["leaving the room late on the last day", "rendre la chambre tard",
                          "das Zimmer später verlassen"],
        "surprise arrangement": ["arranging a surprise for my partner", "organiser une surprise",
                                 "eine Überraschung organisieren"],
        "helicopter": ["a helicopter flight", "un vol en hélicoptère", "ein Hubschrauberflug"],
        "baby cot": ["a small bed for our baby in the room", "un petit lit pour le bébé",
                     "ein Kinderbett für unser Baby"],
        "short drives only": ["avoiding long car journeys", "éviter les longs trajets en voiture",
                              "lange Autofahrten vermeiden"],
        "French-speaking guide": ["a guide who speaks French", "un guide qui parle français",
                                  "ein französischsprachiger Reiseleiter"],
        "German-speaking guide": ["a guide who speaks German", "un guide qui parle allemand",
                                  "ein Reiseleiter, der Deutsch spricht"],
    },
}

Embedder = Callable[[Sequence[str]], np.ndarray]  # texts -> (n, dim) unit vectors


# ---------------------------------------------------------------------------
# Model files and embeddings
# ---------------------------------------------------------------------------


def download_model(model_dir: Path = MODEL_DIR) -> Path:
    """Download the quantised ONNX model and tokenizer once (about 127 MB)."""
    from huggingface_hub import hf_hub_download

    for name in (ONNX_FILE, TOKENIZER_FILE):
        hf_hub_download(REPO_ID, name, local_dir=model_dir)
    return model_dir


class OnnxEmbedder:
    """Mean-pooled, L2-normalised sentence embeddings from the ONNX model."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        onnx_path, tok_path = model_dir / ONNX_FILE, model_dir / TOKENIZER_FILE
        if not onnx_path.exists() or not tok_path.exists():
            raise FileNotFoundError(
                f"model files not found in {model_dir}; run "
                "python -c \"from src.uc2_extractor.semantic import download_model; download_model()\""
            )
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(tok_path))
        pad_id = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_padding(pad_id=pad_id if pad_id is not None else 0)
        self.tokenizer.enable_truncation(MAX_TOKENS)
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feed)[0]  # (n, tokens, dim)
        summed = (hidden * mask[..., None]).sum(axis=1)
        vectors = summed / np.clip(mask.sum(axis=1, keepdims=True), 1, None)
        return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


@lru_cache(maxsize=1)
def default_embedder() -> OnnxEmbedder:
    return OnnxEmbedder()


# ---------------------------------------------------------------------------
# Splitting the enquiry into short pieces
# ---------------------------------------------------------------------------

_BREAK = re.compile(r"[.;:!?\n()\[\]–—]|,\s|\s[-+&]\s|\b(?:and|but|und|aber|et|mais|plus|also)\b", re.I)
_WORD = re.compile(r"\S+")


@dataclass
class Piece:
    start: int
    end: int
    text: str
    negated: bool = False  # the whole clause this piece comes from contains a negation


def split_pieces(text: str, window: int = 6, step: int = 3) -> list[Piece]:
    """Clauses of the enquiry, with long clauses also cut into word windows.

    Every piece keeps its exact position: text[start:end] == piece.text. A
    window inherits the negation of its clause, so "not interested in birds or
    animals" cannot slip through as the window "interested in birds or animals".
    """
    pieces: list[Piece] = []
    pos = 0
    bounds = [(m.start(), m.end()) for m in _BREAK.finditer(text)] + [(len(text), len(text))]
    for b_start, b_end in bounds:
        segment = text[pos:b_start]
        words = [(pos + m.start(), pos + m.end()) for m in _WORD.finditer(segment)]
        if len(words) >= 2:
            clause = text[words[0][0]:words[-1][1]]
            negated = bool(lx.NEGATION_RE.search(clause))
            pieces.append(Piece(words[0][0], words[-1][1], clause, negated))
            if len(words) > window:
                for i in range(0, len(words) - window + 1, step):
                    s, e = words[i][0], words[i + window - 1][1]
                    pieces.append(Piece(s, e, text[s:e], negated))
        pos = b_end
    return pieces


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass
class _LabelBank:
    labels: list[tuple[str, str]]  # (field, label) per prototype row
    vectors: np.ndarray


_BANKS: dict[int, _LabelBank] = {}


def _bank(embed: Embedder) -> _LabelBank:
    key = id(embed)
    if key not in _BANKS:
        rows, phrases = [], []
        for field, labels in PROTOTYPES.items():
            for label, examples in labels.items():
                for ex in examples:
                    rows.append((field, label))
                    phrases.append(ex)
        _BANKS[key] = _LabelBank(rows, embed(phrases))
    return _BANKS[key]


@dataclass
class ModelMatch:
    field: str
    label: str
    confidence: float
    piece: Piece


def find_matches(text: str, embed: Embedder, fields: Sequence[str] = MODEL_FIELDS,
                 threshold: float = DEFAULT_THRESHOLD) -> list[ModelMatch]:
    """Best label per field for each piece, kept when at or above the threshold."""
    pieces = [p for p in split_pieces(text) if not p.negated]
    if not pieces or not fields:
        return []
    bank = _bank(embed)
    sims = embed([p.text for p in pieces]) @ bank.vectors.T  # (pieces, prototypes)

    best: dict[tuple[str, str], ModelMatch] = {}
    for i, piece in enumerate(pieces):
        for field in fields:
            cols = [j for j, (f, _) in enumerate(bank.labels) if f == field]
            j = max(cols, key=lambda c: sims[i, c])
            score = float(sims[i, j])
            if score < threshold:
                continue
            key = (field, bank.labels[j][1])
            if key not in best or score > best[key].confidence:
                best[key] = ModelMatch(field, bank.labels[j][1], score, piece)

    out = []
    for field in fields:
        matches = sorted((m for m in best.values() if m.field == field), key=lambda m: -m.confidence)
        out += matches[:MAX_LABELS_PER_FIELD]
    return out


def add_model_values(profile: ClientProfile, text: str, embed: Optional[Embedder] = None,
                     threshold: float = DEFAULT_THRESHOLD) -> ClientProfile:
    """Fill label fields the rules left empty, marking each value as source="model"."""
    embed = embed or default_embedder()
    empty = [f for f in MODEL_FIELDS if not getattr(profile, f)]
    for m in find_matches(text, embed, empty, threshold):
        getattr(profile, m.field).append(m.label)
        confidence = round(min(max(m.confidence, 0.0), 1.0), 3)
        profile.evidence.append(Evidence(field=m.field, text=m.piece.text, start=m.piece.start,
                                         end=m.piece.end, source="model"))
        profile.provenance.append(Provenance(field=m.field, value=m.label, source="model",
                                             confidence=confidence))
    profile.evidence.sort(key=lambda e: (e.start, e.field))
    return profile
