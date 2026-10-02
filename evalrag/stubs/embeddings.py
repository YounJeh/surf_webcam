"""
[STUB] Modèle d'embeddings local et déterministe.

Remplace le serveur d'embeddings (endpoint /v1/embeddings). Vecteur = sac de mots et
de trigrammes de caractères (sans accents ni mots vides) projeté par hachage dans
512 dimensions, puis normalisé. Beaucoup plus pauvre qu'un vrai modèle, mais stable
d'un lancement à l'autre.

Il n'est pas nécessaire de lire ce fichier pour répondre aux questions.
"""
import hashlib
import math
import re
import unicodedata
from typing import List

DIM = 512
STOPWORDS = {
    "le", "la", "les", "l", "un", "une", "des", "de", "du", "d", "et", "ou", "a", "au", "aux",
    "en", "est", "sont", "pour", "par", "sur", "dans", "ce", "cette", "ces", "il", "elle", "je",
    "vous", "votre", "vos", "mon", "ma", "mes", "que", "qui", "quel", "quelle", "se", "s", "ne",
    "pas", "y", "on", "avec",
}


def _tokens(text: str) -> List[str]:
    norm = unicodedata.normalize("NFKD", text.lower())
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", norm) if t not in STOPWORDS]


def _embed(text: str) -> List[float]:
    vec = [0.0] * DIM
    toks = _tokens(text)
    features = list(toks)
    for t in toks:
        padded = f"<{t}>"
        features += [padded[i:i + 3] for i in range(len(padded) - 2)]
    for feat in features:
        digest = hashlib.md5(feat.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class StubEmbeddings:
    def __init__(self, model: str, base_url: str = ""):
        self.model = model
        self.base_url = base_url

    async def aembed_text(self, text: str) -> List[float]:
        return _embed(text)

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        return [_embed(t) for t in texts]
