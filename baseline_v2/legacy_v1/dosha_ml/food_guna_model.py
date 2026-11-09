"""
MODEL 2 - Food text -> guna vector (10 signed guna axes + 6 rasa intensities).

Design (fast, offline, no GPU):
  1. Seed lexicon of ~95 foods with expert-style guna/rasa vectors  (EXPERT SHOULD REVIEW the numbers).
  2. Character n-gram TF-IDF retrieval + similarity-weighted k-NN regression  -> handles typos,
     Hindi/English synonyms and unseen spellings ("dhal chaval", "panner") in ~1 ms.
  3. Cooking/serving modifiers ("fried", "iced", "spicy"...) shift the retrieved vector.
  4. Multi-item meals ("dal chawal, curd and salad") are split and averaged.
  5. Optional LLM fallback for foods that retrieval cannot place (confidence below threshold);
     answers are cached. Without an API key the item is skipped rather than guessed.
"""
import json, os, re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import GUNA_AXES, RASAS

_AX = {n: i for i, n in enumerate(GUNA_AXES)}
_RS = {n: i for i, n in enumerate(RASAS)}


def E(names, **kw):
    """Entry builder. Guna kwargs are signed: guru=+.5 means guru, guru=-.5 means laghu, etc."""
    g, r = np.zeros(len(GUNA_AXES)), np.zeros(len(RASAS))
    for k, v in kw.items():
        (g.__setitem__(_AX[k], v) if k in _AX else r.__setitem__(_RS[k], v))
    return names.split("|"), g, r

LEXICON = [
    E("rice|white rice|chawal|steamed rice", guru=.2, sheeta=.3, snigdha=.2, sthira=.3, mridu=.5, sandra=.2, madhura=1),
    E("brown rice", guru=.4, snigdha=-.1, mridu=.1, madhura=.8, kashaya=.2),
    E("roti|chapati|phulka|wheat bread|atta", guru=.4, snigdha=.1, sthira=.4, sandra=.3, sheeta=.1, madhura=1),
    E("moong dal|green gram|mung", guru=-.5, sheeta=.2, snigdha=-.3, mridu=.3, madhura=.6, kashaya=.4),
    E("toor dal|arhar dal|dal|dhal|lentil soup", guru=-.1, snigdha=-.2, sheeta=-.2, madhura=.6, kashaya=.4),
    E("masoor dal|red lentil", guru=-.2, sheeta=-.3, snigdha=-.4, madhura=.5, kashaya=.5),
    E("rajma|kidney beans", guru=.7, sheeta=.2, snigdha=-.3, sthira=.5, mridu=-.3, madhura=.6, kashaya=.4),
    E("chana|chole|chickpeas|chole masala", guru=.5, sheeta=.2, snigdha=-.7, mridu=-.4, slakshna=-.2, madhura=.5, kashaya=.5),
    E("paneer|cottage cheese", guru=.8, sheeta=.4, snigdha=.5, sandra=.6, picchila=.4, madhura=1),
    E("curd|dahi|yogurt|yoghurt", guru=.7, sheeta=-.3, snigdha=.5, picchila=.5, sandra=.6, sthira=.3, amla=.6, madhura=.4),
    E("milk|doodh|dudh", guru=.5, sheeta=.5, snigdha=.6, mridu=.6, picchila=.4, sandra=-.3, madhura=1),
    E("ghee|clarified butter", guru=.3, sheeta=.2, snigdha=1, mridu=.7, slakshna=.7, sukshma=.5, sandra=-.2, madhura=1),
    E("butter|makhan", guru=.5, sheeta=.6, snigdha=.9, mridu=.8, picchila=.5, madhura=.8),
    E("chaas|buttermilk|chach|mattha", guru=-.4, sheeta=-.1, snigdha=-.2, sandra=-.7, amla=.3, madhura=.3, kashaya=.4),
    E("lassi", guru=.5, sheeta=.5, snigdha=.4, picchila=.4, sandra=-.4, madhura=.6, amla=.4),
    E("banana|kela", guru=.6, sheeta=.4, snigdha=.4, sthira=.4, mridu=.6, picchila=.5, sandra=.4, madhura=.8, amla=.2),
    E("apple|seb", guru=-.2, sheeta=.3, snigdha=-.3, mridu=-.4, madhura=.5, kashaya=.3, amla=.2),
    E("mango|aam", guru=.4, sheeta=-.2, snigdha=.4, mridu=.6, madhura=.8, amla=.2),
    E("papaya|papita", guru=-.1, sheeta=-.5, snigdha=.1, mridu=.5, madhura=.7, amla=.3),
    E("orange|mosambi|citrus|santra|kinnow", guru=-.2, sheeta=-.3, snigdha=.1, sandra=-.5, manda=-.2, amla=.7, madhura=.3),
    E("guava|amrud", guru=.1, sheeta=.3, snigdha=-.4, mridu=-.3, madhura=.4, kashaya=.4, amla=.2),
    E("watermelon|tarbooz|melon", guru=-.4, sheeta=.8, snigdha=.1, sandra=-.9, mridu=.4, madhura=1),
    E("grapes|angoor", guru=-.1, sheeta=.5, snigdha=.3, mridu=.4, sandra=-.3, madhura=.7, amla=.3),
    E("pomegranate|anar", guru=-.2, sheeta=.4, snigdha=-.2, mridu=-.2, madhura=.4, amla=.2, kashaya=.4),
    E("dates|khajoor|khajur", guru=.6, sheeta=-.3, snigdha=.6, mridu=.5, sandra=.5, madhura=1),
    E("spinach|palak|leafy greens|saag", guru=-.2, sheeta=.4, snigdha=-.5, slakshna=-.3, madhura=.2, kashaya=.5, tikta=.3),
    E("potato|aloo|alu", guru=.3, sheeta=.2, snigdha=-.4, sthira=.3, sandra=.4, madhura=.6, kashaya=.4),
    E("cauliflower|gobhi|cabbage|patta gobhi|broccoli", guru=.1, sheeta=.2, snigdha=-.5, mridu=-.3, slakshna=-.2, madhura=.3, kashaya=.5, tikta=.2),
    E("brinjal|baingan|eggplant", guru=-.2, sheeta=-.2, snigdha=-.2, kashaya=.5, tikta=.2, madhura=.3),
    E("karela|bitter gourd|bitter melon", guru=-.5, sheeta=-.2, snigdha=-.7, tikta=1),
    E("lauki|bottle gourd|ghia|tinda|tori", guru=-.5, sheeta=.6, snigdha=.1, mridu=.5, sandra=-.4, madhura=1),
    E("tomato|tamatar", guru=-.1, sheeta=-.4, snigdha=.1, sandra=-.3, manda=-.2, amla=.6, madhura=.3, kashaya=.1),
    E("onion|pyaz|pyaaz", guru=.2, sheeta=-.2, manda=-.5, katu=.6, madhura=.3),
    E("garlic|lahsun", guru=-.2, sheeta=-.8, manda=-.9, sukshma=.7, katu=.8, madhura=.2),
    E("ginger|adrak|sonth", guru=-.4, sheeta=-.8, manda=-.6, snigdha=-.3, sukshma=.5, katu=1),
    E("carrot|gajar", guru=-.1, sheeta=-.2, snigdha=-.2, mridu=-.2, madhura=.6, kashaya=.2, tikta=.2),
    E("cucumber|kheera|salad|raw vegetables|kakdi", guru=-.4, sheeta=.6, snigdha=-.2, sandra=-.7, madhura=.6, kashaya=.4),
    E("tea|chai|masala chai", guru=-.2, sheeta=-.5, snigdha=-.4, manda=-.3, sandra=-.6, kashaya=.5, tikta=.3, madhura=.2),
    E("coffee|espresso|cappuccino", guru=-.3, sheeta=-.7, snigdha=-.4, manda=-.8, sthira=-.6, sandra=-.7, tikta=.6, kashaya=.2, katu=.2),
    E("sugar|mishri|cheeni", guru=.1, sheeta=.5, snigdha=.1, mridu=.3, madhura=1),
    E("jaggery|gur|gud", guru=.4, sheeta=-.5, snigdha=.2, mridu=.3, madhura=1),
    E("honey|shahad", guru=-.3, sheeta=-.2, snigdha=-.7, sukshma=.5, sandra=.3, madhura=.7, kashaya=.3),
    E("samosa|pakora|pakoda|bhajia|fritters|kachori|vada pav", guru=.5, sheeta=-.6, snigdha=.8, manda=-.3, sandra=.5, madhura=.4, lavana=.3, katu=.3),
    E("poha|flattened rice", guru=-.4, snigdha=-.3, mridu=.3, madhura=.8, kashaya=.2),
    E("upma|semolina|suji|sooji", guru=.1, sheeta=-.3, snigdha=-.2, madhura=.7, lavana=.3),
    E("idli|idly", guru=-.1, sheeta=.2, snigdha=.1, mridu=.5, sthira=.3, picchila=.3, madhura=.6, amla=.4),
    E("dosa|dosai|uttapam", guru=-.1, sheeta=-.1, snigdha=-.3, mridu=-.2, madhura=.5, amla=.5),
    E("paratha|puri|poori|bhatura|kulcha|naan", guru=.7, sheeta=-.2, snigdha=.7, sthira=.5, sandra=.6, madhura=.8, lavana=.2),
    E("biryani|pulao|fried rice", guru=.6, sheeta=-.5, snigdha=.5, manda=-.3, sandra=.4, madhura=.4, lavana=.3, katu=.3),
    E("khichdi|khichri|kitchari", guru=-.5, sheeta=-.1, snigdha=.3, mridu=.7, sandra=-.2, madhura=.8, lavana=.2),
    E("pizza|burger|fries|fast food|nuggets|hot dog|junk food", guru=.8, sheeta=-.4, snigdha=.7, manda=-.2, sandra=.7, madhura=.4, lavana=.4, amla=.2),
    E("maggi|noodles|pasta|macaroni|ramen|chowmein|chow mein", guru=.6, sheeta=-.3, snigdha=.3, sandra=.5, madhura=.4, lavana=.4, katu=.2),
    E("bread|sandwich|toast|bun|pav", guru=.5, sheeta=.2, snigdha=-.2, sthira=.3, madhura=.8, amla=.2),
    E("oats|oatmeal|porridge|daliya|dalia", guru=.3, sheeta=.2, snigdha=.2, mridu=.4, picchila=.4, madhura=.8, kashaya=.2),
    E("egg|eggs|anda|omelette|omelet|boiled egg", guru=.5, sheeta=-.6, snigdha=.3, manda=-.3, sandra=.5, madhura=.6, lavana=.2, katu=.2),
    E("chicken|murgh|chicken curry", guru=-.1, sheeta=-.5, snigdha=-.2, madhura=.7, kashaya=.3),
    E("fish|machhli|machli|seafood|prawn", guru=.3, sheeta=-.6, snigdha=.5, picchila=.3, madhura=.6, lavana=.4),
    E("mutton|lamb|goat meat|red meat|beef|pork", guru=.9, sheeta=-.6, snigdha=.7, sthira=.6, sandra=.7, madhura=1),
    E("almonds|badam|nuts|cashew|kaju|dry fruits|pistachio", guru=.5, sheeta=-.4, snigdha=.8, sandra=.5, sukshma=.3, madhura=1),
    E("peanuts|moongfali|groundnut", guru=.4, sheeta=-.5, snigdha=.6, sandra=.4, madhura=.8, kashaya=.2),
    E("walnut|akhrot", guru=.4, sheeta=-.3, snigdha=.8, madhura=.8, kashaya=.2),
    E("ice cream|kulfi|gelato|frozen dessert", guru=.9, sheeta=1, snigdha=.8, picchila=.6, sandra=.8, sthira=.5, madhura=1),
    E("chocolate|brownie|cake|pastry|cookies|biscuit", guru=.5, sheeta=-.3, snigdha=.4, manda=-.2, sandra=.5, madhura=.6, tikta=.3, kashaya=.1),
    E("chips|namkeen|bhujia|kurkure|crisps|papad|mixture|farsan", guru=.2, sheeta=-.4, snigdha=-.3, mridu=-.6, slakshna=-.5, sthira=-.3, lavana=.6, katu=.3, madhura=.1),
    E("achaar|pickle|chutney", guru=-.2, sheeta=-.7, manda=-.7, sukshma=.4, amla=.4, lavana=.4, katu=.2),
    E("lemon|nimbu|lime", guru=-.4, sheeta=-.2, manda=-.3, sukshma=.3, sandra=-.7, amla=1),
    E("green chilli|chilli|mirchi|chili|hot sauce", guru=-.3, sheeta=-1, snigdha=-.4, manda=-1, katu=1),
    E("sprouts|sprouted moong|ankurit", guru=-.1, sheeta=.2, snigdha=-.6, mridu=-.3, madhura=.4, kashaya=.6),
    E("soup|broth|rasam|shorba", guru=-.4, sheeta=-.5, sandra=-.8, madhura=.4, lavana=.3, amla=.2),
    E("coconut water|nariyal pani|nariyal", guru=-.3, sheeta=.8, snigdha=.2, sandra=-.9, mridu=.4, madhura=.8, kashaya=.2),
    E("soda|cold drink|cola|pepsi|coke|sprite|fizzy drink|energy drink", guru=-.2, sheeta=.8, snigdha=-.2, manda=-.3, sthira=-.7, sandra=-1, slakshna=-.4, madhura=.7, amla=.3),
    E("sattu", guru=-.2, sheeta=.5, snigdha=-.8, madhura=.5, kashaya=.5),
    E("besan|besan chilla|gram flour|kadhi pakora", guru=.3, snigdha=-.6, madhura=.5, kashaya=.5),
    E("gulab jamun|jalebi|mithai|halwa|kheer|sweets|laddu|barfi|rasgulla|payasam|dessert", guru=.8, sheeta=.1, snigdha=.7, mridu=.4, picchila=.4, sandra=.6, madhura=1),
    E("pani puri|golgappa|chaat|bhel|dahi puri|sev puri", guru=-.2, sheeta=.2, snigdha=-.2, manda=-.5, sthira=-.4, sandra=-.5, amla=.4, lavana=.3, katu=.3),
    E("momos|dumplings|spring roll", guru=.4, sheeta=-.3, snigdha=.1, sthira=.3, madhura=.6, lavana=.3, katu=.1),
    E("soya|tofu|soya chunks|soy", guru=.4, sheeta=.3, snigdha=-.4, madhura=.5, kashaya=.5),
    E("mushroom|khumb", guru=.3, sheeta=.2, snigdha=-.3, madhura=.5, kashaya=.5),
    E("corn|bhutta|makai|sweet corn|popcorn", guru=-.2, snigdha=-.7, sthira=.1, mridu=-.3, madhura=.5, kashaya=.5),
    E("bajra|pearl millet|millet", guru=.2, sheeta=-.6, snigdha=-.7, mridu=-.4, madhura=.5, kashaya=.5),
    E("jowar|sorghum", guru=-.3, sheeta=.2, snigdha=-.6, madhura=.5, kashaya=.5),
    E("ragi|finger millet|nachni", guru=.2, sheeta=.4, snigdha=-.4, mridu=-.3, madhura=.4, kashaya=.6),
    E("amla|gooseberry|amla juice", guru=-.2, sheeta=.5, snigdha=-.3, amla=.7, kashaya=.2, tikta=.1),
    E("haldi doodh|turmeric milk|golden milk", guru=.3, sheeta=-.5, snigdha=.3, sandra=-.4, madhura=.6, tikta=.2, katu=.2),
    E("curd rice|thayir sadam|dahi chawal", guru=.6, sheeta=.3, snigdha=.4, picchila=.5, madhura=.5, amla=.5),
    E("water|pani|plain water|warm water", guru=-.5, sheeta=.3, sandra=-1),
    E("juice|fruit juice|packaged juice|shake|smoothie|milkshake", guru=-.3, sheeta=.4, snigdha=.2, sandra=-.9, madhura=.6, amla=.4),
    E("dhokla|khaman", guru=-.3, snigdha=-.2, madhura=.5, amla=.5),
    E("sabzi|vegetable curry|mixed veg|subzi|veg curry", guru=-.1, sheeta=-.2, snigdha=-.1, madhura=.4, kashaya=.3, katu=.2, lavana=.1),
    E("fruit|fruits|fruit salad|seasonal fruit", guru=-.1, sheeta=.3, snigdha=.1, mridu=.4, sandra=-.4, madhura=.7, amla=.3),
]

# cooking / serving modifiers: token -> (guna deltas, rasa deltas)
def _m(**kw):
    g, r = np.zeros(len(GUNA_AXES)), np.zeros(len(RASAS))
    for k, v in kw.items():
        (g.__setitem__(_AX[k], v) if k in _AX else r.__setitem__(_RS[k], v))
    return g, r

MODIFIERS = {
    ("fried", "deep fried", "deep-fried", "tala", "crispy"): _m(guru=.3, snigdha=.4, sheeta=-.3, sandra=.2),
    ("raw", "uncooked"): _m(sheeta=.2, snigdha=-.2, mridu=-.2),
    ("cold", "iced", "chilled", "frozen", "fridge", "ice"): _m(sheeta=.6),
    ("hot", "warm", "steaming", "piping"): _m(sheeta=-.4),
    ("spicy", "masala", "tikha", "teekha"): _m(sheeta=-.6, manda=-.6, katu=.6),
    ("sweet", "sugary", "meetha", "sugar"): _m(madhura=.4, guru=.2),
    ("sour", "tangy", "khatta"): _m(amla=.4),
    ("salty", "namkeen"): _m(lavana=.3),
    ("boiled", "steamed", "stewed"): _m(guru=-.2, snigdha=-.1),
    ("roasted", "baked", "dry", "grilled", "toasted", "tandoori"): _m(snigdha=-.3, sheeta=-.2),
    ("creamy", "cheesy", "cheese", "buttery", "oily", "greasy", "rich"): _m(guru=.3, snigdha=.4, sandra=.2),
    ("large", "heavy", "big", "huge", "extra"): _m(guru=.2),
    ("light", "small", "little"): _m(guru=-.2),
}
_MOD_LOOKUP = {tok: vec for toks, vec in MODIFIERS.items() for tok in toks}
_SPLIT = re.compile(r"\s*(?:,|;|\+|&|\band\b|\bwith\b|\baur\b|\bplus\b|\n)\s*", re.I)


class FoodGunaModel:
    def __init__(self, sim_threshold=0.45, k=3, temperature=0.08, cache_path=None, use_llm=True):
        self.names, self.G, self.R = [], [], []
        self.alias_owner, self.aliases = [], []
        for i, (als, g, r) in enumerate(LEXICON):
            self.names.append(als[0]); self.G.append(g); self.R.append(r)
            for a in als:
                self.aliases.append(a); self.alias_owner.append(i)
        self.G, self.R = np.array(self.G), np.array(self.R)
        self.alias_owner = np.array(self.alias_owner)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(self.aliases)
        self.A = self.vec.transform(self.aliases)
        self.thr, self.k, self.T = sim_threshold, k, temperature
        self.use_llm = use_llm
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path and self.cache_path.exists() else {}

    # -- single item
    def _retrieve(self, text):
        sims = (self.A @ self.vec.transform([text]).T).toarray().ravel()
        best = np.zeros(len(self.names))
        np.maximum.at(best, self.alias_owner, sims)
        top = np.argsort(-best)[: self.k]
        w = np.exp((best[top] - best[top].max()) / self.T); w /= w.sum()
        return (w @ self.G[top], w @ self.R[top], float(best[top[0]]), self.names[top[0]])

    def _llm(self, text):
        if text in self.cache:
            c = self.cache[text]; return np.array(c["g"]), np.array(c["r"])
        if not self.use_llm or not os.getenv("ANTHROPIC_API_KEY"):
            return None
        try:
            import anthropic
            msg = anthropic.Anthropic().messages.create(
                model=os.getenv("DOSHA_LLM_MODEL", "claude-sonnet-5-5"), max_tokens=300,
                system=("You are an Ayurvedic food-property assistant. Reply with ONLY JSON: "
                        '{"gunas": {...}, "rasas": {...}}. gunas keys %s each in [-1,1] (negative = the opposite '
                        "quality: %s); rasas keys %s each in [0,1]." % (list(GUNA_AXES), "laghu,ushna,ruksha,tikshna,chala,kathina,vishada,khara,sthula,drava", list(RASAS))),
                messages=[{"role": "user", "content": text}])
            d = json.loads(msg.content[0].text)
            g = np.clip([d["gunas"].get(a, 0) for a in GUNA_AXES], -1, 1); r = np.clip([d["rasas"].get(a, 0) for a in RASAS], 0, 1)
            self.cache[text] = {"g": g.tolist(), "r": r.tolist()}
            if self.cache_path: self.cache_path.write_text(json.dumps(self.cache))
            return g, r
        except Exception:
            return None

    def _one(self, item):
        toks = re.findall(r"[a-zA-Z\-]+", item.lower())
        mod_g, mod_r, base = np.zeros(len(GUNA_AXES)), np.zeros(len(RASAS)), []
        for t in toks:
            if t in _MOD_LOOKUP and len(toks) > 1:
                mod_g += _MOD_LOOKUP[t][0]; mod_r += _MOD_LOOKUP[t][1]
            else:
                base.append(t)
        base_txt = " ".join(base) or item.lower()
        g, r, conf, match = self._retrieve(base_txt)
        if conf < 0.95 and len(base) >= 2:          # compound ("dal chawal") or unknown word ("paneer tikka")
            parts = [p for p in (self._retrieve(w) for w in base) if p[2] >= (0.7 if conf >= self.thr else 0.5)]
            if parts and (len(parts) == len(base) or conf < self.thr):
                g, r = np.mean([p[0] for p in parts], 0), np.mean([p[1] for p in parts], 0)
                conf, match = min(p[2] for p in parts), "+".join(p[3] for p in parts)
        source = "lexicon"
        if conf < self.thr:
            out = self._llm(item.lower())
            if out is None:
                return None
            (g, r), conf, match, source = out, 0.6, "llm", "llm"
        return {"item": item.strip(), "match": match, "confidence": conf, "source": source,
                "gunas": np.clip(g + mod_g, -1, 1), "rasas": np.clip(r + mod_r, 0, 1)}

    # -- public
    def predict(self, text):
        """text may be a whole meal. Returns dict(gunas(10), rasas(6), items=[...]) or None if nothing recognised."""
        items = [self._one(p) for p in _SPLIT.split(text) if p.strip()]
        items = [i for i in items if i]
        if not items:
            return None
        return {"gunas": np.mean([i["gunas"] for i in items], 0),
                "rasas": np.mean([i["rasas"] for i in items], 0), "items": items}
