"""Word lists for English, French and German used by the rule-based extractors.

All keys are lower case. Matching is case-insensitive.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Months
# ---------------------------------------------------------------------------

MONTHS = {
    # English (May is handled separately because it is also a verb)
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    # French
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "août": 8, "aout": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
    # German
    "januar": 1, "jänner": 1, "februar": 2, "märz": 3, "maerz": 3, "juni": 6,
    "juli": 7, "oktober": 10, "dezember": 12,
}

# ---------------------------------------------------------------------------
# Small numbers written as words
# ---------------------------------------------------------------------------

NUMBER_WORDS = {
    # English
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fourteen": 14, "fifteen": 15, "twenty": 20,
    # French ("six" is spelled as in English; "sept" is also an English
    # month abbreviation, the surrounding pattern decides)
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
    "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "onze": 11,
    "douze": 12, "quatorze": 14, "quinze": 15, "vingt": 20,
    # German
    "ein": 1, "eine": 1, "einen": 1, "einem": 1, "einer": 1, "zwei": 2,
    "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8,
    "neun": 9, "zehn": 10, "elf": 11, "zwölf": 12, "vierzehn": 14,
    "fünfzehn": 15, "zwanzig": 20,
}


def alternation(words) -> str:
    """Regex alternation of words, longest first so 'sept' loses to 'september'."""
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


def to_number(token: str) -> int | None:
    token = token.lower().strip()
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


# ---------------------------------------------------------------------------
# Field lexicons: canonical label -> regex fragments (lower case).
# A label is reported in English whatever the language of the enquiry.
# ---------------------------------------------------------------------------

DESTINATIONS = {
    # countries
    "India": [r"india", r"inde", r"indien"],
    "Sri Lanka": [r"sri[- ]lanka"],
    "Nepal": [r"n[eé]pal"],
    "Bhutan": [r"bhutan", r"bhoutan"],
    "Maldives": [r"maldives", r"malediven"],
    "Bangladesh": [r"bangladesh", r"bangladesch"],
    # Indian regions
    "Rajasthan": [r"rajasthan"],
    "Kerala": [r"kerala"],
    "Goa": [r"goa"],
    "Ladakh": [r"ladakh"],
    "Himachal Pradesh": [r"himachal(?: pradesh)?"],
    "Sikkim": [r"sikkim"],
    "Tamil Nadu": [r"tamil nadu"],
    "Andaman Islands": [r"andaman(?:s| islands)?", r"[iî]les andaman", r"andamanen"],
    "Golden Triangle": [r"golden triangle", r"triangle d'or", r"goldene[sn]? dreieck"],
    "Himalayas": [r"himalayas?", r"himalaja"],
    # cities and sites
    "Delhi": [r"(?:new |neu-)?delhi"],
    "Agra": [r"agra"],
    "Jaipur": [r"jaipur"],
    "Udaipur": [r"udaipur"],
    "Jodhpur": [r"jodhpur"],
    "Jaisalmer": [r"jaisalmer"],
    "Varanasi": [r"varanasi", r"b[eé]nar[eè]s"],
    "Mumbai": [r"mumbai", r"bombay"],
    "Kochi": [r"kochi", r"cochin"],
    "Alleppey": [r"alleppey", r"alappuzha"],
    "Darjeeling": [r"darjeeling"],
    "Rishikesh": [r"rishikesh"],
    "Amritsar": [r"amritsar"],
    "Ranthambore": [r"ranthamb(?:h)?ore"],
    "Khajuraho": [r"khajuraho"],
    "Hampi": [r"hampi"],
    "Kolkata": [r"kolkata", r"calcutta", r"kalkutta"],
    "Chennai": [r"chennai"],
    "Leh": [r"leh"],
    "Kathmandu": [r"kath?mandu", r"kath?mandou"],
    "Pokhara": [r"pokhara"],
    "Chitwan": [r"chitwan"],
    "Paro": [r"paro"],
    "Thimphu": [r"thimphu"],
    "Punakha": [r"punakha"],
    "Colombo": [r"colombo"],
    "Kandy": [r"kandy"],
    "Galle": [r"galle"],
    "Sigiriya": [r"sigiriya"],
    "Ella": [r"ella"],
    "Yala": [r"yala"],
    "Trincomalee": [r"trincomalee"],
}

ACCOMMODATION = {
    "overwater villa": [r"over[- ]?water (?:villa|bungalow)s?", r"water villas?",
                        r"(?:villa|bungalow)s? sur pilotis", r"wasservill(?:as?|en)",
                        r"[üu]berwasser-?(?:villa|bungalow)s?"],
    "private pool": [r"private (?:plunge )?pool", r"(?:own|plunge) pool", r"piscine priv(?:ée|ative)",
                     r"privat(?:em|en|er)? ?pool", r"eigene[mn]? pool"],
    "villa": [r"villas?", r"villa priv[ée]e"],
    "heritage or palace hotel": [r"palace hotels?", r"heritage[- ](?:hotels?|propert(?:y|ies)|havelis?)",
                                 r"havelis?", r"h[ôo]tels?[- ]palais", r"palais", r"palasthotels?",
                                 r"maharaja palace"],
    "boutique hotel": [r"boutique[- ]?hotels?", r"h[ôo]tels? boutique"],
    "5-star": [r"5[- ]?stars?", r"five[- ]stars?", r"5 [ée]toiles", r"cinq [ée]toiles", r"5[- ]sterne",
               r"f[üu]nf[- ]sterne"],
    "suite": [r"suites?"],
    "beachfront": [r"beach[- ]?front", r"on the beach", r"beach (?:resort|villa)s?", r"en bord de mer",
                   r"pieds dans l'eau", r"front de mer", r"direkt am (?:strand|meer)", r"am strand",
                   r"strandnah", r"strand(?:resort|hotel|villa)s?"],
    "tented camp": [r"tented camps?", r"luxury tents?", r"glamping", r"camps? de tentes",
                    r"tentes? de luxe", r"zeltcamps?", r"luxuszelte?"],
    "connecting rooms": [r"(?:inter)?connecting rooms", r"chambres communicantes", r"verbindungszimmer",
                         r"verbundene zimmer"],
    "family room": [r"family (?:room|suite)s?", r"chambres? familiales?", r"suites? familiales?",
                    r"familienzimmer", r"familiensuiten?"],
    "eco-lodge": [r"eco[- ]?lodges?", r"[ée]co-?lodges?", r"[öo]ko-?lodges?"],
    "houseboat": [r"house ?boats?", r"bateaux?-maisons?", r"hausboot(?:e)?"],
    "safari lodge": [r"(?:safari|jungle) lodges?", r"lodges? de safari", r"safari-?lodges?"],
}

FLIGHT_CLASSES = {
    "first class": [r"first[- ]class", r"premi[èe]re classe", r"erste[rn]? klasse"],
    "business class": [r"business[- ]?class", r"classe affaires", r"business[- ]?klasse"],
    "premium economy": [r"premium[- ]econom(?:y|ie)", r"premium [ée]co(?:nomie)?"],
    "economy class": [r"economy[- ]class", r"classe [ée]conomique", r"economy[- ]klasse"],
    "direct flights": [r"direct flights?", r"non-?stop(?: flights?)?", r"vols? directs?",
                       r"direktfl[üu]ge?", r"nonstop-?fl[üu]ge?"],
    "domestic flights": [r"domestic flights?", r"internal flights?", r"vols? int[ée]rieurs?",
                         r"inlandsfl[üu]ge?"],
    "flights not needed": [r"(?:our )?flights (?:are )?already booked",
                           r"(?:we will|we'll) book (?:our own|the) flights",
                           r"no flights (?:needed|required)", r"without flights",
                           r"vols (?:sont )?d[ée]j[àa] r[ée]serv[ée]s", r"sans (?:les )?vols",
                           r"nous r[ée]serv(?:ons|erons) (?:nos|les) vols", r"pas besoin de vols",
                           r"pas de vols [àa] pr[ée]voir",
                           r"fl[üu]ge (?:sind |haben wir )?(?:schon|bereits) gebucht", r"ohne fl[üu]ge",
                           r"fl[üu]ge buchen wir selbst", r"fl[üu]ge brauchen wir nicht"],
}

FLIGHT_ORIGINS = {
    "London": [r"london", r"londres"], "Manchester": [r"manchester"], "Edinburgh": [r"edinburgh"],
    "Paris": [r"paris"], "Lyon": [r"lyon"], "Nice": [r"nice"], "Marseille": [r"marseille"],
    "Geneva": [r"geneva", r"gen[èe]ve", r"genf"], "Zurich": [r"z[üu]rich"], "Basel": [r"basel", r"b[âa]le"],
    "Frankfurt": [r"frankfurt", r"francfort"], "Munich": [r"munich", r"m[üu]nchen"], "Berlin": [r"berlin"],
    "Hamburg": [r"hamburg", r"hambourg"], "Düsseldorf": [r"d[üu]sseldorf"],
    "Vienna": [r"vienna", r"vienne", r"wien"], "Brussels": [r"brussels", r"bruxelles", r"br[üu]ssel"],
    "Luxembourg": [r"luxembourg", r"luxemburg"], "Amsterdam": [r"amsterdam"], "New York": [r"new york"],
}
# words that make a sentence about flying (cabin classes count too)
FLIGHT_WORDS = (
    r"fl(?:y|ying|ights?)|vols?|avion|fl[üu]ge?|fliegen|departures?|abflug|"
    r"(?:business|first|economy)[- ]?(?:class|klasse)|classe affaires|premi[èe]re classe"
)
ORIGIN_JOINERS = r",|and|or|et|ou|und|oder"
ORIGIN_PREPOSITIONS = r"from|out of|depuis|au d[ée]part de|de|d'|ab|von|aus"

INTERESTS = {
    "wildlife": [r"wildlife", r"safaris?", r"tigers?", r"leopards?", r"elephants?", r"bird ?watching",
                 r"national parks?", r"faune", r"animaux sauvages", r"tigres?", r"l[ée]opards?",
                 r"[ée]l[ée]phants?", r"observation des oiseaux", r"parcs? nationa(?:l|ux)", r"tierwelt",
                 r"wildtiere?n?", r"tiger(?:n)?", r"elefanten", r"vogelbeobachtung", r"nationalparks?"],
    "culture and heritage": [r"culture", r"cultural", r"heritage sites?", r"history", r"historical",
                             r"temples?", r"forts?", r"monuments?", r"architecture", r"culturel(?:le)?s?",
                             r"patrimoine", r"histoire", r"kultur(?:ell(?:e[nr]?)?)?", r"geschichte",
                             r"tempel(?:n)?", r"sehensw[üu]rdigkeiten", r"architektur", r"festungen",
                             r"pal[äa]ste?n?"],
    "wellness and yoga": [r"yoga", r"ayurved(?:a|ic)", r"ayurv[ée]dique", r"ayurvedisch(?:e[nr]?)?",
                          r"spa", r"wellness", r"m[ée]ditation", r"bien-[êe]tre", r"retreat"],
    "beach": [r"beach(?:es)?", r"plages?", r"str[äa]nde?", r"baignade", r"badeurlaub"],
    "food and cooking": [r"food", r"cuisine", r"cooking class(?:es)?", r"culinary", r"street food",
                         r"gastronomi(?:e|que)", r"cours de cuisine", r"kochkurs(?:e)?",
                         r"kulinarisch(?:e[nr]?)?"],
    "trekking and hiking": [r"trek(?:s|king(?:reise|tour)?(?:n|en)?)?", r"hik(?:e|es|ing)", r"randonn[ée]es?",
                            r"wandern", r"wanderungen", r"bergwandern"],
    "photography": [r"photography", r"photographie", r"fotografie", r"fotografieren", r"fotoreisen?",
                    r"hobbyfotograf(?:en|in)?"],
    "diving and snorkelling": [r"diving", r"snorkel(?:l)?ing", r"plong[ée]e", r"tauchen", r"schnorcheln"],
    "shopping": [r"shopping", r"markets?", r"march[ée]s", r"artisanat", r"m[äa]rkte", r"einkaufen"],
    "golf": [r"golf"],
    "tea plantations": [r"tea (?:plantations?|estates?|gardens?)", r"plantations? de th[ée]",
                        r"teeplantagen?", r"teeg[äa]rten"],
    "luxury train": [r"luxury train", r"train journey", r"palace on wheels", r"train de luxe", r"luxuszug"],
    "festivals": [r"festivals?", r"holi", r"diwali"],
    "spirituality": [r"spiritual(?:ity)?", r"spirituel(?:le)?", r"spirituell(?:e[nr]?)?",
                     r"monaster(?:y|ies)", r"monast[èe]res?", r"kl[öo]ster"],
}

DIETARY = {
    "vegetarian": [r"vegetarians?", r"v[ée]g[ée]tarien(?:ne)?s?", r"vegetarisch(?:e[nrs]?)?", r"veggie",
                   r"vegetarier(?:in|innen)?"],
    "vegan": [r"vegan(?:e|s|er)?", r"v[ée]gan(?:e|s)?", r"v[ée]g[ée]talien(?:ne)?s?"],
    "gluten-free": [r"gluten[- ]free", r"c(?:o)?eliac", r"sans gluten", r"glutenfrei(?:e[nr]?)?",
                    r"z[öo]liakie", r"c(?:œ|oe)liaque"],
    "lactose-free": [r"lactose[- ]free", r"lactose intoleran(?:t|ce)", r"dairy[- ]free", r"sans lactose",
                     r"intol[ée]ran(?:t|te|ce) au lactose", r"laktosefrei(?:e[nr]?)?", r"laktoseintoleranz"],
    "nut allergy": [r"(?:pea)?nut allerg(?:y|ies)", r"allergic to (?:pea)?nuts",
                    r"allergi(?:e|que)s? aux (?:noix|arachides|fruits [àa] coque|cacahu[èe]tes)",
                    r"(?:erd)?nussallergie", r"allergisch gegen (?:erd)?n[üu]sse"],
    "shellfish allergy": [r"shellfish allerg(?:y|ies)", r"allergic to shellfish",
                          r"allergi(?:e|que)s? aux (?:fruits de mer|crustac[ée]s)",
                          r"schalentierallergie", r"meeresfr[üu]chteallergie",
                          r"allergisch gegen (?:schalentiere|meeresfr[üu]chte)"],
    "halal": [r"halal"],
    "kosher": [r"kosher", r"casher", r"cacher", r"koscher"],
    "jain": [r"jain"],
    "diabetic": [r"diabetic", r"diab[ée]tique", r"diabetiker(?:in)?", r"zuckerkrank"],
}

OCCASIONS = {
    "honeymoon": [r"honeymoon", r"lune de miel", r"voyage de noces", r"flitterwochen", r"hochzeitsreise"],
    "anniversary": [r"(?:wedding )?anniversary", r"anniversaire de mariage", r"\d+ ans de mariage",
                    r"noces d'(?:or|argent|perle|rubis|diamant)", r"hochzeitstag", r"jahrestag",
                    r"silberhochzeit", r"goldene[nr]? hochzeit"],
    "birthday": [r"birthday", r"anniversaire", r"(?:f[êe]te[rs]?|c[ée]l[ée]brer) (?:ses|mes|son) \d+ ans",
                 r"geburtstag", r"turns? \d{2}", r"wird \d{2}(?!\s*(?:jahre alt|personen|n[äa]chte|tage))"],
    # "im Ruhestand" (being retired) is not an occasion, so only the event counts
    "retirement": [r"retirement", r"retiring", r"(?:d[ée]part [àa] la |ma |sa |notre )retraite",
                   r"pensionierung", r"(?:eintritt )?in den ruhestand"],
    # a bare "proposal" is usually a request for a quote
    "proposal": [r"marriage proposal", r"propos(?:e|ing) to", r"demande en mariage",
                 r"demander (?:\w+ ){0,3}en mariage", r"heiratsantrag"],
    "graduation": [r"graduation", r"remise de dipl[ôo]me", r"abschlussfeier"],
    "family reunion": [r"family reunion", r"r[ée]union de famille", r"familientreffen"],
    "babymoon": [r"babymoon"],
    "Christmas or New Year": [r"christmas", r"new year(?:'s eve)?", r"no[ëe]l", r"r[ée]veillon",
                              r"nouvel an", r"weihnachten", r"silvester", r"neujahr"],
}

SPECIAL_REQUESTS = {
    "French-speaking guide": [r"french[- ]speaking guide", r"guide francophone",
                              r"guide parlant fran[çc]ais",
                              r"franz[öo]sischsprachige[nr]? (?:guide|reiseleit\w*)"],
    "German-speaking guide": [r"german[- ]speaking guide", r"guide germanophone", r"guide parlant allemand",
                              r"deutschsprachige[nr]? (?:guide|reiseleit\w*|f[üu]hrer\w*)"],
    "private guide": [r"private guide", r"personal guide", r"guide priv[ée]", r"guide personnel",
                      r"privatf[üu]hrer\w*", r"private[nr]? (?:guide|reiseleit\w*)", r"eigene[nr]? guide"],
    "private driver": [r"private (?:driver|car)", r"chauffeur(?: priv[ée])?", r"voiture priv[ée]e",
                       r"privatfahrer", r"private[nr]? fahrer", r"eigene[nr]? fahrer", r"mit fahrer"],
    "accessibility needs": [r"wheelchair(?: accessible| access)?", r"(?:reduced|limited) mobility",
                            r"fauteuil roulant", r"mobilit[ée] r[ée]duite", r"rollstuhl(?:gerecht)?",
                            r"gehbehindert", r"eingeschr[äa]nkte[nr]? mobilit[äa]t",
                            r"barrierefrei(?:heit|e[nrs]?)?"],
    "airport transfers": [r"airport transfers?", r"transferts? (?:depuis l'|de l'|à l')?a[ée]roport",
                          r"flughafentransfers?"],
    "childcare": [r"babysitt(?:er|ing)", r"baby-sitt(?:er|ing)", r"nanny", r"garde d'enfants",
                  r"kinderbetreuung", r"kids'? club", r"club enfants", r"kinderclub"],
    "butler service": [r"butler(?: service)?", r"majordome"],
    "late checkout": [r"late check-?out", r"d[ée]part tardif", r"sp[äa]te[nr]? check-?out"],
    "surprise arrangement": [r"surprise", r"[üu]berraschung"],
    "helicopter": [r"helicopter", r"h[ée]licopt[èe]re", r"hubschrauber", r"helikopter"],
    "baby cot": [r"(?:baby )?cot", r"crib", r"lit (?:de )?b[ée]b[ée]", r"babybett", r"kinderbett"],
    "short drives only": [r"no long drives", r"avoid long drives", r"pas de longs trajets",
                          r"[ée]viter les longs trajets", r"keine langen (?:auto)?fahrten"],
}

# Currency words. A bare "$" or "dollars" could be another dollar, so it is
# read as USD and flagged as an assumption.
CURRENCIES = {
    "USD": [r"us\$", r"usd", r"us dollars?", r"dollars? am[ée]ricains?", r"us-dollar"],
    "EUR": [r"€", r"eur", r"euros?"],
    "GBP": [r"£", r"gbp", r"pounds? sterling", r"pounds?", r"livres? sterling"],
    "INR": [r"₹", r"inr", r"rupees?", r"rs\.?", r"roupies?", r"rupien"],
    "CHF": [r"chf", r"francs? suisses?", r"franken", r"sfr\.?"],
    "USD?": [r"\$", r"dollars?"],
}

NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|don't|do not|doesn't|nothing|pas|sans|ni|aucune?|jamais|"
    r"kein(?:e[nmrs]?)?|nicht|ohne|nie)\b|\bnon[- ]$",
    re.IGNORECASE,
)
