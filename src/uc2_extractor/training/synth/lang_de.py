"""German wording for the synthetic enquiry generator. Same markup as lang_en.py.

Phrases for accommodation, flights and interests are written in the accusative, and the
sentence frames take the accusative ("Wir suchen {x}.", "Bitte {x} einplanen."), so no phrase
has to change its case. Frames work for both "Sie" and "ihr" by avoiding direct address.
All wording here was written for this project. It has not been checked by a native speaker.
"""

import re

from .parts import S, T

CODE = "de"
JOIN = (", ", " und ")
SHARE_PREPOSITION = True  # "nach Delhi, Agra und Jaipur"
PARTNERS = [
    {"partner": "mein Mann", "partner_acc": "meinen Mann", "partner_dat": "meinem Mann"},
    {"partner": "meine Frau", "partner_acc": "meine Frau", "partner_dat": "meiner Frau"},
    {"partner": "mein Partner", "partner_acc": "meinen Partner", "partner_dat": "meinem Partner"},
    {"partner": "meine Partnerin", "partner_acc": "meine Partnerin", "partner_dat": "meiner Partnerin"},
    {"partner": "mein Verlobter", "partner_acc": "meinen Verlobten", "partner_dat": "meinem Verlobten"},
    {"partner": "meine Verlobte", "partner_acc": "meine Verlobte", "partner_dat": "meiner Verlobten"},
    {"partner": "mein Lebensgefährte", "partner_acc": "meinen Lebensgefährten", "partner_dat": "meinem Lebensgefährten"},
]
PROPOSAL_PARTNERS = [PARTNERS[2], PARTNERS[3], PARTNERS[6]]
NOT_SOLO = re.compile(r"\b(wir|uns|unser\w*)\b", re.IGNORECASE)
NUM = {1: "ein", 2: "zwei", 3: "drei", 4: "vier", 5: "fünf", 6: "sechs", 7: "sieben", 8: "acht", 9: "neun",
       10: "zehn", 11: "elf", 12: "zwölf", 13: "dreizehn", 14: "vierzehn", 15: "fünfzehn", 16: "sechzehn",
       17: "siebzehn", 18: "achtzehn", 19: "neunzehn", 20: "zwanzig", 21: "einundzwanzig"}
MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
          "November", "Dezember"]
MONTHS_ABBR = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]

DATE_FORMS = {
    "month": [("{m}", ["im {D}", "irgendwann im {D}", "ab {D}"]), ("{m} {y}", ["im {D}", "ab {D}"])],
    "early": [("Anfang {m}", ["{D}", "ab {D}"]), ("Anfang {m} {y}", ["{D}"])],
    "mid": [("Mitte {m}", ["{D}", "ab {D}", "etwa {D}"])],
    "late": [("Ende {m}", ["{D}", "ab {D}"]), ("die letzte Woche im {m}", ["{D}", "für {D}"])],
    "range_month": [("{d}. bis {d2}. {m}", ["vom {D}"]), ("{d}.-{d2}. {m}", ["{D}", "vom {D}"]),
                    ("{d}. und {d2}. {m}", ["zwischen dem {D}"]), ("{d}. bis {d2}. {m} {y}", ["vom {D}"])],
    "range_cross": [("{d}. {m} bis {d2}. {m2}", ["vom {D}"]), ("{d}. {ma} - {d2}. {ma2}", ["{D}"])],
    "single": [("{d}. {m}", ["am {D}", "ab dem {D}", "um den {D}"]), ("{d}. {m} {y}", ["am {D}"])],
    "numeric_ambiguous": [("{d}/{mo}", ["am {D}", "ab dem {D}"]), ("{d}/{mo}/{y}", ["am {D}"])],
    "numeric_clear": [("{d}.{mo2}.{y}", ["am {D}", "ab dem {D}"]), ("{y}-{mo2}-{dd}", ["am {D}"])],
    "month_range": [("{m} bis {m2}", ["von {D}", "{D}"]), ("{m}/{m2}", ["im {D}"]), ("{ma}/{ma2}", ["{D}"])],
    "month_choice": [("{m} oder {m2}", ["im {D}", "entweder im {D}"]), ("{ma} oder {ma2}", ["im {D}"])],
}
DURATION_FORMS = {
    "nights": ["{n} Nächte", "{w} Nächte", "{n} Übernachtungen"],
    "days": ["{n} Tage", "{w} Tage"],
    "weeks": ["{w} Wochen", "{n} Wochen"],
    "one_week": ["eine Woche", "sieben Tage"],
    "two_weeks": ["zwei Wochen", "14 Tage", "vierzehn Tage"],
}
CURRENCY_FORMS = {
    "USD": ["{a} Dollar", "{a} USD", "USD {a}", "{a} $"],
    "EUR": ["{a} Euro", "{a} €", "€ {a}", "EUR {a}", "{a} EUR"],
    "GBP": ["{a} Pfund", "{a} GBP", "£{a}"],
    "CHF": ["{a} Franken", "CHF {a}", "{a} CHF", "{a} Fr."],
    "INR": ["{a} Rupien", "INR {a}", "{a} INR"],
}
LAKH_FORMS = ["{l} Lakh Rupien", "INR {l} Lakh"]
THOUSAND_SEP = "."
K_FORMS = ["{k}k", "{k} Tsd.", "{k} Tausend"]
K_WORDS = {3: "drei", 4: "vier", 5: "fünf", 6: "sechs", 8: "acht", 9: "neun", 10: "zehn", 12: "zwölf", 15: "fünfzehn",
           18: "achtzehn", 20: "zwanzig", 25: "fünfundzwanzig", 30: "dreißig", 40: "vierzig", 50: "fünfzig",
           75: "fünfundsiebzig", 100: "hundert"}
WORD_AMOUNT = "{w}tausend"
BUDGET_SHAPES = {
    "about": ["etwa {x}", "ungefähr {x}", "rund {x}", "circa {x}", "ca. {x}", "so um die {x}"],
    "exact": ["{x}"],
    "up_to": ["bis zu {x}", "maximal {x}", "höchstens {x}", "{x} maximal", "nicht mehr als {x}", "max. {x}"],
    "at_least": ["mindestens {x}", "{x} oder mehr", "{x} und mehr"],
    "range": ["zwischen {x} und {y}", "{x} bis {y}", "{n}-{y}"],
}
BASIS = {"per_person": [" pro Person", " p.P.", " pro Kopf", " je Person"],
         "per_trip": [" insgesamt", " für die ganze Reise", " gesamt", " alles zusammen"]}
AGE_JOIN = (", ", " und ")

PARTY = {
    "couple": {
        "subject": ["[[ADULTS|{partner} und ich]]", "[[ADULTS|{partner} und ich]]", "[[PARTY|Wir zwei]]"],
        "np": ["[[ADULTS|zwei Erwachsene]]", "[[PARTY|zwei Personen]]", "ein [[ADULTS|Paar]]", "ein [[ADULTS|Ehepaar]]",
               "[[ADULTS|2 Erwachsene]]", "[[PARTY|zwei Reisende]]"],
        "note": ["[[ADULTS|2 Erw.]]", "[[PARTY|2 Pers.]]", "[[ADULTS|Paar]]", "[[PARTY|2 Personen]]", "[[ADULTS|2 Erwachsene]]"],
    },
    "solo": {
        "subject": [],
        "np": ["[[PARTY|eine Person]]", "[[PARTY|mich allein]]", "[[ADULTS|einen Erwachsenen]]"],
        "note": ["[[PARTY|1 Pers.]]", "[[ADULTS|1 Erw.]]", "[[PARTY|allein]]"],
    },
    "family": {
        "subject": ["[[ADULTS|{partner} und ich]] mit [[CHILDREN|unseren {c} Kindern]]",
                    "[[ADULTS|{partner} und ich]] und [[CHILDREN|unsere {c} Kinder]]"],
        "np": ["[[ADULTS|{a} Erwachsene]] und [[CHILDREN|{c} Kinder]]", "[[ADULTS|{a} Erwachsene]] mit [[CHILDREN|{c} Kindern]]",
               "eine [[PARTY|Familie mit {t} Personen]]", "[[PARTY|{t} Personen]], davon [[CHILDREN|{c} Kinder]]"],
        "note": ["[[ADULTS|{a} Erw.]] + [[CHILDREN|{c} Kinder]]", "[[ADULTS|{a} Erw.]] / [[CHILDREN|{c} Ki.]]",
                 "[[PARTY|Familie, {t} Pers.]]"],
        "one_child": ["[[ADULTS|{a} Erwachsene]] mit [[CHILDREN|einem Kind]]",
                      "[[ADULTS|{a} Erwachsene]] mit [[CHILDREN|unserem Sohn]]",
                      "[[ADULTS|{a} Erwachsene]] mit [[CHILDREN|unserer Tochter]]"],
        "single_parent": ["[[ADULTS|ein Elternteil]] mit [[CHILDREN|{c} Kindern]]",
                          "[[ADULTS|eine alleinerziehende Mutter]] mit [[CHILDREN|{c} Kindern]]"],
    },
    "friends": {
        "subject": ["Wir, [[PARTY|{t} alte Schulfreunde]],", "Wir, [[PARTY|{t} Freunde aus Studienzeiten]],",
                    "Wir, [[PARTY|{t} langjährige Freunde]],"],
        "np": ["[[PARTY|{t} Freunde]]", "eine [[PARTY|Gruppe von {t} Freunden]]", "[[ADULTS|{t} Erwachsene]]", "[[PARTY|{t} Personen]]"],
        "note": ["[[PARTY|{t} Freunde]]", "[[PARTY|{t} Pers.]]", "[[ADULTS|{t} Erw.]]"],
    },
    "two_couples": {
        "subject": ["Wir, [[ADULTS|{p} Paare]],", "Wir, [[ADULTS|{p} befreundete Ehepaare]],"],
        "np": ["[[ADULTS|{p} Paare]]", "[[PARTY|{t} Personen]]", "[[ADULTS|{t} Erwachsene]]"],
        "note": ["[[ADULTS|{p} Paare]]", "[[PARTY|{t} Pers.]]"],
    },
    "multigen": {
        "subject": [],
        "np": ["[[PARTY|{t} Personen]] aus drei Generationen", "[[ADULTS|{a} Erwachsene]] und [[CHILDREN|{c} Enkelkinder]]",
               "[[PARTY|{t} Personen]], Großeltern eingeschlossen"],
        "note": ["[[PARTY|{t} Pers.]]", "[[ADULTS|{a} Erw.]] + [[CHILDREN|{c} Kinder]]"],
    },
    "group": {
        "subject": [],
        "np": ["eine [[PARTY|Gruppe von {t} Personen]]", "[[PARTY|{t} Personen]]", "[[PARTY|{t} Teilnehmer]]",
               "[[PARTY|{t} Gäste]]"],
        "note": ["[[PARTY|{t} Pers.]]", "[[PARTY|Gruppe von {t}]]"],
    },
}

SALUTATIONS = {
    "formal": ["Sehr geehrte Damen und Herren,", "Sehr geehrtes Team,", "Guten Tag,", "Liebes Team von Kenexoft,"],
    "neutral": ["Hallo,", "Guten Tag,", "Guten Abend,", "Hallo zusammen,", "Grüezi,"],
    "casual": ["hallo", "hi", "Hey,", "moin", "servus", ""],
}
SIGN_OFFS = {
    "formal": ["Mit freundlichen Grüßen", "Freundliche Grüße", "Beste Grüße", "Vielen Dank im Voraus"],
    "neutral": ["Danke", "Viele Grüße", "Liebe Grüße", "Herzliche Grüße", "Schöne Grüße"],
    "casual": ["danke", "lg", "vg", "Gruß", ""],
}
NAMES = ["Annika Brandt", "Lorenz Hofer", "Friederike Lang", "Matthias Kessler", "Theresa Vogt", "Benedikt Sommer",
         "Wiebke Ahrens", "Konrad Engel"]
EXTRA_LINES = ["Von meinem iPad gesendet", "Vom Handy geschrieben, bitte Tippfehler entschuldigen", "Mobil: +49 170 0000000"]

OPENERS = [
    "Wir sind über einen Artikel in einem Magazin auf Sie gestoßen, und was wir gelesen haben, hat uns gefallen.",
    "Ein Kollege, der letztes Jahr mit Ihnen gereist ist, hat mir Ihren Namen gegeben.",
    "Ich habe heute Abend Ihre Website durchgelesen und schreibe Ihnen einfach direkt.",
    "Freunde empfehlen Sie uns immer wieder, also melde ich mich jetzt.",
    "Wir stehen noch ganz am Anfang der Planung und wären für Rat dankbar.",
    "Wir planen zum ersten Mal mit einem Spezialisten, bitte haben Sie etwas Geduld mit uns.",
    "Ich versuche, mich kurz zu fassen, auch wenn mir das selten gelingt.",
    "Wir reden schon lange davon, und jetzt scheint der richtige Moment gekommen zu sein.",
    "Entschuldigen Sie bitte die Länge dieser Nachricht.",
]
FILLERS = [
    "Wir haben keine Eile und bleiben lieber länger an einem Ort, als jeden Abend das Hotel zu wechseln.",
    "Das Tempo ist uns wichtiger als die Zahl der Sehenswürdigkeiten.",
    "Wir starten gern früh und halten uns die Nachmittage frei.",
    "Wir sehen lieber weniger, dafür gründlich.",
    "Für Qualität zahlen wir gern, aber Protz mögen wir nicht.",
    "Menschenmassen mag keiner von uns.",
    "Guter Kaffee am Morgen wäre ein Pluspunkt.",
    "Europa kennen wir recht gut, aber so weit im Osten waren wir noch nie.",
    "Die Hitze macht uns wenig aus.",
    "Zuverlässiges WLAN wäre nicht schlecht, weil einer von uns vielleicht nach der Arbeit sehen muss.",
]
CLOSERS = [
    "Was würden Sie vorschlagen?",
    "Bitte schicken Sie mir einen groben Entwurf mit einer Kostenschätzung.",
    "Ich freue mich auf Ihre Antwort.",
    "Gern können wir auch telefonieren, wenn das einfacher ist.",
    "Sagen Sie mir gern, wenn noch etwas fehlt.",
    "Ein erster Vorschlag per E-Mail wäre wunderbar.",
]
VAGUE = ["Bitte schicken Sie mir ein paar Ideen.", "Was würden Sie empfehlen?", "Wo sollte man anfangen?",
         "Können Sie helfen?", "Vorschläge sind willkommen."]

INTRO = [
    "{subject} möchten gern {dur} {dest} verbringen.",
    "{subject} denken {date} an eine Reise {dest_to}.",
    "{subject} planen {dur} {dest} {date}.",
    "Wir sind {party} und träumen von einer Reise {dest_to}.",
    "Wir sind {party} und suchen {dur} {dest} {date}.",
    "Ich organisiere eine Reise {dest_to} für {party}.",
    "Könnten Sie etwas {dest} für {party} zusammenstellen?",
    "Könnten Sie uns bei einer Reise {dest_to} {date} helfen?",
    "Wir haben uns {dest_obj} in den Kopf gesetzt.",
    "Was würden Sie für {dur} {dest} vorschlagen?",
    "Wir hoffen, {date} {dest_to} zu kommen.",
    "Für {party} suche ich {dur} {dest}.",
    "Wir möchten unbedingt {dest_obj} sehen und könnten {date} weg.",
]
INTRO_SOLO = [
    "Ich möchte gern {dur} {dest} verbringen.",
    "Ich denke {date} an eine Reise {dest_to}.",
    "Ich plane {dur} {dest} {date}.",
    "Ich hoffe, {date} {dest_to} zu kommen.",
    "Was würden Sie für {dur} {dest} vorschlagen?",
    "Ich habe mir {dest_obj} in den Kopf gesetzt.",
]
SENT_SOLO = {
    "dest": ["Ich habe {dest_obj} im Blick.", "Der Plan ist eine Reise {dest_to}."],
    "dur": ["Ich habe {dur} Zeit.", "Ich kann {dur} wegfahren.", "Reisedauer: {dur}."],
    "date": ["Ich kann {date} reisen.", "Am besten passt es mir {date}.", "Ich würde {date} aufbrechen."],
    "date_dur": ["Ich möchte {date} los und {dur} bleiben."],
    "party": ["Die Reise ist für {party}.", "Es geht um {party}."],
    "budget": ["Mein Budget: {budget}.", "Ich habe {budget} eingeplant.", "Ausgeben möchte ich {budget}."],
    "accommodation": ["Ich suche {x}.", "Gern hätte ich {x}."],
    "flights": ["Ich brauche {x}.", "Bitte {x} einplanen."],
    "flights_origin": ["Ich wünsche mir {x} ab {origin}."],
    "origin": ["Ich würde ab {origin} fliegen."],
    "interests": ["Ich wünsche mir {x}.", "Gern hätte ich {x}.", "Ich interessiere mich sehr für {x}."],
}
SENT = {
    "dest": ["Wir haben {dest_obj} im Blick.", "Der Plan ist eine Reise {dest_to}.",
             "Wir kommen immer wieder auf {dest_obj} zurück.", "Unsere erste Wahl wäre eine Reise {dest_to}."],
    "dur": ["Wir haben {dur} Zeit.", "Wir können {dur} wegfahren.", "Wir denken an {dur} insgesamt.",
            "Reisedauer: {dur}.", "Wir könnten {dur} einrichten."],
    "date": ["Wir können {date} reisen.", "Am besten passt es uns {date}.", "Wir haben {date} Zeit.",
             "Wir würden {date} aufbrechen.", "Die Termine sind noch offen, am ehesten {date}."],
    "date_dur": ["Wir möchten {date} los und {dur} bleiben.", "Grob gesagt: {dur} {date}."],
    "party": ["Wir sind {party}.", "Wir werden {party} sein.", "Die Reise ist für {party}.", "Mitreisende: {party}."],
    "budget": ["Budget: {budget}.", "Wir haben {budget} eingeplant.", "Ausgeben möchten wir {budget}.",
               "Wir rechnen mit {budget}.", "Preislich denken wir an {budget}."],
    "accommodation": ["Wir suchen {x}.", "Gern hätten wir {x}.", "Bei der Unterkunft wünschen wir uns {x}.",
                      "Bitte {x} einplanen."],
    "flights": ["Wir brauchen {x}.", "Bitte {x} einplanen.", "Bei den Flügen wünschen wir uns {x}."],
    "flights_origin": ["Wir wünschen uns {x} ab {origin}.", "Bitte {x} ab {origin} kalkulieren."],
    "origin": ["Wir würden ab {origin} fliegen.", "Abflug wäre in {origin}.", "Wir starten in {origin}."],
    "interests": ["Wir wünschen uns {x}.", "Gern hätten wir {x}.", "Wir freuen uns besonders auf {x}.",
                  "Wir interessieren uns sehr für {x}."],
    "ages_children": ["Die Kinder sind {ages}.", "Die Kinder sind dann {ages} Jahre alt.", "Alter der Kinder: {ages}."],
    "ages_one": ["Unser Kind ist {ages}.", "Unser Kind ist dann {ages} Jahre alt."],
    "ages_adults": ["Wir sind {ages}.", "Wir sind {ages} Jahre alt."],
    "age_solo": ["Ich bin {ages}.", "Ich bin {ages} Jahre alt."],
}
NOTE_OPENERS = ["Reiseanfrage:", "Anfrage -", "Idee:", "Angebot bitte:", ""]
NOTE_CLOSERS = ["danke", "bitte Vorschläge", "Ideen?", "bitte melden", ""]
NOTE_LABELS = {"budget": ["Budget", "Budget:", "Rahmen", ""]}
NOTE_ORIGIN = ["ab ", "Abflug ", "von "]
ALT_BUDGET = [" (also rund {x})", " (ungefähr {x} zum aktuellen Kurs)"]

ACCOMMODATION = {
    "overwater villa": {"np": ["einen [Bungalow über der Lagune]", "einen [Wasserbungalow]", "eine [Villa auf Stelzen]"],
                        "note": ["Wasservilla", "Overwater-Bungalow"]},
    "private pool": {"np": ["ein Zimmer mit [eigenem Pool]", "einen [kleinen Pool nur für uns]", "einen [Privatpool]"],
                     "note": ["Privatpool", "eigener Pool"]},
    "villa": {"np": ["eine [Villa]", "eine [ganze Villa für uns allein]", "ein [privates Haus mit Personal]"], "note": ["Villa"]},
    "heritage or palace hotel": {"np": ["eine [restaurierte Haveli]", "eine [ehemalige Fürstenresidenz]",
                                        "einen [umgebauten Palast]", "ein [historisches Anwesen]"],
                                 "note": ["Palasthotels", "Haveli"]},
    "boutique hotel": {"np": ["ein [Boutique-Hotel]", "ein [kleines, stilvolles Hotel]", "ein [winziges Designhotel]",
                              "[familiäre Häuser mit höchstens zehn Zimmern]"], "note": ["Boutique"]},
    "5-star": {"np": ["[Fünf-Sterne-Hotels]", "[5-Sterne-Häuser]", "[Fünf-Sterne-Komfort]"], "note": ["5*", "5 Sterne"]},
    "suite": {"np": ["eine [Suite]", "eine [Junior-Suite]", "eine [richtige Suite]"], "note": ["Suite"]},
    "beachfront": {"np": ["ein Zimmer [direkt am Sand]", "etwas [nur ein paar Schritte vom Meer]",
                          "ein Zimmer [in erster Strandreihe]"], "note": ["Strandlage", "erste Reihe am Meer"]},
    "tented camp": {"np": ["ein [luxuriöses Zeltcamp]", "[Safarizelte]", "ein bisschen [Glamping]"],
                    "note": ["Zeltcamp", "Glamping"]},
    "connecting rooms": {"np": ["[Verbindungszimmer]", "[zwei Zimmer mit Tür dazwischen]"], "note": ["Verbindungszi."]},
    "family room": {"np": ["ein [Familienzimmer]", "ein [großes Zimmer für uns alle]"], "note": ["Familienzimmer"]},
    "eco-lodge": {"np": ["eine [Öko-Lodge]", "eine [solarbetriebene Lodge]", "etwas [Nachhaltiges mit wenig Eingriff in die Natur]"],
                  "note": ["Öko-Lodge"]},
    "houseboat": {"np": ["ein [Hausboot]", "eine [traditionelle Reisbarke]", "eine Nacht auf einem [Kettuvallam]"],
                  "note": ["Hausboot"]},
    "safari lodge": {"np": ["eine [Safari-Lodge]", "eine [Dschungel-Lodge]", "eine [Lodge am Rand des Nationalparks]"],
                     "note": ["Safari-Lodge"]},
}
FLIGHTS = {
    "business class": {"np": ["[Business Class]", "[Liegesitze]", "[flache Betten vorne im Flieger]"],
                       "note": ["Business", "Biz Class"]},
    "first class": {"np": ["[First Class]", "[erste Klasse]"], "note": ["First Class", "F Class"]},
    "premium economy": {"np": ["[Premium Economy]", "[Sitze in der Premium Economy]"], "note": ["Prem. Eco"]},
    "economy class": {"np": ["[Economy-Sitze]", "[normale Economy]"], "note": ["Economy", "Eco"]},
    "direct flights": {"np": ["[Direktflüge]", "[Nonstop-Flüge]", "[Flüge ohne Zwischenstopp]"], "note": ["direkt", "nur Nonstop"]},
    "domestic flights": {"np": ["[Inlandsflüge]", "[Flüge innerhalb des Landes]", "[kurze Flüge zwischen den Städten]"],
                         "note": ["Inlandsflüge"]},
}
NO_FLIGHTS = {"sent": ["Wir [haben unsere Flüge selbst gebucht].", "Unsere [Flüge sind schon erledigt].",
                       "Es geht [nur um die Landarrangements].", "Wir [fliegen mit unseren Meilen]."],
              "note": ["nur Landprogramm", "Flüge vorhanden", "ohne Flüge"]}
INTERESTS = {
    "wildlife": {"np": ["[Tigersafaris]", "[Großkatzen in freier Wildbahn]", "[Pirschfahrten im Morgengrauen]",
                        "[wilde Elefanten]", "[alles rund um die Vogelwelt]"], "note": ["Tierwelt", "Tiger", "Safari"]},
    "culture and heritage": {"np": ["[alte Tempel]", "[UNESCO-Stätten]", "die [Geschichte der Moguln]", "[koloniale Altstädte]",
                                    "[Skulpturen und Architektur]"], "note": ["Kultur", "Geschichte"]},
    "wellness and yoga": {"np": ["[tägliches Yoga]", "eine [Ayurveda-Kur]", "[die eine oder andere Massage]",
                                 "[Spa-Anwendungen]", "[echte Ruhe und Meditation]"], "note": ["Yoga", "Ayurveda", "Wellness"]},
    "beach": {"np": ["[faule Tage am Strand]", "[ein paar Tage am Meer]", "[Baden im warmen Wasser]", "[Zeit am Strand]"],
              "note": ["Strand", "Badeurlaub"]},
    "food and cooking": {"np": ["[kulinarische Streifzüge durch die Gassen]", "einen [Kochkurs bei einer einheimischen Familie]",
                                "[Essen dort, wo die Einheimischen essen]", "[Gewürzmärkte und Hausmannskost]"],
                         "note": ["Kulinarik", "Kochkurs"]},
    "trekking and hiking": {"np": ["ein [mehrtägiges Trekking]", "[Wanderungen in den Vorbergen]", "[Touren von Dorf zu Dorf]",
                                   "[Bergpfade]"], "note": ["Trekking", "Wandern"]},
    "photography": {"np": ["[Fotografie]", "[gute Bilder bei Sonnenaufgang]", "[Aufnahmen vom Dorfleben]"],
                    "note": ["Fotografie"]},
    "diving and snorkelling": {"np": ["[Tauchgänge]", "[Schnorcheln mit Schildkröten]", "einen [Tauchkurs]"],
                               "note": ["Tauchen", "Schnorcheln"]},
    "shopping": {"np": ["[Einkäufe auf dem Textilmarkt]", "[Bummeln durch die Basare]", "die [Jagd nach Antiquitäten]",
                        "[Seide und Teppiche]"], "note": ["Shopping"]},
    "golf": {"np": ["[ein oder zwei Runden Golf]", "[Golf]"], "note": ["Golf"]},
    "tea plantations": {"np": ["einen [Aufenthalt auf einer aktiven Teeplantage]", "eine [Teeverkostung in den Bergen]",
                               "die [Teegärten]"], "note": ["Teeplantagen"]},
    "luxury train": {"np": ["eine [Fahrt im Luxuszug]", "[ein paar Nächte in einem königlichen Zug]"], "note": ["Luxuszug"]},
    "festivals": {"np": ["[Holi]", "das [Lichterfest Diwali]", "ein [lokales Fest]", "den [Kamelmarkt in Pushkar]"],
                  "note": ["Feste", "Festivals"]},
    "spirituality": {"np": ["das [Morgengebet in einem Kloster]", "die [Ghats im Morgengrauen]", "[heilige Stätten]",
                            "[Zeit in einem buddhistischen Kloster]"], "note": ["Klöster", "Spiritualität"]},
}
DIETARY = {
    "vegetarian": {"sent": ["Einer von uns [ernährt sich vegetarisch].", "{partner} [verzichtet auf Fleisch und Fisch].",
                            "Ich [habe noch nie Fleisch gegessen].", "Wir [essen ausschließlich vegetarisch]."],
                   "note": ["vegetarisch", "veggie", "ohne Fleisch"]},
    "vegan": {"sent": ["Ich [ernähre mich vegan].", "{partner} [lebt streng pflanzlich].", "Wir [meiden alle tierischen Produkte]."],
              "note": ["vegan"]},
    "gluten-free": {"sent": ["{partner} [hat Zöliakie].", "Einer von uns [braucht eine strikt weizenfreie Kost].",
                             "Ich esse [glutenfrei]."], "note": ["glutenfrei", "Zöliakie"]},
    "lactose-free": {"sent": ["Ich [vertrage keinen Milchzucker].", "{partner} [verträgt keine Milchprodukte].",
                              "Meine Tochter braucht [laktosefreie] Mahlzeiten."], "note": ["laktosefrei", "ohne Milch"]},
    "nut allergy": {"sent": ["{partner} [hat wegen Erdnüssen immer einen Adrenalin-Pen dabei].",
                             "Ich habe [eine schwere Nussallergie].",
                             "[Cashews und Erdnüsse sind gefährlich] für {partner_acc}."], "note": ["Nussallergie"]},
    "shellfish allergy": {"sent": ["Ich bin [allergisch gegen Garnelen].", "{partner} [reagiert heftig auf Krabben und Hummer].",
                                   "Einer von uns hat [eine Schalentierallergie]."],
                          "note": ["Schalentierallergie", "Krebstierallergie"]},
    "halal": {"sent": ["Wir brauchen [Halal-Essen].", "Ich esse [nur Halal-Fleisch]."], "note": ["halal"]},
    "kosher": {"sent": ["Wir brauchen [streng koschere] Mahlzeiten.", "Mein Essen muss [koscher] sein."], "note": ["koscher"]},
    "jain": {"sent": ["Wir essen [jainistisch], also kein Wurzelgemüse.", "Meine Mutter braucht [Jain-Mahlzeiten]."],
             "note": ["Jain-Küche"]},
    "diabetic": {"sent": ["Mein Vater [hat Diabetes].", "Ich habe [Typ-2-Diabetes].", "{partner} [muss auf den Zucker achten]."],
                 "note": ["Diabetes"]},
}
OCCASIONS = {
    "honeymoon": {"sent": ["Das werden unsere [Flitterwochen].", "Wir [heiraten im Frühjahr und möchten direkt danach los].",
                           "Es wäre unsere [erste Reise als Ehepaar]."], "note": ["Flitterwochen", "Hochzeitsreise"]},
    "anniversary": {"sent": ["Wir feiern [25 Jahre Ehe].", "Es ist unsere [Rubinhochzeit].",
                             "Die Reise ist zu unserem [Hochzeitstag]."], "note": ["Hochzeitstag"]},
    "birthday": {"sent": ["Die Reise ist für [meinen 60. Geburtstag].", "Ich [werde fünfzig] und möchte das feiern.",
                          "{partner} hat in dem Monat [einen runden Geburtstag]."], "note": ["Geburtstagsreise", "60. Geburtstag"]},
    "retirement": {"sent": ["Das ist [mein Geschenk an mich selbst zur Pensionierung].",
                            "Ich [habe nach 35 Jahren aufgehört zu arbeiten].", "Wir feiern [das Ende von 40 Berufsjahren]."],
                   "note": ["Ruhestand", "Pensionierung"]},
    "proposal": {"sent": ["Ich möchte {partner_dat} auf dieser Reise [einen Heiratsantrag machen].",
                          "Ich möchte [auf die Knie gehen], und zwar irgendwo, wo es wunderschön ist.",
                          "Ich habe vor, [die große Frage zu stellen], wenn wir dort sind."], "note": ["Heiratsantrag"]},
    "graduation": {"sent": ["Es ist ein [Geschenk zum Studienabschluss] für unsere Tochter.",
                            "Ich habe gerade [mein Medizinstudium abgeschlossen]."], "note": ["Studienabschluss"]},
    "family reunion": {"sent": ["Es ist ein [Familientreffen].", "Die Idee ist, [alle Cousins und Cousinen zusammenzubringen]."],
                       "note": ["Familientreffen"]},
    "babymoon": {"sent": ["Es wäre unsere [letzte Reise, bevor das Baby kommt].", "Wir suchen einen [Babymoon]."],
                 "note": ["Babymoon"]},
    "Christmas or New Year": {"sent": ["Wir möchten über [Weihnachten] weg sein.", "Wir wollen [Silvester] im Warmen feiern.",
                                       "Wir möchten [die Feiertage] im Warmen verbringen."], "note": ["Weihnachten", "Silvester"]},
}
SPECIAL = {
    "private guide": {"sent": ["Wir hätten gern [unseren eigenen Reiseleiter].", "Bitte einen [persönlichen Guide] einplanen.",
                               "Ein [Guide nur für uns] wäre ideal."], "note": ["Privatguide"]},
    "private driver": {"sent": ["Wir möchten [nicht selbst fahren].", "Bitte [ein eigenes Auto samt Chauffeur] einplanen.",
                                "Ein [Privatfahrer] für die ganze Strecke wäre gut."], "note": ["Privatfahrer", "Auto + Fahrer"]},
    "accessibility needs": {"sent": ["{partner} [sitzt im Rollstuhl].", "Wir brauchen [stufenlosen Zugang].",
                                     "Meine Mutter [geht an zwei Stöcken], Treppen sind also schwierig."],
                            "note": ["rollstuhlgerecht"]},
    "airport transfers": {"sent": ["Bitte die [Transfers vom Flughafen] einplanen.", "Wir hätten gern [ein Auto bei der Ankunft].",
                                   "Wir müssten [am Flughafen abgeholt werden]."], "note": ["Flughafentransfer"]},
    "childcare": {"sent": ["Ein [Kindermädchen] für die Abende wäre eine Hilfe.", "Gibt es [Babysitting]?",
                           "Wir hätten gern [jemanden, der nach dem Abendessen auf die Kinder aufpasst]."],
                  "note": ["Babysitter", "Kinderclub"]},
    "butler service": {"sent": ["Wir haben [Butler-Service] schon einmal genossen.", "Ein [persönlicher Butler] wäre ein schöner Luxus."],
                       "note": ["Butler"]},
    "late checkout": {"sent": ["Am letzten Tag hätten wir gern einen [späten Check-out].",
                               "Könnten wir [das Zimmer bis zum Abend behalten], bevor wir abreisen?"], "note": ["Late Check-out"]},
    "surprise arrangement": {"sent": ["Ich brauche Hilfe bei [einer Überraschung] für diesen Abend.",
                                      "Ich hätte gern [ein paar heimliche Extras] für den Tag selbst."], "note": ["Überraschung"]},
    "helicopter": {"sent": ["Wir würden auch [einen Hubschraubertransfer] in Betracht ziehen.",
                            "Ein [Panoramaflug mit dem Helikopter] wäre etwas Besonderes."], "note": ["Helikopter"]},
    "baby cot": {"sent": ["Wir brauchen ein [Babybett] im Zimmer.", "Bitte ein [Reisebett] für das Baby dazustellen."],
                 "note": ["Babybett"]},
    "short drives only": {"sent": ["Wir möchten [die Fahrzeiten kurz halten].", "Bitte [nichts über drei Stunden im Auto]."],
                          "note": ["kurze Fahrten"]},
    "German-speaking guide": {"sent": ["Wir bräuchten einen [Guide, der gut Deutsch spricht]."],
                              "note": ["deutschsprachiger Guide"]},
}

# text that looks like a field but is not, so it must stay unlabelled
PAST_PLACES = {"Thailand": "Thailand", "Bali": "Bali", "Morocco": "Marokko", "Peru": "Peru", "Vietnam": "Vietnam",
               "Japan": "Japan", "Kenya": "Kenia", "Jordan": "Jordanien", "India": "Indien", "Nepal": "Nepal",
               "Sri Lanka": "Sri Lanka", "Goa": "Goa"}
HOME_CITIES = ["Köln", "Stuttgart", "Bremen", "Leipzig", "Dresden", "Graz", "Linz", "Bern"]
DISTRACTORS = {
    "past": ["Vor ein paar Jahren waren wir in {past} und waren begeistert.",
             "Letzten Winter waren wir in {past}, deshalb suchen wir jetzt etwas anderes.",
             "Unsere Nachbarn sind gerade aus {past} zurückgekommen.", "{past} kennen wir schon, das muss also nicht sein."],
    "home": ["Wir wohnen in der Nähe von {home}.", "Wir leben in {home}.", "Ich schreibe aus {home}, wo es seit einer Woche regnet."],
    "other_date": ["Ich habe am {d}. {m} mit einem Ihrer Kollegen gesprochen.", "Mein Pass läuft im {m} 2031 ab.",
                   "Ich bin bis zum {d}. {m} beruflich unterwegs, Antworten können also dauern."],
    "other_money": ["Unsere letzte große Reise hat rund {money} gekostet.", "Ein Freund hat dort {money} pro Nacht für ein Zimmer bezahlt.",
                    "Für eine Safari haben wir vor Jahren etwa {money} ausgegeben."],
    "other_people": ["Unsere zwei Hunde bleiben bei meiner Schwester.", "Meine Eltern, beide über achtzig, kommen nicht mit.",
                     "Unsere drei erwachsenen Kinder bleiben zu Hause."],
    "part_stay": ["Vielleicht {n} Nächte {place_in} für den Anfang.", "Wir stellen uns {n} Nächte {place_in} vor und den Rest woanders."],
    "refusal": ["Auf {x} können wir verzichten.", "Bitte {x} weglassen.", "{x} brauchen wir nicht."],
}

NAMES_DE = {"India": "Indien", "Maldives": "Malediven", "Baa Atoll": "Baa-Atoll", "Ari Atoll": "Ari-Atoll"}
CITIES = {"Geneva": "Genf", "Zurich": "Zürich", "Munich": "München", "Vienna": "Wien", "Brussels": "Brüssel",
          "Luxembourg": "Luxemburg", "Nice": "Nizza"}


def place(name: str, case: str) -> list:
    span = S("DESTINATION", NAMES_DE.get(name, name), name)
    if case == "note":
        return [span]
    if name == "Maldives":
        before = {"in": "auf den ", "to": "auf die ", "obj": "die "}[case]
    elif name.endswith("Atoll"):
        before = {"in": "im ", "to": "ins ", "obj": "das "}[case]
    else:
        before = {"in": "in ", "to": "nach ", "obj": ""}[case]
    return [T(before), span] if before else [span]
