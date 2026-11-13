"""French wording for the synthetic enquiry generator. Same markup as lang_en.py.

Sentence frames avoid "de" and "à" right before a slot, so a phrase can keep its own article
without any contraction. Adjectives that would need to agree with the writer are avoided too.
All wording here was written for this project. It has not been checked by a native speaker.
"""

import re

from .parts import S, T

CODE = "fr"
JOIN = (", ", " et ")
PARTNERS = [{"partner": p} for p in ("mon mari", "ma femme", "mon épouse", "mon compagnon", "ma compagne",
                                     "mon fiancé", "ma fiancée", "mon conjoint")]
PROPOSAL_PARTNERS = [{"partner": p} for p in ("mon compagnon", "ma compagne")]
NOT_SOLO = re.compile(r"\b(nous|notre|nos)\b", re.IGNORECASE)
NUM = {1: "un", 2: "deux", 3: "trois", 4: "quatre", 5: "cinq", 6: "six", 7: "sept", 8: "huit", 9: "neuf", 10: "dix",
       11: "onze", 12: "douze", 13: "treize", 14: "quatorze", 15: "quinze", 16: "seize", 17: "dix-sept",
       18: "dix-huit", 19: "dix-neuf", 20: "vingt", 21: "vingt et un"}
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
          "novembre", "décembre"]
MONTHS_ABBR = ["janv", "févr", "mars", "avr", "mai", "juin", "juil", "août", "sept", "oct", "nov", "déc"]
OF_MONTH = ["de janvier", "de février", "de mars", "d'avril", "de mai", "de juin", "de juillet", "d'août",
            "de septembre", "d'octobre", "de novembre", "de décembre"]


def day(n: int):
    return "1er" if n == 1 else n


DATE_FORMS = {
    "month": [("{m}", ["en {D}", "courant {D}", "vers {D}", "dès {D}"]), ("{m} {y}", ["en {D}", "vers {D}"])],
    "early": [("début {m}", ["{D}", "vers {D}", "dès {D}"]), ("tout début {m}", ["{D}"]),
              ("la première semaine {dm}", ["{D}", "durant {D}"])],
    "mid": [("mi-{m}", ["{D}", "vers la {D}", "à la {D}", "à partir de la {D}"]),
            ("milieu {dm}", ["vers le {D}", "au {D}"])],
    "late": [("fin {m}", ["{D}", "vers {D}", "à partir de {D}"]), ("la dernière semaine {dm}", ["{D}", "durant {D}"]),
             ("les derniers jours {dm}", ["durant {D}", "dans {D}"])],
    "range_month": [("{d} au {d2} {m}", ["du {D}"]), ("{n}-{d2} {m}", ["{D}"]), ("{d} au {d2} {m} {y}", ["du {D}"]),
                    ("{d} et le {d2} {m}", ["entre le {D}"])],
    "range_cross": [("{d} {m} au {d2} {m2}", ["du {D}"]), ("{d} {ma} - {d2} {ma2}", ["{D}"])],
    "single": [("{d} {m}", ["le {D}", "vers le {D}", "à partir du {D}"]), ("{d} {m} {y}", ["le {D}"])],
    "numeric_ambiguous": [("{n}/{mo}", ["le {D}", "vers le {D}", "à partir du {D}"]),
                          ("{n}/{mo}/{y}", ["le {D}", "à partir du {D}"])],
    "numeric_clear": [("{n}/{mo2}/{y}", ["le {D}", "à partir du {D}"]), ("{n}.{mo2}.{y}", ["le {D}"])],
    "month_range": [("{m}-{m2}", ["{D}", "en {D}"]), ("{m}/{m2}", ["en {D}"]), ("{ma}/{ma2}", ["{D}"]),
                    ("{m} et {m2}", ["entre {D}"])],
    "month_choice": [("{m} ou {m2}", ["en {D}", "soit en {D}"]), ("{ma} ou {ma2}", ["en {D}"])],
}
DURATION_FORMS = {
    "nights": ["{n} nuits", "{w} nuits", "{n} nuitées"],
    "days": ["{n} jours", "{w} jours"],
    "weeks": ["{w} semaines", "{n} semaines"],
    "one_week": ["une semaine", "sept jours", "huit jours"],
    "two_weeks": ["deux semaines", "quinze jours", "2 semaines"],
}
CURRENCY_FORMS = {
    "USD": ["{a} $", "{a} dollars", "{a} USD", "USD {a}"],
    "EUR": ["{a} €", "{a} euros", "{a} EUR", "EUR {a}", "€{a}"],
    "GBP": ["{a} £", "{a} livres", "{a} livres sterling", "{a} GBP"],
    "CHF": ["{a} CHF", "CHF {a}", "{a} francs suisses"],
    "INR": ["{a} roupies", "INR {a}", "{a} INR"],
}
LAKH_FORMS = ["{l} lakhs de roupies", "INR {l} lakh"]
THOUSAND_SEP = " "
K_FORMS = ["{k}k", "{k} K", "{k} mille"]
K_WORDS = {3: "trois", 4: "quatre", 5: "cinq", 6: "six", 8: "huit", 9: "neuf", 10: "dix", 12: "douze", 15: "quinze",
           18: "dix-huit", 20: "vingt", 25: "vingt-cinq", 30: "trente", 40: "quarante", 50: "cinquante",
           75: "soixante-quinze", 100: "cent"}
WORD_AMOUNT = "{w} mille"
BUDGET_SHAPES = {
    "about": ["environ {x}", "à peu près {x}", "aux alentours de {x}", "autour de {x}", "dans les {x}", "{x} environ"],
    "exact": ["{x}"],
    "up_to": ["jusqu'à {x}", "{x} maximum", "au maximum {x}", "au plus {x}", "pas plus de {x}", "{x} grand maximum"],
    "at_least": ["au moins {x}", "{x} minimum", "{x} ou plus"],
    "range": ["entre {x} et {y}", "de {x} à {y}", "{n}-{y}"],
}
BASIS = {"per_person": [" par personne", " chacun", " par voyageur", " par pers."],
         "per_trip": [" au total", " en tout", " pour tout le voyage", " pour l'ensemble du séjour"]}
AGE_JOIN = (", ", " et ")

PARTY = {
    "couple": {
        "subject": ["[[ADULTS|{partner} et moi]]", "[[ADULTS|{partner} et moi]]", "[[PARTY|Tous les deux]], nous"],
        "np": ["[[ADULTS|deux adultes]]", "[[PARTY|deux personnes]]", "un [[ADULTS|couple]]", "un [[ADULTS|couple marié]]",
               "[[ADULTS|2 adultes]]", "[[PARTY|deux voyageurs]]"],
        "note": ["[[ADULTS|2 adultes]]", "[[PARTY|2 pers.]]", "[[ADULTS|couple]]", "[[PARTY|2 pax]]", "[[ADULTS|2 ad.]]"],
    },
    "solo": {
        "subject": [],
        "np": ["[[PARTY|une seule personne]]", "[[ADULTS|un adulte]]", "[[PARTY|une personne]]"],
        "note": ["[[PARTY|1 pers.]]", "[[ADULTS|1 adulte]]", "[[PARTY|solo]]"],
    },
    "family": {
        "subject": ["[[ADULTS|{partner} et moi]] et [[CHILDREN|nos {c} enfants]]",
                    "[[ADULTS|{partner} et moi]] avec [[CHILDREN|nos {c} enfants]]"],
        "np": ["[[ADULTS|{a} adultes]] et [[CHILDREN|{c} enfants]]", "[[ADULTS|{a} adultes]] avec [[CHILDREN|{c} enfants]]",
               "une [[PARTY|famille de {t} personnes]]", "[[PARTY|{t} personnes]], dont [[CHILDREN|{c} enfants]]"],
        "note": ["[[ADULTS|{a} adultes]] + [[CHILDREN|{c} enfants]]", "[[ADULTS|{a} ad.]] / [[CHILDREN|{c} enf.]]",
                 "[[PARTY|famille de {t}]]"],
        "one_child": ["[[ADULTS|{a} adultes]] et [[CHILDREN|un enfant]]", "[[ADULTS|{a} adultes]] et [[CHILDREN|notre fils]]",
                      "[[ADULTS|{a} adultes]] et [[CHILDREN|notre fille]]"],
        "single_parent": ["[[ADULTS|un parent]] et [[CHILDREN|{c} enfants]]",
                          "[[ADULTS|une mère seule]] avec [[CHILDREN|{c} enfants]]"],
    },
    "friends": {
        "subject": ["Nous, [[PARTY|{t} amis d'université]],", "Nous, [[PARTY|{t} copains de lycée]],",
                    "Nous, [[PARTY|{t} amis de longue date]],"],
        "np": ["[[PARTY|{t} amis]]", "un [[PARTY|groupe de {t} amis]]", "[[ADULTS|{t} adultes]]", "[[PARTY|{t} personnes]]"],
        "note": ["[[PARTY|{t} amis]]", "[[PARTY|{t} pax]]", "[[ADULTS|{t} adultes]]"],
    },
    "two_couples": {
        "subject": ["Nous, [[ADULTS|{p} couples]],", "Nous, [[ADULTS|{p} couples de retraités]],"],
        "np": ["[[ADULTS|{p} couples]]", "[[PARTY|{t} personnes]]", "[[ADULTS|{t} adultes]]"],
        "note": ["[[ADULTS|{p} couples]]", "[[PARTY|{t} pax]]"],
    },
    "multigen": {
        "subject": [],
        "np": ["[[PARTY|{t} personnes]] sur trois générations", "[[ADULTS|{a} adultes]] et [[CHILDREN|{c} petits-enfants]]",
               "[[PARTY|{t} personnes]], grands-parents compris"],
        "note": ["[[PARTY|{t} pax]]", "[[ADULTS|{a} adultes]] + [[CHILDREN|{c} enfants]]"],
    },
    "group": {
        "subject": [],
        "np": ["un [[PARTY|groupe de {t} personnes]]", "[[PARTY|{t} personnes]]", "[[PARTY|{t} participants]]"],
        "note": ["[[PARTY|{t} pax]]", "[[PARTY|groupe de {t}]]"],
    },
}

SALUTATIONS = {
    "formal": ["Madame, Monsieur,", "Chère équipe,", "Bonjour à toute l'équipe,", "Chère équipe Kenexoft,"],
    "neutral": ["Bonjour,", "Bonsoir,", "Bonjour à vous,", "Bonjour Kenexoft,"],
    "casual": ["salut", "bonjour", "coucou", "Hello,", ""],
}
SIGN_OFFS = {
    "formal": ["Cordialement,", "Bien cordialement,", "Avec mes remerciements,", "Sincères salutations,"],
    "neutral": ["Merci,", "Bien à vous,", "À bientôt,", "Belle journée,", "Merci d'avance,"],
    "casual": ["merci", "a+", "bises", ""],
}
NAMES = ["Clémence Aubry", "Gaspard Lemoine", "Aurélie Vasseur", "Bastien Roche", "Mathilde Perrin", "Lucien Garnier",
         "Capucine Morel", "Édouard Fabre"]
EXTRA_LINES = ["Envoyé de mon iPad", "Envoyé depuis mon téléphone, merci d'excuser les fautes", "Tél. : +33 6 39 98 01 23"]

OPENERS = [
    "Nous avons découvert votre agence dans un article de magazine et ce que nous y avons lu nous a plu.",
    "Un collègue qui a voyagé avec vous l'an dernier m'a donné votre nom.",
    "J'ai passé la soirée sur votre site et j'ai préféré vous écrire directement.",
    "Des amis nous parlent régulièrement de vous, me voici donc.",
    "Nous n'en sommes qu'au tout début de nos réflexions et vos conseils seraient les bienvenus.",
    "C'est la première fois que nous passons par un spécialiste, merci de votre indulgence.",
    "Je vais tenter de faire court, même si j'y arrive rarement.",
    "Nous en parlons depuis longtemps et le moment semble enfin venu.",
    "Veuillez excuser la longueur de ce message.",
]
FILLERS = [
    "Nous ne sommes pas pressés et préférons rester sur place plutôt que changer d'hôtel chaque soir.",
    "Le rythme compte plus pour nous que le nombre de sites visités.",
    "Nous aimons partir tôt et garder des après-midi calmes.",
    "Nous préférons voir moins de choses, mais bien.",
    "Nous acceptons de payer pour la qualité, mais sans ostentation.",
    "Aucun de nous n'aime la foule.",
    "Un bon café le matin serait un plus.",
    "Nous connaissons assez bien l'Europe, mais nous ne sommes jamais allés aussi loin à l'est.",
    "La chaleur ne nous dérange pas vraiment.",
    "Un wifi fiable compte un peu, car l'un de nous devra peut-être garder un œil sur son travail.",
]
CLOSERS = [
    "Que conseilleriez-vous ?",
    "Merci de m'envoyer une première ébauche avec une idée du coût.",
    "Dans l'attente de votre réponse.",
    "Je suis disponible par téléphone si c'est plus simple.",
    "N'hésitez pas à me dire s'il vous manque quelque chose.",
    "Une première proposition par e-mail serait parfaite.",
]
VAGUE = ["Merci de m'envoyer quelques idées.", "Que recommanderiez-vous ?", "Par où commencer ?",
         "Pouvez-vous m'aider ?", "Toutes vos suggestions sont les bienvenues."]

INTRO = [
    "{subject} aimerions passer {dur} {dest}.",
    "{subject} pensons partir {dest} {date}.",
    "{subject} envisageons {dur} {dest} {date}.",
    "Nous sommes {party} et nous rêvons de partir {dest}.",
    "Nous sommes {party} et cherchons à passer {dur} {dest} {date}.",
    "J'organise un voyage {dest} pour {party}.",
    "Pourriez-vous imaginer un séjour {dest} pour {party} ?",
    "Pourriez-vous nous aider à préparer un voyage {dest} {date} ?",
    "Nous avons très envie de découvrir {dest_obj}.",
    "Que proposeriez-vous pour {dur} {dest} ?",
    "Nous espérons partir {dest} {date}.",
    "Je me renseigne pour {dur} {dest} pour {party}.",
    "Nous rêvons de découvrir {dest_obj} et pourrions nous libérer {date}.",
]
INTRO_SOLO = [
    "J'aimerais passer {dur} {dest}.",
    "Je pense partir {dest} {date}.",
    "J'envisage {dur} {dest} {date}.",
    "J'espère partir {dest} {date}.",
    "Que proposeriez-vous pour {dur} {dest} ?",
    "J'ai très envie de découvrir {dest_obj}.",
]
SENT_SOLO = {
    "dest": ["Je pense à un voyage {dest}.", "L'idée serait de partir {dest}."],
    "dur": ["J'ai {dur} devant moi.", "Je peux me libérer {dur}.", "Durée du séjour : {dur}."],
    "date": ["Je peux partir {date}.", "L'idéal pour moi serait {date}.", "Je partirais {date}."],
    "date_dur": ["J'espère partir {date} pour {dur}."],
    "party": ["Le voyage concerne {party}.", "C'est pour {party}."],
    "budget": ["Mon budget : {budget}.", "J'ai prévu {budget}.", "Côté budget, je vise {budget}."],
    "accommodation": ["J'aimerais loger dans {x}.", "Je recherche {x}.", "Je privilégie {x}."],
    "flights": ["Il me faut {x}.", "Merci d'inclure {x}."],
    "flights_origin": ["Je souhaite {x} depuis {origin}."],
    "origin": ["Je prendrais l'avion à {origin}."],
    "interests": ["J'aimerais prévoir {x}.", "J'aimerais beaucoup {x}.", "Sur ma liste : {x}."],
}
SENT = {
    "dest": ["Nous pensons à un voyage {dest}.", "L'idée serait de partir {dest}.", "Nous aimerions aller {dest}.",
             "Notre premier choix serait {dest_obj}."],
    "dur": ["Nous avons {dur} devant nous.", "Nous pouvons nous libérer {dur}.", "Nous comptons {dur} en tout.",
            "Durée du séjour : {dur}.", "Nous pensons à {dur} environ."],
    "date": ["Nous pouvons partir {date}.", "L'idéal pour nous serait {date}.", "Nous sommes libres {date}.",
             "Nous partirions {date}.", "Nos dates restent souples, mais plutôt {date}."],
    "date_dur": ["Nous espérons partir {date} pour {dur}.", "En gros, {dur} {date}."],
    "party": ["Nous serons {party}.", "Nous sommes {party}.", "Le voyage concerne {party}.",
              "Côté participants, nous sommes {party}."],
    "budget": ["Notre budget : {budget}.", "Nous avons prévu {budget}.", "Côté budget, nous visons {budget}.",
               "Nous comptons mettre {budget}.", "Enveloppe prévue : {budget}."],
    "accommodation": ["Nous aimerions loger dans {x}.", "Nous aimerions {x}.", "Nous recherchons {x}.",
                      "Nous privilégions {x}.", "Merci de prévoir {x}."],
    "flights": ["Il nous faut {x}.", "Merci d'inclure {x}.", "Pour les vols, nous souhaitons {x}."],
    "flights_origin": ["Nous souhaitons {x} depuis {origin}.", "Merci de chiffrer {x} depuis {origin}."],
    "origin": ["Nous prendrions l'avion à {origin}.", "Nous décollerions depuis {origin}.",
               "Les vols partiraient depuis {origin}."],
    "interests": ["Nous aimerions prévoir {x}.", "Au programme, si possible : {x}.", "Sur notre liste : {x}.",
                  "Pourriez-vous inclure {x} ?", "Nous aimerions beaucoup {x}."],
    "ages_children": ["Les enfants ont {ages} ans.", "Les enfants auront {ages} ans au moment du voyage.",
                      "Leurs âges : {ages} ans."],
    "ages_one": ["Notre enfant a {ages} ans.", "Notre enfant aura {ages} ans au moment du départ."],
    "ages_adults": ["Nous avons {ages} ans.", "Pour information, nous avons {ages} ans."],
    "age_solo": ["J'ai {ages} ans.", "J'aurai {ages} ans au moment du voyage."],
}
NOTE_OPENERS = ["Demande de voyage :", "Demande -", "Projet :", "Devis svp :", ""]
NOTE_CLOSERS = ["merci", "svp", "vos idées ?", "idées ?", ""]
NOTE_LABELS = {"budget": ["budget", "budget :", "enveloppe", ""]}
NOTE_ORIGIN = ["depuis ", "départ ", "vols depuis "]
ALT_BUDGET = [" (soit environ {x})", " (à peu près {x} au cours du jour)"]

ACCOMMODATION = {
    "overwater villa": {"np": ["un [bungalow au-dessus du lagon]", "un [bungalow sur pilotis]", "une [villa sur l'eau]"],
                        "note": ["villa sur l'eau", "bungalow sur pilotis"]},
    "private pool": {"np": ["une chambre avec [piscine privée]", "une [petite piscine rien que pour nous]",
                            "une [piscine à nous]"], "note": ["piscine privée"]},
    "villa": {"np": ["une [villa]", "une [villa entière pour nous seuls]", "une [maison privée avec personnel]"],
              "note": ["villa"]},
    "heritage or palace hotel": {"np": ["un [haveli restauré]", "une [ancienne résidence royale]",
                                        "un [palais transformé en hôtel]", "une [demeure historique]"],
                                 "note": ["hôtels historiques", "haveli"]},
    "boutique hotel": {"np": ["un [boutique-hôtel]", "un [petit hôtel plein de style]", "un [minuscule hôtel design]",
                              "des [adresses intimes de dix chambres au plus]"], "note": ["boutique-hôtel"]},
    "5-star": {"np": ["des [hôtels cinq étoiles]", "des [adresses 5 étoiles]", "un [confort cinq étoiles]"],
               "note": ["5*", "5 étoiles"]},
    "suite": {"np": ["une [suite]", "une [suite junior]", "une [véritable suite]"], "note": ["suite"]},
    "beachfront": {"np": ["une chambre [les pieds dans le sable]", "un endroit [à deux pas de la mer]",
                          "une chambre [face à la mer]"], "note": ["front de mer", "sur la plage"]},
    "tented camp": {"np": ["un [camp de tentes de luxe]", "des [tentes façon safari]", "un peu de [glamping]"],
                    "note": ["camp de tentes", "glamping"]},
    "connecting rooms": {"np": ["des [chambres communicantes]", "des [chambres voisines reliées par une porte]"],
                         "note": ["ch. communicantes"]},
    "family room": {"np": ["une [chambre familiale]", "une [grande chambre pour nous tous]"],
                    "note": ["chambre familiale"]},
    "eco-lodge": {"np": ["un [écolodge]", "un [lodge à l'énergie solaire]", "un endroit [durable et à faible impact]"],
                  "note": ["écolodge"]},
    "houseboat": {"np": ["une [maison flottante]", "une [barge à riz traditionnelle]",
                         "une nuit à bord d'un [kettuvallam]"], "note": ["houseboat", "maison flottante"]},
    "safari lodge": {"np": ["un [lodge de safari]", "un [lodge en lisière du parc national]", "un [lodge en pleine jungle]"],
                     "note": ["lodge safari"]},
}
FLIGHTS = {
    "business class": {"np": ["la [classe affaires]", "des [sièges-lits]", "des [lits à plat à l'avant de l'avion]"],
                       "note": ["classe affaires", "business"]},
    "first class": {"np": ["la [première classe]", "des [sièges en première]"], "note": ["première classe", "1ère classe"]},
    "premium economy": {"np": ["la [classe premium]", "des [sièges en premium economy]"], "note": ["premium éco"]},
    "economy class": {"np": ["la [classe économique]", "des [sièges standard en éco]"], "note": ["classe éco", "éco"]},
    "direct flights": {"np": ["des [vols directs]", "des [vols sans escale]", "des [liaisons sans correspondance]"],
                       "note": ["vol direct", "sans escale"]},
    "domestic flights": {"np": ["des [vols intérieurs]", "des [vols domestiques]",
                                "de [petits sauts en avion entre les villes]"], "note": ["vols intérieurs"]},
}
NO_FLIGHTS = {"sent": ["Nous [avons réservé nos vols nous-mêmes].", "Nos [vols sont déjà pris].",
                       "Il s'agirait [uniquement des prestations terrestres].", "Nous [utiliserons nos miles pour les vols]."],
              "note": ["terrestre uniquement", "vols déjà pris", "hors vols"]}
INTERESTS = {
    "wildlife": {"np": ["un [safari pour voir le tigre]", "l'[observation des grands fauves]",
                        "des [sorties à l'aube dans les parcs]", "l'[approche des éléphants sauvages]",
                        "[tout ce qui touche aux oiseaux]"], "note": ["faune sauvage", "tigres", "safari"]},
    "culture and heritage": {"np": ["les [temples anciens]", "les [sites classés à l'UNESCO]", "l'[histoire moghole]",
                                    "les [anciens comptoirs coloniaux]", "la [sculpture et l'architecture]"],
                             "note": ["culture", "histoire", "patrimoine"]},
    "wellness and yoga": {"np": ["du [yoga chaque matin]", "une [cure ayurvédique]", "un [massage ou deux]",
                                 "des [soins au spa]", "de [vrais moments de repos et de méditation]"],
                          "note": ["yoga", "ayurveda", "bien-être"]},
    "beach": {"np": ["des [journées tranquilles sur la plage]", "[quelques jours au bord de la mer]",
                     "des [baignades en eau chaude]", "du [temps sur le sable]"], "note": ["plage", "bord de mer"]},
    "food and cooking": {"np": ["une [visite gourmande dans les rues]", "un [cours de cuisine chez l'habitant]",
                                "des [repas là où mangent les gens du coin]", "les [marchés aux épices et la cuisine maison]"],
                         "note": ["cuisine", "gastronomie"]},
    "trekking and hiking": {"np": ["un [trek de plusieurs jours]", "des [marches dans les contreforts]",
                                   "des [randonnées de village en village]", "des [sentiers de montagne]"],
                            "note": ["trekking", "rando"]},
    "photography": {"np": ["la [photographie]", "de [belles prises de vue au lever du soleil]",
                           "des [photos de la vie des villages]"], "note": ["photo", "photographie"]},
    "diving and snorkelling": {"np": ["de la [plongée bouteille]", "du [snorkeling avec les tortues]",
                                      "un [stage de plongée]"], "note": ["plongée", "snorkeling"]},
    "shopping": {"np": ["l'[achat de textiles]", "une [flânerie dans les bazars]", "la [chasse aux antiquités]",
                        "l'[achat de soie et de tapis]"], "note": ["shopping"]},
    "golf": {"np": ["une [partie de golf ou deux]", "du [golf]"], "note": ["golf"]},
    "tea plantations": {"np": ["un [séjour dans une plantation de thé en activité]",
                               "une [dégustation de thé dans les collines]", "les [jardins de thé]"],
                        "note": ["plantations de thé"]},
    "luxury train": {"np": ["un [voyage en train de luxe]", "[quelques nuits à bord d'un train royal]"],
                     "note": ["train de luxe"]},
    "festivals": {"np": ["[Holi]", "les [fêtes de Diwali]", "une [fête locale]", "la [grande foire de Pushkar]"],
                  "note": ["festivals", "fêtes locales"]},
    "spirituality": {"np": ["les [prières du matin dans un monastère]", "les [ghats à l'aube]", "les [lieux sacrés]",
                            "un [séjour dans un monastère bouddhiste]"], "note": ["monastères", "spiritualité"]},
}
DIETARY = {
    "vegetarian": {"sent": ["L'un de nous [ne mange pas de viande].", "{partner} [évite la viande et le poisson].",
                            "Je [n'ai jamais mangé de viande].", "Nous [mangeons uniquement végétarien]."],
                   "note": ["végétarien", "végé", "sans viande"]},
    "vegan": {"sent": ["Je [suis un régime végétalien].", "{partner} [se nourrit uniquement de végétal].",
                       "Nous [évitons tout produit animal]."], "note": ["vegan", "végane"]},
    "gluten-free": {"sent": ["{partner} [a la maladie cœliaque].", "L'un de nous [doit manger strictement sans blé].",
                             "Je mange [sans gluten]."], "note": ["sans gluten", "cœliaque"]},
    "lactose-free": {"sent": ["Je [digère mal le lactose].", "[Le lait ne réussit pas à] {partner}.",
                              "Ma fille a besoin de repas [sans produits laitiers]."], "note": ["sans lactose", "sans laitages"]},
    "nut allergy": {"sent": ["{partner} [a toujours un stylo d'adrénaline sur soi à cause des arachides].",
                             "J'ai [une forte allergie aux fruits à coque].",
                             "[Les noix de cajou et les cacahuètes sont dangereuses] pour {partner}."],
                    "note": ["allergie aux noix", "allergie fruits à coque"]},
    "shellfish allergy": {"sent": ["Je suis [allergique aux crevettes].", "{partner} [réagit très mal au crabe et au homard].",
                                   "L'un de nous a [une allergie aux crustacés]."],
                          "note": ["allergie crustacés", "allergie fruits de mer"]},
    "halal": {"sent": ["Nous avons besoin de repas [halal].", "Je mange [uniquement de la viande halal]."], "note": ["halal"]},
    "kosher": {"sent": ["Nous avons besoin de repas [strictement casher].", "Ma nourriture doit être [casher]."],
               "note": ["casher", "kasher"]},
    "jain": {"sent": ["Nous suivons un [régime jaïn], donc pas de légumes-racines.", "Ma mère a besoin de [repas jaïns]."],
             "note": ["repas jaïn"]},
    "diabetic": {"sent": ["Mon père est [diabétique].", "J'ai [un diabète de type 2].",
                          "{partner} [doit surveiller son taux de sucre]."], "note": ["diabétique"]},
}
OCCASIONS = {
    "honeymoon": {"sent": ["Ce sera notre [voyage de noces].", "Nous [nous marions au printemps et voulons partir juste après].",
                           "Ce serait notre [premier voyage en tant que mariés]."],
                  "note": ["voyage de noces", "lune de miel", "jeunes mariés"]},
    "anniversary": {"sent": ["Nous fêtons [nos 25 ans de mariage].", "Ce sont nos [noces de rubis].",
                             "Ce voyage marque notre [anniversaire de mariage]."], "note": ["anniversaire de mariage"]},
    "birthday": {"sent": ["C'est pour [mes 60 ans].", "Je [vais avoir cinquante ans] et je veux marquer le coup.",
                          "{partner} fête [un anniversaire important] ce mois-là."], "note": ["anniversaire", "60 ans"]},
    "retirement": {"sent": ["C'est [mon cadeau de départ à la retraite].",
                            "Je viens de [prendre ma retraite après 35 ans de travail].",
                            "Nous fêtons [la fin d'une carrière de 40 ans]."], "note": ["voyage retraite", "départ en retraite"]},
    "proposal": {"sent": ["J'ai l'intention de [demander {partner} en mariage] pendant ce voyage.",
                          "Je voudrais [faire ma demande en mariage] dans un endroit magnifique.",
                          "Je compte [faire ma demande] une fois sur place."], "note": ["demande en mariage"]},
    "graduation": {"sent": ["C'est un [cadeau de fin d'études] pour notre fille.", "Je viens de [terminer mes études de médecine]."],
                   "note": ["fin d'études"]},
    "family reunion": {"sent": ["Il s'agit d'une [réunion de famille].", "L'idée est de [réunir tous les cousins]."],
                       "note": ["réunion de famille"]},
    "babymoon": {"sent": ["Ce serait notre [dernier voyage avant l'arrivée du bébé].", "Nous cherchons un [babymoon]."],
                 "note": ["babymoon"]},
    "Christmas or New Year": {"sent": ["Nous aimerions partir pour [Noël].", "Nous voulons fêter [le réveillon du Nouvel An] au soleil.",
                                       "Nous comptons nous échapper pendant [les fêtes de fin d'année]."],
                              "note": ["Noël", "Nouvel An"]},
}
SPECIAL = {
    "private guide": {"sent": ["Nous aimerions [notre propre guide].", "Merci de prévoir un [guide personnel].",
                               "Un [guide rien que pour nous] serait idéal."], "note": ["guide privé"]},
    "private driver": {"sent": ["Nous aimerions [ne pas conduire nous-mêmes].", "Merci d'inclure [une voiture avec son chauffeur].",
                                "Un [chauffeur privé] sur tout le circuit, s'il vous plaît."],
                       "note": ["chauffeur privé", "voiture + chauffeur"]},
    "accessibility needs": {"sent": ["{partner} [se déplace en fauteuil roulant].", "Nous avons besoin d'un [accès sans marches].",
                                     "Ma mère [marche avec deux cannes], les escaliers sont donc difficiles."],
                            "note": ["accès fauteuil"]},
    "airport transfers": {"sent": ["Merci d'inclure les [transferts depuis l'aéroport].", "Nous aimerions [une voiture à l'arrivée].",
                                   "Il faudrait [venir nous chercher à l'aéroport]."], "note": ["transfert aéroport"]},
    "childcare": {"sent": ["Une [nounou] pour les soirées nous aiderait.", "Y a-t-il un [service de garde d'enfants] ?",
                           "Nous aimerions [quelqu'un pour garder les enfants] après le dîner."],
                  "note": ["baby-sitter", "club enfants"]},
    "butler service": {"sent": ["Nous avons apprécié le [service de majordome] par le passé.",
                                "Un [majordome personnel] serait un vrai plaisir."], "note": ["majordome"]},
    "late checkout": {"sent": ["Nous aimerions un [départ tardif] le dernier jour.",
                               "Pourrions-nous [garder la chambre jusqu'au soir] avant de partir ?"], "note": ["départ tardif"]},
    "surprise arrangement": {"sent": ["Aidez-moi à préparer [une surprise] pour cette soirée.",
                                      "J'aimerais [quelques petites attentions secrètes] pour ce jour-là."],
                             "note": ["surprise"]},
    "helicopter": {"sent": ["Nous envisagerions [un transfert en hélicoptère].", "Un [vol panoramique en hélicoptère] serait formidable."],
                   "note": ["hélicoptère"]},
    "baby cot": {"sent": ["Il nous faut un [lit bébé] dans la chambre.", "Merci d'ajouter un [berceau de voyage] pour le petit."],
                 "note": ["lit bébé"]},
    "short drives only": {"sent": ["Nous aimerions [limiter le temps de route].",
                                   "[Pas plus de trois heures de voiture] par jour, s'il vous plaît."],
                          "note": ["trajets courts"]},
    "French-speaking guide": {"sent": ["Il nous faudrait un [guide parlant bien français]."], "note": ["guide francophone"]},
}

# text that looks like a field but is not, so it must stay unlabelled
PAST_PLACES = {"Thailand": "la Thaïlande", "Bali": "Bali", "Morocco": "le Maroc", "Peru": "le Pérou",
               "Vietnam": "le Vietnam", "Japan": "le Japon", "Kenya": "le Kenya", "Oman": "Oman",
               "Jordan": "la Jordanie", "India": "l'Inde", "Nepal": "le Népal", "Sri Lanka": "le Sri Lanka",
               "Goa": "Goa", "Maldives": "les Maldives"}
HOME_CITIES = ["Nantes", "Bordeaux", "Lille", "Strasbourg", "Toulouse", "Rennes", "Namur", "Lausanne"]
DISTRACTORS = {
    "past": ["Nous avons fait {past} il y a quelques années et avons adoré.",
             "Nous avons visité {past} l'hiver dernier, nous cherchons donc autre chose.",
             "Nos voisins ont adoré {past} cette année.", "Nous connaissons déjà {past}, inutile de proposer cette destination."],
    "home": ["Nous habitons tout près de {home}.", "Nous vivons à {home}.",
             "Je vous écris de {home}, où il pleut depuis une semaine."],
    "other_date": ["J'ai parlé à l'un de vos collègues le {d} {m}.", "Mon passeport expire en {m} 2031.",
                   "Je suis en déplacement jusqu'au {d} {m}, mes réponses risquent de tarder."],
    "other_money": ["Notre dernier grand voyage nous a coûté environ {money}.", "Un ami a payé {money} la nuit pour une chambre là-bas.",
                    "Nous avons dépensé près de {money} pour un safari il y a des années."],
    "other_people": ["Nos deux chiens resteront chez ma sœur.", "Mes parents, qui ont plus de quatre-vingts ans, ne seront pas du voyage.",
                     "Nos trois grands enfants restent à la maison."],
    "part_stay": ["Peut-être {n} nuits {place_in} pour commencer.", "Nous imaginons {n} nuits {place_in}, puis le reste ailleurs."],
    "refusal": ["Inutile de prévoir {x}.", "Merci d'éviter {x}.", "Nous ferons l'impasse sur {x}."],
}

# place -> (name in French, text before it after "in" or "to", text before it as an object)
PLACES = {
    "India": ("Inde", "en ", "l'"), "Rajasthan": ("Rajasthan", "au ", "le "),
    "Madhya Pradesh": ("Madhya Pradesh", "au ", "le "), "Uttarakhand": ("Uttarakhand", "dans l'", "l'"),
    "Kerala": ("Kerala", "au ", "le "), "Tamil Nadu": ("Tamil Nadu", "au ", "le "),
    "Karnataka": ("Karnataka", "au ", "le "), "Nepal": ("Népal", "au ", "le "), "Bhutan": ("Bhoutan", "au ", "le "),
    "Ladakh": ("Ladakh", "au ", "le "), "Sikkim": ("Sikkim", "au ", "le "),
    "Himachal Pradesh": ("Himachal Pradesh", "dans l'", "l'"), "Sri Lanka": ("Sri Lanka", "au ", "le "),
    "Maldives": ("Maldives", "aux ", "les "), "Baa Atoll": ("atoll de Baa", "sur l'", "l'"),
    "Ari Atoll": ("atoll d'Ari", "sur l'", "l'"), "Kathmandu": ("Katmandou", "à ", ""),
    "Thimphu": ("Thimphou", "à ", ""), "Pondicherry": ("Pondichéry", "à ", ""), "Kochi": ("Cochin", "à ", ""),
    "Varanasi": ("Bénarès", "à ", ""),
}
CITIES = {"London": "Londres", "Edinburgh": "Édimbourg", "Geneva": "Genève", "Basel": "Bâle", "Frankfurt": "Francfort",
          "Hamburg": "Hambourg", "Vienna": "Vienne", "Brussels": "Bruxelles", "Dubai": "Dubaï"}


def place(name: str, case: str) -> list:
    shown, before_in, before_obj = PLACES.get(name, (name, "à ", ""))
    before = {"in": before_in, "to": before_in, "obj": before_obj, "note": ""}[case]
    span = S("DESTINATION", shown, name)
    return [T(before), span] if before else [span]
