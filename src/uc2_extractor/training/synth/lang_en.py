"""English wording for the synthetic enquiry generator.

Markup in phrases: [text] marks the labelled part, so articles can stay outside it. A phrase
without brackets is labelled as a whole. [[LABEL|text]] sets the label itself. {name} is a slot.
All wording here was written for this project; none of it comes from a test set. lint.py checks that.
"""

import re

CODE = "en"
JOIN = (", ", " and ")
# one partner is chosen per enquiry, so a text never mentions both a wife and a husband
PARTNERS = [{"partner": p} for p in ("my wife", "my husband", "my partner", "my fiancée", "my fiancé",
                                     "my boyfriend", "my girlfriend")]
PROPOSAL_PARTNERS = [{"partner": p} for p in ("my partner", "my boyfriend", "my girlfriend")]
NOT_SOLO = re.compile(r"\b(we|our|ours|ourselves|us)\b", re.IGNORECASE)
NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
       11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
       18: "eighteen", 19: "nineteen", 20: "twenty", 21: "twenty-one"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
MONTHS_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]

# kind -> list of (text of the date span, wordings around it)
DATE_FORMS = {
    "month": [("{m}", ["in {D}", "sometime in {D}", "during {D}", "this coming {D}"]),
              ("{m} {y}", ["in {D}", "around {D}"])],
    "early": [("early {m}", ["in {D}", "around {D}"]), ("the first days of {m}", ["in {D}"]),
              ("the beginning of {m}", ["at {D}", "around {D}"])],
    "mid": [("mid-{m}", ["in {D}", "around {D}", "from {D}"]), ("the middle of {m}", ["around {D}"])],
    "late": [("late {m}", ["in {D}", "from {D}"]), ("the last week of {m}", ["in {D}"]),
             ("the end of {m}", ["towards {D}", "at {D}"])],
    "range_month": [("{d} to {d2} {m}", ["from {D}", "{D}"]), ("{d}-{d2} {m}", ["{D}", "on {D}"]),
                    ("{m} {d}-{d2}", ["{D}", "on {D}"]), ("{d} to {d2} {m} {y}", ["from {D}"]),
                    ("{d} and {d2} {m}", ["between {D}"])],
    "range_cross": [("{d} {m} to {d2} {m2}", ["from {D}", "{D}"]), ("{m} {d} to {m2} {d2}", ["from {D}"]),
                    ("{d} {ma} - {d2} {ma2}", ["{D}"])],
    "single": [("{d} {m}", ["on {D}", "around {D}", "from {D}"]), ("{m} {d}", ["on {D}", "from {D}"]),
               ("{d} {m} {y}", ["on {D}"]), ("the {o} of {m}", ["on {D}", "around {D}"])],
    "numeric_ambiguous": [("{d}/{mo}", ["on {D}", "from {D}", "around {D}"]), ("{d}/{mo}/{y}", ["on {D}", "from {D}"])],
    "numeric_clear": [("{d}/{mo2}/{y}", ["on {D}", "from {D}"]), ("{y}-{mo2}-{dd}", ["on {D}", "from {D}"])],
    "month_range": [("{m} to {m2}", ["{D}", "from {D}"]), ("{ma}/{ma2}", ["{D}", "over {D}"]),
                    ("{m}/{m2}", ["{D}"])],
    "month_choice": [("{m} or {m2}", ["in {D}", "either in {D}"]), ("{ma} or {ma2}", ["in {D}"])],
}
DURATION_FORMS = {
    "nights": ["{n} nights", "{w} nights"],
    "days": ["{n} days", "{w} days"],
    "weeks": ["{w} weeks", "{n} weeks"],
    "one_week": ["a week", "one week", "seven days"],
    "two_weeks": ["two weeks", "a fortnight", "2 weeks"],
}
CURRENCY_FORMS = {
    "USD": ["${a}", "US${a}", "USD {a}", "{a} USD", "{a} dollars", "{a} US dollars"],
    "EUR": ["€{a}", "EUR {a}", "{a} euros", "{a} EUR"],
    "GBP": ["£{a}", "GBP {a}", "{a} pounds", "{a} sterling"],
    "CHF": ["CHF {a}", "{a} Swiss francs", "{a} CHF"],
    "INR": ["₹{a}", "INR {a}", "{a} rupees"],
}
LAKH_FORMS = ["{l} lakh rupees", "INR {l} lakh", "₹{l} lakh"]
THOUSAND_SEP = ","
K_FORMS = ["{k}k", "{k}K", "{k} thousand"]
K_WORDS = {3: "three", 4: "four", 5: "five", 6: "six", 8: "eight", 9: "nine", 10: "ten", 12: "twelve", 15: "fifteen",
           18: "eighteen", 20: "twenty", 25: "twenty-five", 30: "thirty", 40: "forty", 50: "fifty",
           75: "seventy-five", 100: "a hundred"}
WORD_AMOUNT = "{w} thousand"
BUDGET_SHAPES = {
    "about": ["around {x}", "about {x}", "in the region of {x}", "somewhere near {x}", "approximately {x}"],
    "exact": ["{x}"],
    "up_to": ["up to {x}", "no more than {x}", "{x} max", "at most {x}", "a ceiling of {x}", "{x} tops"],
    "at_least": ["at least {x}", "{x} or more", "{x} upwards"],
    "range": ["between {x} and {y}", "{x} to {y}", "{n}-{y}"],
}
BASIS = {"per_person": [" per person", " each", " a person", " pp"],
         "per_trip": [" in total", " for the whole trip", " altogether", " all told"]}
AGE_JOIN = (", ", " and ")

PARTY = {
    "couple": {
        "subject": ["[[ADULTS|{partner} and I]]", "[[ADULTS|{partner} and I]]", "[[PARTY|The two of us]]"],
        "np": ["[[ADULTS|two adults]]", "[[PARTY|two people]]", "[[PARTY|just the pair of us]]",
               "a [[ADULTS|married couple]]", "[[ADULTS|2 adults]]", "a [[PARTY|party of two]]"],
        "note": ["[[ADULTS|2 adults]]", "[[PARTY|2 pax]]", "[[ADULTS|couple]]", "[[PARTY|2 persons]]", "[[ADULTS|2 adlts]]"],
    },
    "solo": {
        "subject": [],
        "np": ["[[PARTY|just me]]", "[[PARTY|only myself]]", "[[ADULTS|one adult]]", "[[PARTY|one person]]"],
        "note": ["[[PARTY|1 pax]]", "[[ADULTS|1 adult]]", "[[PARTY|solo]]"],
    },
    "family": {
        "subject": ["[[ADULTS|{partner} and I]] and [[CHILDREN|our {c} children]]",
                    "[[ADULTS|{partner} and I]] with [[CHILDREN|our {c} kids]]"],
        "np": ["[[ADULTS|{a} adults]] and [[CHILDREN|{c} children]]", "[[ADULTS|{a} adults]] plus [[CHILDREN|{c} kids]]",
               "[[ADULTS|{a} grown-ups]] and [[CHILDREN|{c} youngsters]]", "a [[PARTY|family of {t}]]",
               "[[PARTY|{t} of us]], [[CHILDREN|{c} of them children]]"],
        "note": ["[[ADULTS|{a} adults]] + [[CHILDREN|{c} kids]]", "[[ADULTS|{a}A]] [[CHILDREN|{c}C]]",
                 "[[PARTY|family of {t}]]", "[[ADULTS|{a} ad]] / [[CHILDREN|{c} chd]]"],
        "one_child": ["[[ADULTS|{a} adults]] and [[CHILDREN|one child]]", "[[ADULTS|{a} adults]] and [[CHILDREN|our son]]",
                      "[[ADULTS|{a} adults]] and [[CHILDREN|our daughter]]"],
        "single_parent": ["[[ADULTS|one parent]] and [[CHILDREN|{c} children]]",
                          "[[ADULTS|a single mum]] with [[CHILDREN|{c} kids]]"],
    },
    "friends": {
        "subject": ["[[PARTY|{t} university friends]]", "[[PARTY|{t} old schoolfriends]]", "[[PARTY|A group of {t} friends]]"],
        "np": ["[[PARTY|{t} friends]]", "a [[PARTY|group of {t} friends]]", "[[ADULTS|{t} adults]]", "[[PARTY|{t} people]]"],
        "note": ["[[PARTY|{t} friends]]", "[[PARTY|{t} pax]]", "[[ADULTS|{t} adults]]"],
    },
    "two_couples": {
        "subject": ["[[ADULTS|{p} couples]]", "[[ADULTS|{p} retired couples]]"],
        "np": ["[[ADULTS|{p} couples]]", "[[PARTY|{t} people]]", "[[ADULTS|{t} adults]]"],
        "note": ["[[ADULTS|{p} couples]]", "[[PARTY|{t} pax]]"],
    },
    "multigen": {
        "subject": [],
        "np": ["[[PARTY|{t} of us]] across three generations", "[[ADULTS|{a} adults]] and [[CHILDREN|{c} grandchildren]]",
               "[[PARTY|{t} people]], grandparents included"],
        "note": ["[[PARTY|{t} pax]]", "[[ADULTS|{a} adults]] + [[CHILDREN|{c} kids]]"],
    },
    "group": {
        "subject": [],
        "np": ["a [[PARTY|group of {t}]]", "[[PARTY|{t} people]]", "[[PARTY|{t} guests]]", "a [[PARTY|party of {t}]]"],
        "note": ["[[PARTY|{t} pax]]", "[[PARTY|group of {t}]]"],
    },
}

SALUTATIONS = {
    "formal": ["To whom it may concern,", "Dear reservations team,", "Dear travel specialists,", "Good day,",
               "Dear Kenexoft team,"],
    "neutral": ["Hello,", "Hi,", "Good evening,", "Good morning,", "Hello Kenexoft,"],
    "casual": ["hi", "hey", "Hiya,", "hello!", ""],
}
SIGN_OFFS = {
    "formal": ["Yours faithfully,", "With thanks,", "Many thanks in advance,", "Best regards,", "Sincerely,"],
    "neutral": ["Thanks,", "Cheers,", "All the best,", "Best,", "Warm wishes,"],
    "casual": ["thx", "cheers", "ta", ""],
}
NAMES = ["Eleanor Whitcombe", "Graham Aldous", "Rosalind Keane", "Dominic Farrow", "Imogen Pryce", "Harriet Lowell",
         "Oliver Tennant", "Fiona Maddox", "Callum Reid", "Beatrice Holt", "Rupert Sands", "Tessa Marlowe"]
EXTRA_LINES = ["Sent from my iPad", "Sent from my phone, so please excuse any slips", "Mobile: +44 7700 900123"]

OPENERS = [
    "We came across your company in a magazine feature and liked what we read.",
    "A colleague who travelled with you last year passed on your name.",
    "I have been reading through your website this evening and thought I would simply get in touch.",
    "Friends of ours keep recommending you, so here I am.",
    "We are at the very early stages of planning and would welcome some guidance.",
    "This is our first time using a travel specialist, so please bear with me.",
    "I will try to keep this brief, though I rarely manage it.",
    "We have wanted to do this for a long while and the time finally feels right.",
    "Apologies for writing at such length.",
]
FILLERS = [
    "We are in no hurry and prefer to stay put rather than change hotels every night.",
    "Pace matters to us more than ticking off sights.",
    "We like to start early and keep the afternoons quiet.",
    "We would rather see less and see it well.",
    "We do not mind paying for quality but dislike anything showy.",
    "Neither of us enjoys crowds.",
    "Good coffee in the morning would be a bonus.",
    "We have travelled a fair amount in Europe but never this far east.",
    "I should say that heat does not bother us much.",
    "Reliable wifi matters a little, as one of us may need to check in with work.",
]
CLOSERS = [
    "Could you let me know what you would suggest?",
    "Please send over a rough outline and an idea of cost.",
    "I look forward to hearing from you.",
    "Happy to speak on the phone if that is easier.",
    "Do tell me if you need anything else from us.",
    "A first draft by email would be wonderful.",
]
VAGUE = ["Please send me some thoughts.", "What would you recommend?", "Where would you start?",
         "Can you help?", "Any suggestions welcome."]

INTRO = [
    "{subject} would love to spend {dur} in {dest}.",
    "{subject} are thinking about {dest} {date}.",
    "{subject} are planning {dur} in {dest} {date}.",
    "We are {party} and we are dreaming of {dest}.",
    "We are {party}, looking for {dur} in {dest} {date}.",
    "I am arranging a trip to {dest} for {party}.",
    "Could you build something for {party} in {dest}?",
    "Would you be able to help with a trip to {dest} {date}?",
    "We have set our hearts on {dest}.",
    "Our thoughts have turned to {dest} for {dur}.",
    "What might you suggest for {dur} in {dest}?",
    "We are hoping to reach {dest} {date}.",
    "I would like to ask about {dur} in {dest} for {party}.",
    "We are longing to see {dest}, and we could get away {date}.",
]
INTRO_SOLO = [
    "I would love to spend {dur} in {dest}.",
    "I am thinking about {dest} {date}.",
    "I am planning {dur} in {dest} {date}.",
    "I am hoping to reach {dest} {date}.",
    "What might you suggest for {dur} in {dest}?",
    "I have set my heart on {dest}.",
]
SENT_SOLO = {
    "dest": ["I have {dest} in mind.", "The plan is {dest}."],
    "dur": ["I have {dur} available.", "I can get away for {dur}.", "Length of stay: {dur}."],
    "date": ["I can travel {date}.", "The best time for me would be {date}.", "I would set off {date}."],
    "date_dur": ["I hope to leave {date} and be away for {dur}."],
    "party": ["It is for {party}.", "This one is for {party}."],
    "budget": ["My budget is {budget}.", "I have set aside {budget} for this.", "I would like to keep it to {budget}."],
    "accommodation": ["I would like to stay in {x}.", "I love {x}.", "I am after {x}."],
    "flights": ["I need {x}.", "Please include {x}."],
    "flights_origin": ["I would want {x} from {origin}."],
    "origin": ["I would be flying out of {origin}."],
    "interests": ["I would love to include {x}.", "I am drawn to {x}.", "I am very keen on {x}."],
}
SENT = {
    "dest": ["We have {dest} in mind.", "The plan is {dest}.", "We keep coming back to {dest}.",
             "Our first choice would be {dest}."],
    "dur": ["We have {dur} available.", "We can get away for {dur}.", "We are thinking {dur} in all.",
            "Length of stay: {dur}.", "We could manage {dur}."],
    "date": ["We can travel {date}.", "The best time for us would be {date}.", "We are free {date}.",
             "We would set off {date}.", "Our dates are not fixed, but we are looking at {date}."],
    "date_dur": ["We hope to leave {date} and be away for {dur}.", "Roughly speaking, {dur} {date}."],
    "party": ["We will be {party}.", "There will be {party}.", "It is for {party}.", "Numbers-wise, we are {party}."],
    "budget": ["Our budget is {budget}.", "We have set aside {budget} for this.", "We were thinking of {budget}.",
               "Budget-wise, {budget} feels right.", "I would say {budget} is comfortable for us.",
               "We would like to keep it to {budget}."],
    "accommodation": ["We would like to stay in {x}.", "We love {x}.", "On the hotel side, {x} would suit us.",
                      "Please look at {x}.", "We are after {x}."],
    "flights": ["We need {x}.", "Please include {x}.", "For the flights we would want {x}."],
    "flights_origin": ["We would want {x} from {origin}.", "Please price {x} out of {origin}."],
    "origin": ["We would be flying out of {origin}.", "We would leave from {origin} by plane.",
               "Flights would need to start in {origin}."],
    "interests": ["We would love to include {x}.", "We are drawn to {x}.", "High on our list: {x}.",
                  "We are very keen on {x}.", "Please make room for {x}."],
    "ages_children": ["The children are {ages}.", "The kids will be {ages} when we travel.", "Their ages are {ages}."],
    "ages_adults": ["We are {ages}.", "For reference, we are {ages} years old."],
    "ages_one": ["Our child is {ages}.", "Our little one will be {ages} by then."],
    "age_solo": ["I am {ages}.", "I turned {ages} this year, for what it is worth."],
}
NOTE_OPENERS = ["Trip enquiry:", "Enquiry -", "Looking at:", "Quote please:", ""]
NOTE_CLOSERS = ["thanks", "pls advise", "pls send ideas", "ideas?", ""]
NOTE_LABELS = {"budget": ["budget", "budget:", "spend", ""]}
NOTE_ORIGIN = ["from ", "ex ", "departing "]
ALT_BUDGET = [" (roughly {x})", " (about {x} at today's rate)"]

ACCOMMODATION = {
    "overwater villa": {"np": ["a [bungalow out over the lagoon]", "an [over-water bungalow]", "a [villa on stilts]"],
                        "note": ["water villa", "overwater bungalow"]},
    "private pool": {"np": ["a [plunge pool of our own]", "a room with a [private pool]", "our [own small pool]"],
                     "note": ["private pool", "own pool"]},
    "villa": {"np": ["a [villa]", "a [whole villa to ourselves]", "a [private house with staff]"], "note": ["villa"]},
    "heritage or palace hotel": {"np": ["a [restored haveli]", "a [former royal residence]", "a [converted palace]",
                                        "a [heritage property]"], "note": ["heritage hotels", "haveli"]},
    "boutique hotel": {"np": ["a [boutique hotel]", "a [small, stylish hotel]", "a [tiny design hotel]",
                              "[intimate places with a dozen rooms at most]"], "note": ["boutique"]},
    "5-star": {"np": ["[five-star] places", "[5-star] hotels", "[five star] comfort"], "note": ["5*", "5 star"]},
    "suite": {"np": ["a [suite]", "a [junior suite]", "a [proper suite]"], "note": ["suite"]},
    "beachfront": {"np": ["a room [right on the sand]", "somewhere [steps from the sea]", "a [beachfront] room"],
                   "note": ["beachfront", "on the beach"]},
    "tented camp": {"np": ["a [luxury tented camp]", "[safari-style tents]", "a spot of [glamping]"],
                    "note": ["tented camp", "glamping"]},
    "connecting rooms": {"np": ["[connecting rooms]", "[adjoining rooms with a door between them]"],
                         "note": ["connecting rms"]},
    "family room": {"np": ["a [family room]", "[one big room for all of us]"], "note": ["family room"]},
    "eco-lodge": {"np": ["an [eco-lodge]", "a [solar-powered lodge]", "somewhere [low-impact and sustainable]"],
                  "note": ["eco lodge"]},
    "houseboat": {"np": ["a [houseboat]", "a [traditional rice barge]", "a night aboard a [kettuvallam]"],
                  "note": ["houseboat"]},
    "safari lodge": {"np": ["a [safari lodge]", "a [jungle lodge]", "a [lodge on the edge of the national park]"],
                     "note": ["safari lodge"]},
}
FLIGHTS = {
    "business class": {"np": ["[business class]", "[lie-flat seats]", "[flat beds at the front of the plane]"],
                       "note": ["biz class", "business"]},
    "first class": {"np": ["[first class]", "[first-class seats]"], "note": ["first class", "F class"]},
    "premium economy": {"np": ["[premium economy]", "[premium economy seats]"], "note": ["prem eco"]},
    "economy class": {"np": ["[economy seats]", "[standard economy]"], "note": ["economy", "eco"]},
    "direct flights": {"np": ["[direct flights]", "[non-stop flights]", "[flights without a stopover]"],
                       "note": ["direct", "nonstop only"]},
    "domestic flights": {"np": ["[internal flights]", "[domestic flights]", "[short hops by air between cities]"],
                         "note": ["domestic flts", "internal flights"]},
}
NO_FLIGHTS = {"sent": ["We [have booked our own flights].", "Our [flights are taken care of].",
                       "This would be [land arrangements only].", "We [will use air miles for the flights]."],
              "note": ["land only", "flights sorted", "own flights"]}
INTERESTS = {
    "wildlife": {"np": ["[tiger safaris]", "[seeing big cats in the wild]", "[game drives at dawn]", "[wild elephants]",
                        "[anything to do with birdlife]"], "note": ["wildlife", "tigers"]},
    "culture and heritage": {"np": ["[ancient temples]", "[UNESCO sites]", "[Mughal history]", "[colonial-era towns]",
                                    "[sculpture and architecture]"], "note": ["culture", "history"]},
    "wellness and yoga": {"np": ["[daily yoga]", "[an ayurveda programme]", "[a good massage or two]",
                             "[spa treatments]", "[proper rest and meditation]"], "note": ["yoga", "ayurveda"]},
    "beach": {"np": ["[lazy days on the beach]", "[a few days by the sea]", "[swimming in warm water]",
                     "[time on the sand]"], "note": ["beach"]},
    "food and cooking": {"np": ["[street food tours]", "[a cookery lesson with a local family]",
                                "[eating where the locals eat]", "[spice markets and home cooking]"],
                         "note": ["food", "cookery"]},
    "trekking and hiking": {"np": ["[a multi-day trek]", "[walking in the foothills]", "[hikes between villages]",
                                   "[mountain trails]"], "note": ["trekking", "hiking"]},
    "photography": {"np": ["[photography]", "[getting good shots at sunrise]", "[photographing village life]"],
                    "note": ["photography"]},
    "diving and snorkelling": {"np": ["[scuba diving]", "[snorkelling with turtles]", "[a dive course]"],
                               "note": ["diving", "snorkelling"]},
    "shopping": {"np": ["[textile shopping]", "[browsing the bazaars]", "[hunting for antiques]",
                        "[buying silk and carpets]"], "note": ["shopping"]},
    "golf": {"np": ["[a round or two of golf]", "[golf]"], "note": ["golf"]},
    "tea plantations": {"np": ["[a stay on a working tea estate]", "[tea tasting in the hills]",
                               "[the tea gardens]"], "note": ["tea estates"]},
    "luxury train": {"np": ["[a luxury train journey]", "[a few nights on one of the royal trains]"],
                     "note": ["luxury train"]},
    "festivals": {"np": ["[Holi]", "[Diwali celebrations]", "[a local festival]", "[the camel fair at Pushkar]"],
                  "note": ["festivals"]},
    "spirituality": {"np": ["[morning prayers at a monastery]", "[the ghats at dawn]", "[sacred sites]",
                            "[time in a Buddhist monastery]"], "note": ["monasteries", "spiritual sites"]},
}
DIETARY = {
    "vegetarian": {"sent": ["One of us is [vegetarian].", "{partner} [avoids meat and fish].",
                            "I [have never eaten meat].", "We [eat only vegetarian food]."],
                   "note": ["vegetarian", "veggie", "no meat"]},
    "vegan": {"sent": ["I am [vegan].", "{partner} is [strictly plant-based].", "We [avoid all animal products]."],
              "note": ["vegan"]},
    "gluten-free": {"sent": ["{partner} [has coeliac disease].", "One of us [needs a strict wheat-free diet].",
                             "I eat [gluten-free]."], "note": ["gluten free", "GF", "coeliac"]},
    "lactose-free": {"sent": ["I [am lactose intolerant].", "[Milk does not agree with] {partner}.",
                              "My daughter needs [dairy-free] meals."], "note": ["dairy free", "no lactose"]},
    "nut allergy": {"sent": ["{partner} [carries an EpiPen for peanuts].", "I have [a severe nut allergy].",
                             "[Cashews and peanuts are dangerous] for {partner}."], "note": ["nut allergy"]},
    "shellfish allergy": {"sent": ["I am [allergic to prawns].", "{partner} [reacts badly to crab and lobster].",
                                   "One of us has [a shellfish allergy]."], "note": ["shellfish allergy"]},
    "halal": {"sent": ["We need [halal] meals.", "I eat [only halal meat]."], "note": ["halal"]},
    "kosher": {"sent": ["We need [strictly kosher] meals.", "My food must be [kosher]."], "note": ["kosher"]},
    "jain": {"sent": ["We follow a [Jain diet], so no root vegetables.", "My mother needs [Jain meals]."],
             "note": ["Jain food"]},
    "diabetic": {"sent": ["My father is [diabetic].", "I have [type 2 diabetes].", "{partner} [has to keep an eye on sugar]."],
                 "note": ["diabetic"]},
}
OCCASIONS = {
    "honeymoon": {"sent": ["This will be our [honeymoon].", "We are [getting married in the spring and want to go straight afterwards].",
                           "It would be our [first trip as a married couple]."], "note": ["honeymoon", "newlyweds"]},
    "anniversary": {"sent": ["We are celebrating [25 years of marriage].", "It is our [ruby anniversary].",
                             "The trip marks our [wedding anniversary]."], "note": ["anniversary"]},
    "birthday": {"sent": ["It is for [my 60th birthday].", "I am [turning fifty] and want to mark it.",
                          "{partner} has [a milestone birthday] that month."], "note": ["birthday trip", "60th bday"]},
    "retirement": {"sent": ["This is [my retirement present to myself].", "I have just [stopped working after 35 years].",
                            "We are celebrating [the end of a 40-year career]."], "note": ["retirement trip"]},
    "proposal": {"sent": ["I am planning to [ask {partner} to marry me] on this trip.",
                          "I would like to [go down on one knee] somewhere beautiful.",
                          "I intend to [pop the question] while we are there."], "note": ["proposal trip"]},
    "graduation": {"sent": ["It is a [graduation gift] for our daughter.", "I have just [finished medical school]."],
                   "note": ["graduation trip"]},
    "family reunion": {"sent": ["It is a [family reunion].", "The idea is [to get all the cousins together]."],
                       "note": ["family reunion"]},
    "babymoon": {"sent": ["It would be our [last trip before the baby arrives].", "We are looking for a [babymoon]."],
                 "note": ["babymoon"]},
    "Christmas or New Year": {"sent": ["We would like to be away over [Christmas].", "We want to see in [New Year's Eve] somewhere warm.",
                                       "We plan to escape for [the festive season]."], "note": ["Xmas", "New Year"]},
}
SPECIAL = {
    "private guide": {"sent": ["We would like [our own guide].", "Please arrange a [personal guide].",
                               "A [guide just for us] would be ideal."], "note": ["private guide"]},
    "private driver": {"sent": ["We would like [someone else to do the driving].", "Please include [our own car and driver].",
                                "A [private driver] throughout, please."], "note": ["private driver", "car + driver"]},
    "accessibility needs": {"sent": ["{partner} [uses a wheelchair].", "We need [step-free access].",
                                     "My mother [walks with two sticks], so stairs are hard."], "note": ["wheelchair access"]},
    "airport transfers": {"sent": ["Please include [transfers from the airport].", "We would like [a car waiting at arrivals].",
                                   "We need [collecting from the airport]."], "note": ["airport pickup"]},
    "childcare": {"sent": ["A [nanny] for the evenings would help.", "Is there [babysitting]?",
                           "We would like [someone to mind the children] after dinner."], "note": ["babysitter", "kids club"]},
    "butler service": {"sent": ["We have enjoyed [butler service] before.", "A [personal butler] would be a treat."],
                       "note": ["butler"]},
    "late checkout": {"sent": ["We would like a [late checkout] on the last day.",
                               "Could we [keep the room until the evening] before we leave?"], "note": ["late checkout"]},
    "surprise arrangement": {"sent": ["Please help me plan [a surprise] for that evening.",
                                      "I would love [some secret extra touches] for the day itself."], "note": ["surprise"]},
    "helicopter": {"sent": ["We would consider [a helicopter transfer].", "A [scenic helicopter flight] would be special."],
                   "note": ["helicopter"]},
    "baby cot": {"sent": ["We need a [cot] in the room.", "Please add a [travel crib] for the baby."], "note": ["baby cot"]},
    "short drives only": {"sent": ["We would like [to keep road time short].", "Please [nothing over three hours in a car]."],
                          "note": ["short drives"]},
    "French-speaking guide": {"sent": ["We would need a [guide who speaks good French]."], "note": ["French guide"]},
    "German-speaking guide": {"sent": ["We would need a [guide fluent in German]."], "note": ["German guide"]},
}

# text that looks like a field but is not, so it must stay unlabelled
PAST_PLACES = {"Thailand": "Thailand", "Bali": "Bali", "Morocco": "Morocco", "Peru": "Peru", "Vietnam": "Vietnam",
               "Japan": "Japan", "Kenya": "Kenya", "Oman": "Oman", "Jordan": "Jordan", "India": "India",
               "Nepal": "Nepal", "Sri Lanka": "Sri Lanka", "Goa": "Goa", "Maldives": "the Maldives"}
HOME_CITIES = ["Bristol", "Leeds", "Glasgow", "Cork", "Norwich", "Boston", "Melbourne", "Reading"]
DISTRACTORS = {
    "past": ["We did {past} a few years ago and loved it.", "We were in {past} last winter, so we want something different.",
             "Our neighbours have just come back from {past}.", "We have already seen {past}, so please leave it out."],
    "home": ["We live just outside {home}.", "We are based in {home}.", "I am writing from {home}, where it has rained all week."],
    "other_date": ["I spoke to one of your colleagues on {d} {m}.", "My passport runs out in {m} 2031.",
                   "I am away for work until {d} {m}, so replies may be slow."],
    "other_money": ["Our last big trip cost around {money}.", "A friend paid {money} a night for a room there.",
                    "We spent about {money} on a safari years ago."],
    "other_people": ["Our two dogs will stay with my sister.", "My parents, who are in their eighties, will not be joining us.",
                     "Our three grown-up children are staying at home."],
    "part_stay": ["Perhaps {n} nights in {place_in} to begin with.", "We imagine {n} nights in {place_in} and the rest elsewhere."],
    "refusal": ["We are not interested in {x}.", "Please leave out {x}.", "We do not need {x}."],
}
