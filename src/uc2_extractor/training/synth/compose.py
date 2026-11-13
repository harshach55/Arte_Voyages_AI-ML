"""Write a sampled Trip as an enquiry, using the wording of one language module.

Three styles: long client emails, medium web-form messages, and short telegraphic notes.
Every piece of text is a Part, so labels travel with the text through typos and case changes.
"""

from __future__ import annotations

import dataclasses
import random
import re
import zlib
from typing import Optional

from .parts import (
    Fragment,
    S,
    T,
    add_typos,
    capitalise,
    fill,
    german_ascii,
    join,
    map_text,
    plain,
    slots_in,
    strip_accents,
    tidy,
)
from .trips import DateInfo, Trip

_MARK = re.compile(r"\[\[(\w+)\|(.*?)\]\]|\[(.*?)\]")
_PARTNER = re.compile(r"\{partner")
FIELD_OF_SLOT = {"subject": "party", "party": "party", "dest": "dest", "dest_in": "dest", "dest_to": "dest",
                 "dest_obj": "dest", "dur": "dur", "date": "date", "budget": "budget"}
REGISTERS = {"long": (("formal", "neutral"), (6, 4)), "medium": (("neutral", "casual"), (7, 3)),
             "short": (("casual",), (1,))}
NOTE_SEPARATORS = (", ", ", ", " / ", " - ", "; ", " | ", "\n")


def template_id(template: str) -> int:
    return zlib.crc32(template.encode("utf-8"))


class Writer:
    def __init__(self, lang, rng: random.Random):
        self.L = lang
        self.r = rng
        self.words_for_numbers = False
        self.used: list[int] = []
        self.solo = False  # one traveller, so no "we"
        self.has_partner = True  # a partner may be mentioned
        self.partner: dict[str, str] = {}
        self.done: set[str] = set()

    # --- small helpers ------------------------------------------------------------------------

    def pick(self, options):
        return self.r.choice(list(options))

    def num(self, n: int) -> str:
        if self.words_for_numbers and n in self.L.NUM:
            return self.L.NUM[n]
        return str(n)

    def personal(self, text: str) -> str:
        """Use the same partner all through one enquiry."""
        for key, value in self.partner.items():
            text = text.replace("{" + key + "}", value)
        return text

    def fitting(self, options) -> list[str]:
        """Leave out wording that does not fit this traveller."""
        out = list(options)
        if self.solo:
            out = [o for o in out if not self.L.NOT_SOLO.search(o)]
        if not self.has_partner:
            out = [o for o in out if not _PARTNER.search(o)]
        return out

    def parse(self, markup: str, label: Optional[str] = None, value=None, values: Optional[dict] = None) -> Fragment:
        """Turn phrase markup into parts. A phrase with no brackets is labelled as a whole."""
        values = values or {}
        parts: Fragment = []
        pos = 0
        for m in _MARK.finditer(markup):
            if m.start() > pos:
                parts.append(T(markup[pos:m.start()]))
            if m.group(1):
                parts.append(S(m.group(1), m.group(2), values.get(m.group(1), value)))
            else:
                parts.append(S(label, m.group(3), value))
            pos = m.end()
        if not parts:
            return [S(label, markup, value)] if label else [T(markup)]
        if pos < len(markup):
            parts.append(T(markup[pos:]))
        return parts

    def sentence(self, template: str, slots: Optional[dict] = None) -> Fragment:
        self.used.append(template_id(template))
        return capitalise(fill(self.personal(template), slots or {}))

    def label_sentence(self, template: str, field: str, label: str) -> Fragment:
        self.used.append(template_id(template))
        return capitalise(self.parse(self.personal(template), field, label))

    def add_sentence(self, out: list[Fragment], key: str, slots: dict) -> None:
        """Add a sentence for a field, if this traveller has a frame that fits."""
        table = self.L.SENT_SOLO if self.solo and key in self.L.SENT_SOLO else self.L.SENT
        options = self.fitting(table[key])
        if options:
            out.append(self.sentence(self.pick(options), slots))

    # --- field text ---------------------------------------------------------------------------

    def place(self, name: str, case: str) -> Fragment:
        return self.L.place(name, case) if hasattr(self.L, "place") else self._place_en(name, case)

    def _place_en(self, name: str, case: str) -> Fragment:
        span = S("DESTINATION", name, name)
        return [T("the "), span] if name == "Maldives" and case != "note" else [span]

    def places(self, trip: Trip, case: str) -> Fragment:
        fragments = [self.place(p, case) for p in trip.places]
        if getattr(self.L, "SHARE_PREPOSITION", False) and len(fragments) > 1:
            lead = fragments[0][0]
            if not lead.label and all(f[0] == lead for f in fragments):  # nach Delhi, Agra und Jaipur
                fragments = fragments[:1] + [f[1:] for f in fragments[1:]]
        return join(fragments, *self.L.JOIN)

    def date_span(self, d: DateInfo) -> tuple[str, list[str]]:
        L = self.L
        key = d.part if d.kind == "part_month" else d.kind
        forms = [f for f in L.DATE_FORMS[key] if d.year or "{y}" not in f[0]] or L.DATE_FORMS[key]
        text, wraps = self.pick(forms)
        year = d.year or 2027
        values = dict(m=L.MONTHS[d.month - 1], m2=L.MONTHS[d.month2 - 1], ma=L.MONTHS_ABBR[d.month - 1],
                      ma2=L.MONTHS_ABBR[d.month2 - 1], d=self.L.day(d.day) if hasattr(L, "day") else d.day,
                      n=d.day, d2=d.day2, dd=f"{d.day:02d}", mo=d.month, mo2=f"{d.month:02d}", y=year,
                      o=self._ordinal(d.day), dm=L.OF_MONTH[d.month - 1] if hasattr(L, "OF_MONTH") else "")
        return text.format(**values), wraps

    @staticmethod
    def _ordinal(n: int) -> str:
        return f"{n}{'th' if 11 <= n <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

    def date_fragment(self, d: DateInfo) -> Fragment:
        text, wraps = self.date_span(d)
        before, _, after = self.pick(wraps).partition("{D}")
        out: Fragment = [T(before)] if before else []
        out.append(S("DATE", text, dataclasses.asdict(d)))
        return out + ([T(after)] if after else [])

    def date_note(self, d: DateInfo) -> Fragment:
        return [S("DATE", self.date_span(d)[0], dataclasses.asdict(d))]

    def duration(self, n: int, unit: str) -> Fragment:
        F = self.L.DURATION_FORMS
        if unit == "weeks" and n == 1:
            text = self.pick(F["one_week"])
        elif unit == "weeks" and n == 2:
            text = self.pick(F["two_weeks"])
        else:
            text = self.pick(F[unit]).format(n=n, w=self.L.NUM.get(n, n))
        out: Fragment = [S("DURATION", text, f"{n} {unit}")]
        if unit == "weeks" and self.r.random() < .12:  # the same length in another unit
            nights = n * 7 - 1
            shown = self.pick(F["nights"]).format(n=nights, w=self.L.NUM.get(nights, nights))
            out += [T(" ("), S("DURATION", shown, f"{nights} nights"), T(")")]
        return out

    def grouped(self, amount: int, currency: str) -> str:
        sep = self.L.THOUSAND_SEP
        if currency == "CHF" and self.r.random() < .4:
            sep = "'"
        if currency == "INR" and self.L.CODE == "en" and self.r.random() < .5:
            head, tail = str(amount)[:-3], str(amount)[-3:]
            pairs = []
            while len(head) > 2:
                pairs.insert(0, head[-2:])
                head = head[:-2]
            return ",".join([head] + pairs + [tail])
        return f"{amount:,}".replace(",", sep)

    def amount(self, amount: int, currency: str, style: str) -> str:
        L = self.L
        if currency == "INR" and amount >= 100000 and style in ("k", "words") and amount % 100000 == 0:
            return self.pick(L.LAKH_FORMS).format(l=amount // 100000)
        word_forms = [f for f in L.CURRENCY_FORMS[currency] if f.startswith("{a} ")]
        k = amount // 1000
        if style == "words" and amount % 1000 == 0 and k in L.K_WORDS and word_forms:
            return self.pick(word_forms).format(a=L.WORD_AMOUNT.format(w=L.K_WORDS[k]))
        if style == "k" and amount % 1000 == 0 and amount < 1000000:
            return self.pick(L.CURRENCY_FORMS[currency]).format(a=self.pick(L.K_FORMS).format(k=k))
        if style == "plain":
            return self.pick(L.CURRENCY_FORMS[currency]).format(a=amount)
        return self.pick(L.CURRENCY_FORMS[currency]).format(a=self.grouped(amount, currency))

    def budget(self, trip: Trip) -> Fragment:
        b = trip.budget
        style = self.r.choices(("grouped", "plain", "k", "words"), weights=(6, 1, 3, 1))[0]
        value = dict(low=b.low, high=b.high, currency=b.currency, basis=b.basis)
        shape = self.pick(self.L.BUDGET_SHAPES[b.shape])
        base = b.high if b.low is None else b.low
        x = self.amount(base, b.currency, style)
        y = self.amount(b.high, b.currency, style) if b.high and b.high != base else x
        text = shape.format(x=x, y=y, n=self.grouped(base, b.currency))
        if b.basis:
            text += self.pick(self.L.BASIS[b.basis])
        out: Fragment = [S("BUDGET", text, value)]
        if b.currency != "INR" and b.shape in ("about", "exact", "up_to") and self.r.random() < .12:
            # a second figure in another currency, as clients abroad often add
            other = self.pick(c for c in ("USD", "EUR", "GBP", "CHF") if c != b.currency)
            second = int(round(base * self.pick((.9, 1.1, 1.2)), -3))
            before, _, after = self.pick(self.L.ALT_BUDGET).partition("{x}")
            shown = self.amount(second, other, "grouped")
            out += [T(before), S("BUDGET", shown, dict(value, low=second, high=second, currency=other)), T(after)]
        return out

    def party(self, trip: Trip, kind: str) -> Optional[Fragment]:
        table = self.L.PARTY[trip.persona]
        options = list(table.get(kind, []))
        if trip.persona == "family":
            if kind == "subject" and (trip.adults != 2 or (trip.children or 0) < 2):
                options = []
            elif kind in ("np", "note") and trip.children == 1:
                options = table["one_child"] + [o for o in options if "{c}" not in o]
            elif kind in ("np", "note") and trip.adults == 1:
                options = table["single_parent"]
        if trip.persona == "group" and trip.adults is None:
            options = [o for o in options if "ADULTS" not in o]
        if not options:
            return None
        num = str if kind == "note" else self.num
        nums = dict(a=num(trip.adults or 0), c=num(trip.children or 0), t=num(trip.total or 0),
                    p=num((trip.adults or 0) // 2))
        values = {"ADULTS": trip.adults, "CHILDREN": trip.children, "PARTY": trip.total}
        return self.parse(self.personal(self.pick(options)).format(**nums), values=values)

    def ages_text(self, ages: list[int]) -> str:
        sep, last = self.L.AGE_JOIN
        text = sep.join(str(a) for a in ages[:-1])
        return f"{text}{last}{ages[-1]}" if len(ages) > 1 else str(ages[0])

    def label_phrase(self, table: dict, label: str, kind: str, field: str) -> Fragment:
        options = self.fitting(table[label][kind]) or table[label][kind]
        return self.parse(self.personal(self.pick(options)), field, label)

    def phrase_list(self, table: dict, labels: list[str], kind: str, field: str) -> Fragment:
        return join([self.label_phrase(table, x, kind, field) for x in labels], *self.L.JOIN)

    # --- text that looks like a field but is not ---------------------------------------------

    def distractor(self, trip: Trip) -> Optional[Fragment]:
        L = self.L
        kind = self.pick(L.DISTRACTORS)
        options = [o for o in self.fitting(L.DISTRACTORS[kind]) if o not in self.done]
        if not options:
            return None
        template = self.pick(options)
        self.done.add(template)  # never the same distractor twice in one enquiry
        slots: dict[str, Fragment] = {}
        if kind == "past":
            places = [text for name, text in L.PAST_PLACES.items() if name not in trip.places]
            slots["past"] = [T(self.pick(places))]
        elif kind == "home":
            slots["home"] = [T(self.pick(L.HOME_CITIES))]
        elif kind == "other_date":
            slots.update(d=[T(str(self.r.randint(1, 28)))], m=[T(self.pick(L.MONTHS))])
        elif kind == "other_money":
            currency = trip.budget.currency if trip.budget else "EUR"
            slots["money"] = [T(self.amount(self.pick((800, 900, 1200, 6000, 9000)), currency, "grouped"))]
        elif kind == "part_stay":
            if not trip.places or trip.duration is None:
                return None
            # the length of one stop stays plain, but the place is still a destination
            slots.update(n=[T(str(self.r.randint(2, 4)))], place_in=self.place(self.pick(trip.places), "in"))
        elif kind == "refusal":
            refused = self.pick(k for k in L.INTERESTS if k not in trip.interests)
            slots["x"] = plain(self.label_phrase(L.INTERESTS, refused, "np", "INTEREST"))
        return self.sentence(template, slots)

    # --- whole enquiries ---------------------------------------------------------------------

    def _slots(self, trip: Trip) -> dict[str, Fragment]:
        slots: dict[str, Fragment] = {}
        if trip.places:
            for name, case in (("dest", "in"), ("dest_in", "in"), ("dest_to", "to"), ("dest_obj", "obj")):
                slots[name] = self.places(trip, case)
        if trip.total:
            np = self.party(trip, "np")
            if np:
                slots["party"] = np
            subject = self.party(trip, "subject")
            if subject:
                slots["subject"] = subject
        if trip.duration:
            slots["dur"] = self.duration(*trip.duration)
        if trip.date:
            slots["date"] = self.date_fragment(trip.date)
        if trip.budget:
            slots["budget"] = self.budget(trip)
        return slots

    def _intro(self, slots: dict, remaining: set[str]) -> Optional[Fragment]:
        usable = []
        for t in (self.L.INTRO_SOLO if self.solo else self.L.INTRO):
            needed = slots_in(t)
            if needed and needed <= set(slots) and {FIELD_OF_SLOT[s] for s in needed} <= remaining:
                usable.append((t, len({FIELD_OF_SLOT[s] for s in needed})))
        if not usable:
            return None
        t = self.r.choices([u[0] for u in usable], weights=[u[1] ** 2 for u in usable])[0]
        remaining -= {FIELD_OF_SLOT[s] for s in slots_in(t)}
        return self.sentence(t, slots)

    def _field_sentences(self, trip: Trip, slots: dict, remaining: set[str]) -> list[Fragment]:
        L, out = self.L, []
        if {"date", "dur"} <= remaining and self.r.random() < .4:
            self.add_sentence(out, "date_dur", slots)
            remaining -= {"date", "dur"}
        for field in ("dest", "party", "dur", "date", "budget"):
            if field in remaining and field in slots:
                self.add_sentence(out, field, slots)
        if trip.ages:
            if trip.persona == "solo":
                key = "age_solo"
            elif trip.persona == "family":
                key = "ages_children" if len(trip.ages) > 1 else "ages_one"
            else:
                key = "ages_adults"
            self.add_sentence(out, key, {"ages": [S("AGE", self.ages_text(trip.ages), trip.ages)]})
        if trip.accommodation:
            x = self.phrase_list(L.ACCOMMODATION, trip.accommodation, "np", "ACCOMMODATION")
            self.add_sentence(out, "accommodation", {"x": x})
        if trip.no_flights:
            options = self.fitting(L.NO_FLIGHTS["sent"])
            if options:
                out.append(self.label_sentence(self.pick(options), "FLIGHT", "flights not needed"))
        elif trip.flights:
            x = self.phrase_list(L.FLIGHTS, trip.flights, "np", "FLIGHT")
            if trip.origin:
                origin = [S("ORIGIN", self.city(trip.origin), trip.origin)]
                self.add_sentence(out, "flights_origin", {"x": x, "origin": origin})
            else:
                self.add_sentence(out, "flights", {"x": x})
        elif trip.origin:
            self.add_sentence(out, "origin", {"origin": [S("ORIGIN", self.city(trip.origin), trip.origin)]})
        if trip.interests:
            x = self.phrase_list(L.INTERESTS, trip.interests, "np", "INTEREST")
            self.add_sentence(out, "interests", {"x": x})
        for table, labels, field in ((L.DIETARY, trip.dietary, "DIETARY"),
                                     (L.OCCASIONS, [trip.occasion] if trip.occasion else [], "OCCASION"),
                                     (L.SPECIAL, trip.special, "SPECIAL_REQUEST")):
            for label in labels:
                if label not in table:
                    continue
                options = self.fitting(table[label]["sent"])
                if options:  # with no frame that fits, the detail is simply missing from this enquiry
                    out.append(self.label_sentence(self.pick(options), field, label))
        return out

    def city(self, name: str) -> str:
        return getattr(self.L, "CITIES", {}).get(name, name)

    def email(self, trip: Trip, register: str) -> Fragment:
        L, r = self.L, self.r
        slots = self._slots(trip)
        remaining = {f for f in ("dest", "party", "dur", "date", "budget") if f in slots}
        intro = self._intro(slots, remaining)
        body = self._field_sentences(trip, slots, remaining)
        r.shuffle(body)
        if trip.style == "long":
            for _ in range(r.randint(1, 3)):
                d = self.distractor(trip)
                if d:
                    body.insert(r.randint(0, len(body)), d)
            fillers = self.fitting(L.FILLERS)
            for filler in r.sample(fillers, min(len(fillers), r.randint(0, 2))):
                body.insert(r.randint(0, len(body)), [T(filler)])
        elif r.random() < .3:
            d = self.distractor(trip)
            if d:
                body.insert(r.randint(0, len(body)), d)
        lead: list[Fragment] = []
        openers = self.fitting(L.OPENERS)
        if trip.style == "long" and r.random() < .75 and openers:
            lead += [[T(s)] for s in r.sample(openers, min(len(openers), r.randint(1, 2)))]
        if intro:
            lead.append(intro)
        if not lead and not body:
            lead = [[T(self.pick(L.VAGUE))]]

        def paragraph(sentences: list[Fragment]) -> Fragment:
            return join(sentences, " ", " ")

        blocks: list[Fragment] = []
        if trip.style == "long":
            salutation = self.pick(L.SALUTATIONS[register])
            if salutation:
                blocks.append([T(salutation)])
            if len(body) >= 4:
                cut = r.randint(2, len(body) - 2)
                blocks += [paragraph(lead + body[:cut]), paragraph(body[cut:])]
            else:
                blocks.append(paragraph(lead + body))
            closing = self.pick(self.fitting(L.CLOSERS))
            sign_off = self.pick(L.SIGN_OFFS[register])
            blocks.append([T(closing)])
            tail = [p for p in (sign_off, self.pick(L.NAMES)) if p]
            if r.random() < .25:
                tail.append(self.pick(L.EXTRA_LINES))
            blocks.append([T("\n".join(tail))])
            return join(blocks, "\n\n", "\n\n")
        text = paragraph(lead + body)
        if r.random() < .5:
            text += [T(" " + self.pick(self.fitting(L.CLOSERS)))]
        if r.random() < .3:
            words = [p for p in (self.pick(L.SIGN_OFFS[register]), self.pick(L.NAMES).split()[0]) if p]
            text += [T(" " + " ".join(words))]
        return text

    def note(self, trip: Trip) -> Fragment:
        L, r = self.L, self.r
        pieces: list[Fragment] = []
        if trip.places:
            pieces.append(join([self.place(p, "note") for p in trip.places], " + ", " + "))
        if trip.duration:
            pieces.append(self.duration(*trip.duration))
        if trip.date:
            pieces.append(self.date_note(trip.date))
        if trip.total:
            party = self.party(trip, "note")
            if party:
                if trip.ages and r.random() < .6:
                    party = party + [T(" ("), S("AGE", self.ages_text(trip.ages), trip.ages), T(")")]
                pieces.append(party)
        if trip.budget:
            word = self.pick(L.NOTE_LABELS["budget"])
            pieces.append(([T(word + " ")] if word else []) + self.budget(trip))
        tables = ((L.ACCOMMODATION, trip.accommodation, "ACCOMMODATION"), (L.FLIGHTS, trip.flights, "FLIGHT"),
                  (L.INTERESTS, trip.interests, "INTEREST"), (L.DIETARY, trip.dietary, "DIETARY"),
                  (L.OCCASIONS, [trip.occasion] if trip.occasion else [], "OCCASION"),
                  (L.SPECIAL, trip.special, "SPECIAL_REQUEST"))
        for table, labels, field in tables:
            for label in labels:
                if label in table:
                    pieces.append([S(field, self.pick(table[label]["note"]), label)])
        if trip.no_flights:
            pieces.append([S("FLIGHT", self.pick(L.NO_FLIGHTS["note"]), "flights not needed")])
        if trip.origin and not trip.no_flights:
            pieces.append([T(self.pick(L.NOTE_ORIGIN)), S("ORIGIN", self.city(trip.origin), trip.origin)])
        if len(pieces) > 2:
            head, rest = pieces[:1], pieces[1:]
            r.shuffle(rest)
            pieces = head + rest
        if not pieces:
            return [T(self.pick(L.VAGUE))]
        sep = self.pick(NOTE_SEPARATORS)
        text = join(pieces, sep, sep)
        opener = self.pick(L.NOTE_OPENERS)
        closer = self.pick(L.NOTE_CLOSERS)
        return ([T(opener + " ")] if opener else []) + text + ([T(sep + closer)] if closer else [])

    def write(self, trip: Trip) -> tuple[Fragment, dict]:
        r = self.r
        self.solo = trip.persona == "solo"
        self.has_partner = not self.solo and not (trip.persona == "family" and trip.adults == 1)
        self.partner = self.pick(self.L.PROPOSAL_PARTNERS if trip.occasion == "proposal" else self.L.PARTNERS)
        names, weights = REGISTERS[trip.style]
        register = r.choices(names, weights=weights)[0]
        self.words_for_numbers = r.random() < .45
        parts = tidy(self.note(trip) if trip.style == "short" else self.email(trip, register))

        meta = {"style": trip.style, "register": register, "persona": trip.persona, "group": trip.group}
        casual = register == "casual"
        rate = 0.0
        if r.random() < (.5 if casual else .2):
            rate = r.uniform(.02, .07 if casual else .04)
        if rate:
            parts = add_typos(parts, rate, r)
        meta["typo_rate"] = round(rate, 3)
        meta["lower_case"] = casual and r.random() < .35
        if meta["lower_case"]:
            parts = map_text(parts, str.lower)
        if casual and self.L.CODE == "fr" and r.random() < .35:
            parts = map_text(parts, strip_accents)
            meta["accents_dropped"] = True
        if casual and self.L.CODE == "de" and r.random() < .2:
            parts = map_text(parts, german_ascii)
            meta["accents_dropped"] = True
        meta["templates"] = sorted(set(self.used))
        return parts, meta
