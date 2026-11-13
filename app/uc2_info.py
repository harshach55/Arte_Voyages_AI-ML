"""Info icons for the demo pages: the wording, and the plain CSS and JS behind the popups.

Both pages use this file, so the wording stays the same on each. A popup shows on hover,
keyboard focus and tap. It needs no library and inserts its text with textContent.
"""

INFO = {
    "mode": "Rules only uses hand-written rules. Rules + model adds a small pretrained model that fills gaps, "
            "only when it is very confident (0.88 or more). Rules + tagger adds a tagger we trained on public "
            "data to fill gaps. Rules + new tagger uses a tagger also trained on synthetic enquiries, so it knows "
            "more fields.",
    "mode_highlight": "This page uses hand-written rules, and a tagger we trained on public data fills gaps the "
                      "rules leave.",
    "comparison": "Old tagger: trained on public data only. New tagger: also trained on synthetic enquiries. "
                  "All these scores are optimistic, because the sets were seen before or come from the same "
                  "generator as the training data.",
    "export": "Download the results shown here as a JSON or CSV file, saved on this computer only.",
    "several": "Paste many enquiries, one per line, and get one results table.",
    "source": "The exact words the value was taken from, so a person can check it.",
    "tag_rule": "Found by hand-written rules.",
    "tag_model": "Found by the pretrained model. The number is how similar the text is to the label, from 0 to 1.",
    "tag_crf": "Found by the trained tagger. The number is its confidence from 0 to 1. Values below 0.5 are dropped.",
    "flags": "Values that are missing, unclear or assumed. Nothing is guessed, and a person should confirm them.",
    "unclear_dates": "A date like 3/4 can mean 3 April or 4 March, so both readings are kept and flagged.",
    "budget_band": "Band limits are placeholders until the team confirms the real ranges.",
    "language": "Detected by counting common words. Very short texts can fail.",
    "save": "Saves your fix on this computer as possible training data, after someone checks it. "
            "Do not paste real client enquiries.",
    "real_number": "Scored once on 45 enquiries written and locked before scoring. This is the most trustworthy number.",
    "optimistic": "Rules or the model were changed after seeing errors on these same enquiries, so these scores "
                  "are higher than on new text.",
    "pipeline": "The stages an enquiry goes through, and which parts are trained, rules or still pending.",
    "highlight": "Each coloured part is text the extractor used for a value. The colour shows which field it belongs to.",
}

INFO_CSS = """
  button.info { display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px;
                padding: 0; margin: 0 2px 0 8px; border: 2px solid #1f4e8c; border-radius: 50%; background: #fff;
                color: #1f4e8c; font: italic bold 17px Georgia, serif; line-height: 1; cursor: help;
                vertical-align: middle; }
  button.info:hover, button.info:focus { background: #1f4e8c; color: #fff; outline: 3px solid #f59e0b; }
  #info-pop { position: absolute; z-index: 1000; box-sizing: border-box; max-width: min(480px, calc(100vw - 20px));
              padding: 12px 16px; border-radius: 6px; background: #111; color: #fff;
              font: normal 21px/1.4 Arial, Helvetica, sans-serif; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35); }
  #info-pop[hidden] { display: none; }
"""

INFO_JS = """
const INFO = __INFO__;
function infoButton(key) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "info";
  button.textContent = "i";
  button.dataset.info = key;
  button.setAttribute("aria-label", "More information: " + INFO[key]);
  return button;
}
function infoKeys(label, text) {
  const keys = [];
  if (label === "Language") keys.push("language");
  if (text.includes("is unclear")) keys.push("unclear_dates");
  if (text.includes("(placeholder)")) keys.push("budget_band");
  return keys;
}
for (const slot of document.querySelectorAll("[data-info-slot]")) slot.replaceWith(infoButton(slot.dataset.infoSlot));

const infoPop = document.createElement("div");
infoPop.id = "info-pop";
infoPop.setAttribute("role", "tooltip");
infoPop.hidden = true;
document.body.appendChild(infoPop);
let infoPinned = null;  // the icon that was tapped or clicked, so its popup stays open

function infoTarget(event) { return event.target.closest ? event.target.closest("button.info") : null; }
function showInfo(button) {
  infoPop.textContent = INFO[button.dataset.info];
  infoPop.hidden = false;
  const box = button.getBoundingClientRect();
  const left = Math.max(10, Math.min(box.left, document.documentElement.clientWidth - infoPop.offsetWidth - 10));
  const below = box.bottom + 8;
  const above = box.top - infoPop.offsetHeight - 8;
  const top = below + infoPop.offsetHeight > window.innerHeight && above > 0 ? above : below;
  infoPop.style.left = (left + window.scrollX) + "px";
  infoPop.style.top = (top + window.scrollY) + "px";
}
function hideInfo() { infoPop.hidden = true; infoPinned = null; }
document.addEventListener("mouseover", (e) => { const b = infoTarget(e); if (b) showInfo(b); });
document.addEventListener("mouseout", (e) => {
  const b = infoTarget(e);
  if (b && infoPinned !== b && document.activeElement !== b) infoPop.hidden = true;
});
document.addEventListener("focusin", (e) => { const b = infoTarget(e); if (b) showInfo(b); });
document.addEventListener("focusout", (e) => { if (infoTarget(e)) hideInfo(); });
document.addEventListener("click", (e) => {
  const b = infoTarget(e);
  if (!b) { hideInfo(); return; }
  if (infoPinned === b) { hideInfo(); return; }
  infoPinned = b;
  showInfo(b);
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideInfo(); });
"""


def info_script(to_json) -> str:
    return INFO_JS.replace("__INFO__", to_json(INFO))
