"""Layer 1 — DOM/provider detection + Layer 4 structural flags (one evaluate).

Single batched ``page.evaluate`` keeps normal-page cost to one round-trip.
Read-only: querySelector / getBoundingClientRect / getComputedStyle only.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Provider-specific selectors. data-sitekey is config evidence, not blocking.
RECAPTCHA_SELECTORS = [
    ".g-recaptcha",
    ".grecaptcha-badge",
    "iframe[src*='recaptcha']",
    "iframe[src*='google.com/recaptcha']",
    'iframe[title*="reCAPTCHA"]',
    "[data-sitekey]",
]

HCAPTCHA_SELECTORS = [
    ".h-captcha",
    "iframe[src*='hcaptcha']",
    'iframe[title*="hCaptcha"]',
    "[data-hcaptcha-response]",
    ".hcaptcha-container",
]

TURNSTILE_SELECTORS = [
    ".cf-turnstile",
    "iframe[src*='challenges.cloudflare.com']",
    "iframe[src*='turnstile']",
    "[data-cf-turnstile]",
    "[data-turnstile-sitekey]",
]

CLOUDFLARE_SELECTORS = [
    "#cf-challenge-stage",
    "#challenge-stage",
    "[data-ray-id]",
]

# Generic attribute substring scan (guarded; see _GENERIC_DENY).
GENERIC_SUBSTRINGS = [
    "captcha",
    "recaptcha",
    "hcaptcha",
    "turnstile",
    "human-verification",
    "verify-human",
    "security-check",
]

# Class names containing these words are NOT captcha by themselves.
_GENERIC_DENY = (
    "coding-challenge",
    "code-challenge",
    "challenge-article",
    "challenge-list",
)

DOM_JS = """() => {
  const out = { matched: [], generic: [], structural: {}, sitekey_present: false };
  const isVisible = (el) => {
    try {
      const r = el.getBoundingClientRect();
      if (!r || r.width <= 0 || r.height <= 0) return {visible: false, rect: null};
      const cs = window.getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity || '1') <= 0) {
        return {visible: false, rect: {x: r.x, y: r.y, width: r.width, height: r.height}};
      }
      const inViewport = (r.bottom > 0 && r.right > 0 &&
        r.top < window.innerHeight && r.left < window.innerWidth);
      return {visible: inViewport, rect: {x: r.x, y: r.y, width: r.width, height: r.height}};
    } catch (e) { return {visible: false, rect: null}; }
  };
  const SELS = __SELS__;
  for (const sel of SELS) {
    try {
      const els = document.querySelectorAll(sel);
      els.forEach((el) => {
        const v = isVisible(el);
        // Record iframe src: CSS substring selectors (e.g. src*='recaptcha')
        // also match tokens inside query params such as "recaptchacompat";
        // the interpreter confirms provider attribution against the URL host.
        let src = '';
        try {
          if ((el.tagName || '').toLowerCase() === 'iframe') {
            src = (el.src || el.getAttribute('src') || '').slice(0, 500);
          }
        } catch (e) {}
        out.matched.push({selector: sel, visible: v.visible, rect: v.rect,
          tag: (el.tagName || '').toLowerCase(), src: src});
      });
    } catch (e) {}
  }
  try {
    out.sitekey_present = !!document.querySelector('[data-sitekey]');
  } catch (e) {}
  // Generic attribute scan, capped.
  try {
    const els = document.querySelectorAll('[class],[id],[name]');
    let n = 0;
    for (const el of els) {
      if (n > 40) break;
      const blob = ((el.className && el.className.baseVal !== undefined ? '' : (el.className || '')) + ' ' +
        (el.id || '') + ' ' + (el.getAttribute('name') || '')).toLowerCase();
      if (!blob.trim()) continue;
      const low = ' ' + blob + ' ';
      const deny = ['coding-challenge','code-challenge','challenge-article','challenge-list'];
      if (deny.some(d => blob.includes(d))) continue;
      const hits = [];
      for (const s of __GENERIC__) { if (blob.includes(s)) hits.push(s); }
      // bare word "challenge" alone is NOT enough (too noisy) -> skip
      if (hits.length && !(hits.length === 1 && hits[0] === 'challenge')) {
        const v = isVisible(el);
        out.generic.push({attr: blob.slice(0, 160), hits: hits, visible: v.visible});
        n++;
      }
    }
  } catch (e) {}
  // Structural flags.
  try {
    const bodyText = (document.body ? document.body.innerText : '') || '';
    const hasRobot = /i['\\u2019']?m not a robot|i am not a robot/i.test(bodyText);
    const hasVerifyBtn = Array.from(document.querySelectorAll('button,input[type="submit"],input[type="button"]'))
      .some(b => /verify|i am human|i'm human|confirm you are human/i.test((b.innerText || b.value || '')));
    const overlays = Array.from(document.querySelectorAll(
      '[role="dialog"],.modal,.overlay,.challenge,.captcha-modal,#captcha,.cf-challenge'))
      .map(el => isVisible(el)).filter(v => v.visible).length;
    const imgs = document.querySelectorAll('.captcha-images img, .image-select img, .challenge-images img, [class*="captcha"] img');
    let gridImgs = imgs.length;
    if (!gridImgs) {
      // image-grid heuristic: >=6 sibling images in a tight container mentioning select/click
      const divs = document.querySelectorAll('div');
      for (const d of divs) {
        try {
          const im = d.querySelectorAll('img');
          if (im.length >= 6 && /select|click|choose/i.test((d.innerText || '').slice(0, 300))) {
            gridImgs = im.length; break;
          }
        } catch (e) {}
      }
    }
    const disabledForm = !!document.querySelector('form fieldset[disabled], form [disabled]');
    out.structural = { has_robot_checkbox_text: !!hasRobot, has_verify_button: !!hasVerifyBtn,
      visible_overlays: overlays, grid_images: gridImgs, disabled_form: !!disabledForm };
  } catch (e) { out.structural = {}; }
  return out;
}"""

_ALL_SELECTORS = (
    RECAPTCHA_SELECTORS + HCAPTCHA_SELECTORS + TURNSTILE_SELECTORS + CLOUDFLARE_SELECTORS
)


def build_js() -> str:
    import json

    body = (
        DOM_JS.replace("__SELS__", json.dumps(_ALL_SELECTORS))
        .replace("__GENERIC__", json.dumps(GENERIC_SUBSTRINGS))
        .replace("'challenge'", "'__never__'")
    )
    # DOM_JS is an arrow function "() => {...}"; invoke it so evaluate
    # returns the payload dict instead of the function object.
    return f"({body})()"


def _selector_family(selector: str) -> str:
    s = selector.lower()
    if "recaptcha" in s or "g-recaptcha" in s or "grecaptcha" in s or "data-sitekey" in s:
        return "recaptcha"
    if "hcaptcha" in s or "h-captcha" in s:
        return "hcaptcha"
    if "turnstile" in s or "challenges.cloudflare.com" in s or "cf-turnstile" in s:
        return "turnstile"
    if "cf-challenge" in s or "challenge-stage" in s or "data-ray-id" in s:
        return "cloudflare"
    return "generic"


def _url_host(url: str) -> str:
    """Lowercased URL host, '' when unparseable (never raises)."""
    try:
        from urllib.parse import urlparse

        return (urlparse(url or "").hostname or "").lower()
    except Exception:
        return ""


# Host tokens corroborating each provider family for iframe-src matches.
_FAMILY_HOST_TOKENS = {
    "recaptcha": ("recaptcha", "google"),
    "hcaptcha": ("hcaptcha",),
    "turnstile": ("turnstile", "cloudflare"),
    "cloudflare": ("cloudflare",),
}


def _iframe_src_corroborates(selector: str, src: str) -> bool:
    """True unless an iframe[src*=...] match is contradicted by its URL host.

    CSS substring matching cannot distinguish a provider host from a query
    token such as hCaptcha's ``recaptchacompat=true``. Attribution therefore
    requires the expected token in the iframe's host. Empty/unparseable src
    keeps the match (fail-open: a src-less iframe match is still evidence).
    Non-iframe-src selectors are unaffected.
    """
    s = (selector or "").lower()
    if not s.startswith("iframe[src*="):
        return True
    family = _selector_family(selector)
    tokens = _FAMILY_HOST_TOKENS.get(family)
    if not tokens:
        return True
    host = _url_host(src)
    if not host:
        return True
    return any(t in host for t in tokens)


def run_dom_evaluate(page: Any) -> Dict[str, Any]:
    """Execute the batched JS. Returns {} on any failure (fail-closed)."""
    try:
        return dict(page.evaluate(build_js()) or {})
    except Exception:
        return {}


def interpret_dom_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pure function: payload -> signals/evidence/provider hints (testable w/o browser)."""
    matched: List[Dict[str, Any]] = list((payload or {}).get("matched") or [])
    generic: List[Dict[str, Any]] = list((payload or {}).get("generic") or [])
    structural: Dict[str, Any] = dict((payload or {}).get("structural") or {})
    sitekey = bool((payload or {}).get("sitekey_present"))

    signals: List[str] = []
    matching_selectors: List[str] = []
    attributed_selectors: List[str] = []  # host-corroborated (drives families)
    families = set()
    visible_widget = False
    for m in matched:
        sel = str(m.get("selector", ""))
        if sel not in matching_selectors:
            matching_selectors.append(sel)
        if _iframe_src_corroborates(sel, str(m.get("src", "") or "")):
            if sel not in attributed_selectors:
                attributed_selectors.append(sel)
            families.add(_selector_family(sel))
        if m.get("visible"):
            visible_widget = True

    # Per-family real-widget matches ([data-sitekey] alone is weak: many
    # providers document it, and hCaptcha/Turnstile fixtures reuse the attr).
    recaptcha_real = any(
        _selector_family(s) == "recaptcha" and s != "[data-sitekey]"
        for s in attributed_selectors
    )
    if recaptcha_real:
        signals.append("recaptcha_dom")
        signals.append("g_recaptcha_element")
    if "hcaptcha" in families:
        signals.append("hcaptcha_dom")
    if "turnstile" in families:
        signals.append("turnstile_dom")
    if "cloudflare" in families:
        signals.append("cloudflare_challenge_dom")
    if sitekey and "data_sitekey_present" not in signals:
        signals.append("data_sitekey_present")  # weak/config evidence

    generic_visible = any(g.get("visible") for g in generic)
    if generic:
        signals.append("generic_captcha_attribute")
    if generic_visible:
        signals.append("generic_captcha_attribute_visible")

    # Structural signals (evidence, not verdicts).
    if structural.get("has_robot_checkbox_text"):
        signals.append("checkbox_like_challenge")
    if structural.get("has_verify_button"):
        signals.append("verification_button")
    if int(structural.get("visible_overlays") or 0) > 0:
        signals.append("challenge_overlay")
    if int(structural.get("grid_images") or 0) >= 4:
        signals.append("image_grid_selection")
    if structural.get("disabled_form"):
        signals.append("disabled_form_behind_verification")

    signals = [s for s in signals if s]
    return {
        "signals": sorted(set(signals)),
        "matching_selectors": matching_selectors,
        "generic": generic,
        "structural": structural,
        "families": sorted(families),
        "visible_widget": visible_widget,
        "generic_visible": generic_visible,
        "sitekey_present": sitekey,
    }
