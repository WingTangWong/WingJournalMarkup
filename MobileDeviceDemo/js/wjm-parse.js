/* WJM hand-markup grammar + element parser — a direct port of
 *   src/wingjournal/recognition/tags.py
 *   src/wingjournal/recognition/parse.py
 *
 * Pure text: no OpenCV, no DOM. Runs in the worker and in Node (tests).
 * Keep this in lock-step with the Python originals — CLI first, then port.
 */
(function (root) {
  "use strict";

  // ---- tags.py ---------------------------------------------------------

  //  #term   or   #[term with spaces]
  const TAG_SRC = "#(?:\\[(?<bracketed>[^\\]\\n]+)\\]|(?<word>[^\\s#\\[\\]]+))";
  const TAG_RE = new RegExp(TAG_SRC, "g");
  const TAG_FULL_RE = new RegExp("^(?:" + TAG_SRC + ")$");

  const REF_BRACKET_RE = /(?:->|REF:)\s*\[\s*(?<body>[^\]\n]+?)\s*\]/i;
  const REF_BARE_RE = /(?:->|REF:)\s*(?<body>[^\]\n]+?)\s*$/i;

  const canonical = (name) => String(name == null ? "" : name).replace(/\s+/g, " ").trim();

  function parseTag(token) {
    const m = TAG_FULL_RE.exec(String(token == null ? "" : token).trim());
    if (!m) return null;
    return canonical(m.groups.bracketed || m.groups.word);
  }

  function parseTags(text) {
    const out = [];
    const re = new RegExp(TAG_SRC, "g");
    let m;
    while ((m = re.exec(text || ""))) out.push(canonical(m.groups.bracketed || m.groups.word));
    return out;
  }

  const firstTag = (text) => {
    const t = parseTags(text);
    return t.length ? t[0] : null;
  };

  function parseMetadataCells(row1, row2) {
    const r1 = row1.concat([null, null, null]).slice(0, 3);
    const r2 = row2.concat([null, null, null, null]).slice(0, 4);
    return {
      document_id: firstTag(r1[0] || ""),
      page_id: firstTag(r1[1] || ""),
      topic_tags: parseTags(r1[2] || ""),
      left: firstTag(r2[0] || ""),
      above: firstTag(r2[1] || ""),
      below: firstTag(r2[2] || ""),
      right: firstTag(r2[3] || ""),
    };
  }

  // "document : page : anchor" — fewer components fill from the right
  function splitQualified(name) {
    let parts = name.split(":").map(canonical);
    parts = parts.filter((p) => p !== "").map((p) => parseTag(p) || p);
    if (!parts.length) return [null, null, ""];
    const anchor = parts[parts.length - 1];
    const page = parts.length >= 2 ? parts[parts.length - 2] : null;
    const document = parts.length >= 3 ? parts[parts.length - 3] : null;
    return [document, page, anchor];
  }

  function parseReference(text) {
    const m = REF_BRACKET_RE.exec(text || "") || REF_BARE_RE.exec(text || "");
    if (!m) return null;
    let body = m.groups.body.trim().replace(/^\[+|\]+$/g, "").trim();
    let document = null;
    let page = null;
    let anchor;
    if (body.includes(":")) {
      [document, page, anchor] = splitQualified(body);
    } else {
      anchor = parseTag(body) || canonical(body.replace(/^[#[]+/, ""));
    }
    if (!anchor) return null;
    return { anchor, document, page, raw: (text || "").trim() };
  }

  // ---- parse.py -------------------------------------------------------

  // leading glyph -> bullet-journal state (spec §18)
  const BULLET_STATES = {
    "•": "open", "*": "open", "·": "open",
    x: "completed", "×": "completed", X: "completed",
    ">": "migrated", "<": "scheduled",
    "-": "note", "–": "note", "—": "note",
    o: "event", O: "event", "○": "event", "◦": "event",
    "!": "important", "?": "question",
  };

  const TEMPORAL_RE = /\[\s*(DUE|EVENT|RANGE)\s*:\s*(.+?)\s*(?:->\s*(.+?)\s*)?\]/gi;
  const CONTACT_START_RE = /^\+?\s*CONTACT\b/i;
  const CONTACT_END_RE = /^\+[-\s]*\+?\s*$/;
  const EMAIL_RE = /[\w.+-]+@[\w-]+\.[\w.-]+/;
  const PHONE_RE = /\+?\d[\d\-.\s()]{6,}\d/;

  const titleCase = (s) =>
    s.replace(/[A-Za-z]+/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase());

  const mkElement = (kind, text, bbox, confidence = 0, data = {}) => ({
    kind, text, bbox, confidence, data,
  });

  function temporalElements(line) {
    const out = [];
    const re = new RegExp(TEMPORAL_RE.source, "gi");
    let m;
    while ((m = re.exec(line.text))) {
      const data = { type: m[1].toLowerCase(), start: m[2].trim(), match: m[0] };
      if (m[3]) data.end = m[3].trim();
      out.push(mkElement("temporal", line.text, line.bbox, line.confidence, data));
    }
    return out;
  }

  function bulletElement(line) {
    const s = line.text.replace(/^\s+/, "");
    if (s.length < 2 || !/\s/.test(s[1])) return null;
    const state = BULLET_STATES[s[0]];
    if (state === undefined) return null;
    const body = s.slice(2).trim();
    return mkElement("bullet", line.text, line.bbox, line.confidence, {
      state, glyph: s[0], item: body, tags: parseTags(body),
    });
  }

  function finishContact(lines) {
    const joined = lines.map((l) => l.text.trim()).filter(Boolean).join("\n");
    const email = EMAIL_RE.exec(joined);
    const phone = PHONE_RE.exec(joined);
    let remaining = joined;
    for (const hit of [email, phone]) if (hit) remaining = remaining.replace(hit[0], "");
    const parts = remaining.split("\n").map((p) => p.trim()).filter(Boolean);
    const data = {
      name: parts[0] || null,
      email: email ? email[0] : null,
      phone: phone ? phone[0].trim() : null,
      organization: parts.length > 1 ? parts[1] : null,
    };
    const bbox = lines.length ? lines[0].bbox : [0, 0, 0, 0];
    const conf = lines.length
      ? Math.round((lines.reduce((s, l) => s + l.confidence, 0) / lines.length) * 1000) / 1000
      : 0;
    return mkElement("contact", joined, bbox, conf, data);
  }

  /** @param {{text:string,bbox:number[],confidence:number}[]} lines */
  function parseLines(lines) {
    const elements = [];
    let contact = null;

    for (const line of lines) {
      const text = line.text.trim();

      if (contact !== null) {
        if (CONTACT_END_RE.test(text) || !text) {
          elements.push(finishContact(contact));
          contact = null;
          continue;
        }
        contact.push(line);
        continue;
      }
      if (!text) continue;

      if (CONTACT_START_RE.test(text)) {
        contact = [];
        continue;
      }

      const temporal = temporalElements(line);
      elements.push(...temporal);

      const ref = parseReference(text);
      if (ref !== null) {
        elements.push(mkElement("reference", text, line.bbox, line.confidence, {
          anchor: ref.anchor, document: ref.document, page: ref.page,
        }));
        continue;
      }

      const bullet = bulletElement(line);
      if (bullet !== null) {
        elements.push(bullet);
        continue;
      }

      const tags = parseTags(text);
      const stripped = text.replace(/#(?:\[[^\]]+\]|\S+)/g, "").trim();
      if (tags.length && !stripped) {
        elements.push(mkElement("tags", text, line.bbox, line.confidence, { tags }));
        continue;
      }

      if (temporal.length) continue;

      const kind = text.length <= 40 && text === titleCase(text) ? "heading" : "text";
      elements.push(mkElement(kind, text, line.bbox, line.confidence, tags.length ? { tags } : {}));
    }

    if (contact !== null) elements.push(finishContact(contact));
    return elements;
  }

  root.WJMParse = {
    canonical, parseTag, parseTags, firstTag, parseMetadataCells,
    splitQualified, parseReference,
    BULLET_STATES, parseLines, titleCase,
  };
})(typeof self !== "undefined" ? self : globalThis);

if (typeof module !== "undefined" && module.exports) module.exports = globalThis.WJMParse;
