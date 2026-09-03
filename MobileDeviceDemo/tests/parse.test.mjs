/* Ports the assertions from
 *   tests/test_tags.py  and  tests/test_parse.py
 * to keep js/wjm-parse.js in lock-step with the Python grammar.
 *
 *   node tests/parse.test.mjs
 */
import { createRequire } from "module";
import assert from "assert";
const require = createRequire(import.meta.url);
global.self = global;
const P = require("../js/wjm-parse.js");

let pass = 0;
const t = (name, fn) => {
  try { fn(); pass++; console.log("ok  " + name); }
  catch (e) { console.error("FAIL " + name + "\n     " + e.message); process.exitCode = 1; }
};
const line = (text, y = 0) => ({ text, bbox: [10, y, 300, 20], confidence: 0.9 });

// ---- tags.py ------------------------------------------------------------
t("parse_tag single", () => {
  assert.equal(P.parseTag("#auth"), "auth");
  assert.equal(P.parseTag("#[Data Science]"), "Data Science");
  assert.equal(P.parseTag("  #P017 "), "P017");
  assert.equal(P.parseTag("plain"), null);
  assert.equal(P.parseTag("#"), null);
});

t("parse_tags in line", () => {
  assert.deepEqual(P.parseTags("#AI #[Data Science] and #python"), ["AI", "Data Science", "python"]);
  assert.deepEqual(P.parseTags("no tags here"), []);
  assert.deepEqual(P.parseTags("#a #a"), ["a", "a"]);
});

t("parse_metadata_cells (spec example)", () => {
  const md = P.parseMetadataCells(
    ["#Research", "#P017", "#AI #[Data Science]"],
    ["#P016", "", "#P027", "#P018"],
  );
  assert.deepEqual(md, {
    document_id: "Research", page_id: "P017", topic_tags: ["AI", "Data Science"],
    left: "P016", above: null, below: "P027", right: "P018",
  });
});

t("parse_metadata_cells all blank", () => {
  assert.deepEqual(P.parseMetadataCells([], []), {
    document_id: null, page_id: null, topic_tags: [],
    left: null, above: null, below: null, right: null,
  });
});

t("split_qualified", () => {
  assert.deepEqual(P.splitQualified("Research:P017:AUTH"), ["Research", "P017", "AUTH"]);
  assert.deepEqual(P.splitQualified("P017:AUTH"), [null, "P017", "AUTH"]);
  assert.deepEqual(P.splitQualified("AUTH"), [null, null, "AUTH"]);
  assert.deepEqual(P.splitQualified("Research : P017 : #[Auth Service]"), ["Research", "P017", "Auth Service"]);
});

t("parse_reference forms", () => {
  assert.equal(P.parseReference("-> [#AUTH]").anchor, "AUTH");
  assert.equal(P.parseReference("REF: #AUTH").anchor, "AUTH");
  const r = P.parseReference("see -> [Research:P017:AUTH] for details");
  assert.deepEqual([r.document, r.page, r.anchor], ["Research", "P017", "AUTH"]);
  assert.equal(P.parseReference("-> [#[Auth Service]]").anchor, "Auth Service");
});

// ---- parse.py ---------------------------------------------------------
t("bullet states", () => {
  const els = P.parseLines([line("* open task"), line("x done"), line("> migrated"),
    line("? research question")]);
  assert.deepEqual(els.map((e) => e.kind), ["bullet", "bullet", "bullet", "bullet"]);
  assert.deepEqual(els.map((e) => e.data.state), ["open", "completed", "migrated", "question"]);
  assert.equal(els[0].data.item, "open task");
});

t("bullet glyph needs a space", () => {
  assert.equal(P.parseLines([line("xylophone practice")])[0].kind, "text");
});

t("tag-only line vs text with tags", () => {
  const els = P.parseLines([line("#AI #[Data Science]"), line("some prose #backend here")]);
  assert.equal(els[0].kind, "tags");
  assert.deepEqual(els[0].data.tags, ["AI", "Data Science"]);
  assert.equal(els[1].kind, "text");
  assert.deepEqual(els[1].data.tags, ["backend"]);
});

t("temporal tags", () => {
  const els = P.parseLines([
    line("[DUE: 2026-09-14]"),
    line("[EVENT: 2026-09-18 14:00]"),
    line("[RANGE: 2026-09-12 -> 2026-09-19]"),
  ]);
  assert.deepEqual(
    els.map((e) => [e.data.type, e.data.start, e.data.end ?? null]),
    [["due", "2026-09-14", null], ["event", "2026-09-18 14:00", null], ["range", "2026-09-12", "2026-09-19"]],
  );
});

t("reference line", () => {
  const els = P.parseLines([line("-> [Research:P017:AUTH]")]);
  assert.equal(els[0].kind, "reference");
  assert.deepEqual([els[0].data.document, els[0].data.page, els[0].data.anchor], ["Research", "P017", "AUTH"]);
});

t("contact block", () => {
  const els = P.parseLines([
    line("+ CONTACT"), line("Jane Smith"), line("jane@example.com  555-123-4567"),
    line("Acme Corp"), line("+---------------+"), line("back to notes"),
  ]);
  const c = els.find((e) => e.kind === "contact");
  assert.equal(c.data.name, "Jane Smith");
  assert.equal(c.data.email, "jane@example.com");
  assert.ok(c.data.phone.includes("555-123-4567"));
  assert.equal(c.data.organization, "Acme Corp");
  assert.equal(els[els.length - 1].kind, "text");
  assert.equal(els[els.length - 1].text, "back to notes");
});

t("bullet glyph vocabulary", () => {
  const states = new Set(Object.values(P.BULLET_STATES));
  for (const s of ["open", "completed", "migrated", "scheduled", "note", "event", "important", "question"]) {
    assert.ok(states.has(s), "missing state " + s);
  }
});

console.log(`\n${pass} passed`);
