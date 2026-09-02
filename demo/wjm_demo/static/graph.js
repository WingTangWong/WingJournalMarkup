(async function () {
  const data = await (await fetch("/api/graph")).json();

  const nodes = data.nodes.map((n) => ({
    data: { id: n.id, label: n.label, doc: n.document || "", orphan: n.orphan,
            captures: n.captures, topics: (n.topics || []).join(", ") },
  }));

  const edges = data.edges.map((e, i) => ({
    data: { id: "e" + i, source: e.source, target: e.target,
            label: e.relation, kind: e.kind },
  }));

  // dashed "same document" hulls -> add invisible grouping edges
  Object.entries(data.documents || {}).forEach(([doc, members]) => {
    for (let i = 1; i < members.length; i++) {
      edges.push({ data: { id: `doc-${doc}-${i}`, source: members[0],
                           target: members[i], kind: "document", label: doc } });
    }
  });

  const cy = cytoscape({
    container: document.getElementById("cy"),
    elements: { nodes, edges },
    style: [
      { selector: "node", style: {
          "background-color": "#28468c", label: "data(label)", color: "#fff",
          "text-valign": "center", "font-size": 11, width: 42, height: 42,
          "text-outline-color": "#28468c", "text-outline-width": 2 } },
      { selector: 'node[?orphan]', style: { "background-color": "#c82828",
          "text-outline-color": "#c82828" } },
      { selector: "edge", style: { width: 2, "line-color": "#148c3c",
          "target-arrow-color": "#148c3c", "target-arrow-shape": "triangle",
          "curve-style": "bezier", label: "data(label)", "font-size": 9,
          color: "#557", "text-background-color": "#fff",
          "text-background-opacity": 1 } },
      { selector: 'edge[kind = "inferred"]', style: { "line-color": "#c8c8c8",
          "target-arrow-color": "#c8c8c8", "line-style": "dashed" } },
      { selector: 'edge[kind = "document"]', style: { "line-color": "#7840a0",
          "line-style": "dotted", "target-arrow-shape": "none", width: 1.5,
          label: "", opacity: 0.6 } },
      { selector: ".sel", style: { "border-width": 3, "border-color": "#f0a000" } },
    ],
    layout: { name: "cose", animate: false, padding: 30 },
  });

  const detail = document.getElementById("detail");
  cy.on("tap", "node", (evt) => {
    cy.$(".sel").removeClass("sel");
    evt.target.addClass("sel");
    const d = evt.target.data();
    detail.innerHTML = `<h2>${d.label}</h2>
      <p>document: <b>${d.doc || "—"}</b> &nbsp; captures: ${d.captures}
      &nbsp; topics: ${d.topics || "—"}</p>
      <p><a href="/api/capture/${d.id}">raw capture JSON for latest</a></p>`;
  });

  const spatial = document.getElementById("spatial");
  spatial.addEventListener("change", () => {
    if (spatial.checked && Object.keys(data.layout).length) {
      cy.layout({
        name: "preset", fit: true, padding: 40,
        positions: (n) => {
          const p = data.layout[n.id()];
          return p ? { x: p[0] * 120, y: p[1] * 120 } : { x: 0, y: 0 };
        },
      }).run();
    } else {
      cy.layout({ name: "cose", animate: false, padding: 30 }).run();
    }
  });
})();
