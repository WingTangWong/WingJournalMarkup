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
          "curve-style": "bezier", label: "data(label)", "font-size": 8,
          color: "#889", "text-rotation": "autorotate",
          "text-background-color": "#fff", "text-background-opacity": 0.85,
          "text-background-padding": 1 } },
      { selector: 'edge[kind = "inferred"]', style: { "line-color": "#c8c8c8",
          "target-arrow-color": "#c8c8c8", "line-style": "dashed" } },
      { selector: 'edge[kind = "document"]', style: { "line-color": "#7840a0",
          "line-style": "dotted", "target-arrow-shape": "none", width: 1.5,
          label: "", opacity: 0.6 } },
      { selector: ".sel", style: { "border-width": 3, "border-color": "#f0a000" } },
    ],
    layout: coseLayout(),
  });

  function coseLayout() {
    return {
      name: "cose", animate: false, padding: 40,
      idealEdgeLength: 150, nodeRepulsion: 12000, nodeOverlap: 24,
      componentSpacing: 120, gravity: 0.3,
    };
  }

  function spatialLayout() {
    const minX = Math.min(0, ...Object.values(data.layout).map((p) => p[0]));
    let orphanRow = 0;
    return {
      name: "preset", fit: true, padding: 50,
      positions: (n) => {
        const p = data.layout[n.id()];
        if (p) return { x: p[0] * 150, y: p[1] * 150 };
        return { x: (minX - 2) * 150, y: (orphanRow++) * 150 }; // orphans in a side column
      },
    };
  }

  // spatial layout by default when the relationships give us positions
  if (Object.keys(data.layout).length >= 2) {
    cy.layout(spatialLayout()).run();
  }

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
  spatial.checked = Object.keys(data.layout).length >= 2;
  spatial.addEventListener("change", () => {
    cy.layout(spatial.checked ? spatialLayout() : coseLayout()).run();
  });
})();
