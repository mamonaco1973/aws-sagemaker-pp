#!/usr/bin/env python3
"""Generate the README architecture diagram, one SVG per colour scheme.

Two files rather than one self-switching file: GitHub strips <style> blocks
from inline SVG, so a `prefers-color-scheme` media query inside the document is
silently dropped. The README pairs them with <picture>, which GitHub honours.

A README diagram is studied, not narrated, so it is COMPLETE rather than
minimal: the notebook or CLI, the S3 bucket, the training job, the
serverless endpoint, and the whole MCP path (Claude, the HTTP API and router
Lambda, Cognito and the OAuth state table) are all on it.
Run:  python3 make_diagram.py
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------
# Geometry and style (shared with the other projects' diagrams)
# ------------------------------------------------------------------------------
NW, NH = 420, 110                          # node width, height
GUTTER = 140                               # between columns: room for a label
MARGIN = 40
COL = [MARGIN + i * (NW + GUTTER) for i in range(3)]
ROW = [40 + i * 200 for i in range(3)]
W = MARGIN * 2 + NW * 3 + GUTTER * 2
H = ROW[-1] + NH + 40 + 0
TITLE_PX, SUB_PX, EDGE_PX = 26, 17, 16

THEMES = {
    "dark": {
        "bg": "#0d1117", "panel": "#161b22", "text": "#e6edf3",
        "muted": "#8b949e", "line": "#58a6ff", "local": "#39c5bb",
        "purple": "#c39be8", "navy": "#c8d6e8", "blue": "#58a6ff",
        "amber": "#f2c163", "store": "#7ee787",
    },
    "light": {
        "bg": "#ffffff", "panel": "#f6f8fa", "text": "#1f2328",
        "muted": "#59636e", "line": "#0969da", "local": "#137e77",
        "purple": "#8250df", "navy": "#475467", "blue": "#0969da",
        "amber": "#9a6700", "store": "#1a7f37",
    },
}

# Lucide icon paths, on a 24-unit grid.
ICONS = {
    "globe": ['<circle cx="12" cy="12" r="10"/>',
              '<path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>',
              '<path d="M2 12h20"/>'],
    "route": ['<circle cx="6" cy="19" r="3"/>', '<circle cx="18" cy="5" r="3"/>',
              '<path d="M12 19h4.5a3.5 3.5 0 0 0 0-7h-8a3.5 3.5 0 0 1 0-7H12"/>'],
    "network": ['<rect x="16" y="16" width="6" height="6" rx="1"/>',
                '<rect x="2" y="16" width="6" height="6" rx="1"/>',
                '<rect x="9" y="2" width="6" height="6" rx="1"/>',
                '<path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3"/>',
                '<path d="M12 12V8"/>'],
    "layers": ['<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83'
               'l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/>',
               '<path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/>',
               '<path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>'],
    "database": ['<ellipse cx="12" cy="5" rx="9" ry="3"/>',
                 '<path d="M3 5V19A9 3 0 0 0 21 19V5"/>',
                 '<path d="M3 12A9 3 0 0 0 21 12"/>'],
    "key": ['<path d="M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 '
            '1-1v-1a1 1 0 0 1 1-1h1a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 '
            '1.414-.586l.814-.814a6.5 6.5 0 1 0-4-4z"/>',
            '<circle cx="16.5" cy="7.5" r=".5"/>'],
    "monitor": ['<rect width="20" height="14" x="2" y="3" rx="2"/>',
                '<line x1="8" x2="16" y1="21" y2="21"/>',
                '<line x1="12" x2="12" y1="17" y2="21"/>'],
    "terminal": ['<polyline points="4 17 10 11 4 5"/>',
                 '<line x1="12" x2="20" y1="19" y2="19"/>'],
    "users": ['<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>',
              '<circle cx="9" cy="7" r="4"/>',
              '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/>',
              '<path d="M16 3.13a4 4 0 0 1 0 7.75"/>'],
    "file": ['<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>',
             '<path d="M14 2v4a2 2 0 0 0 2 2h4"/>', '<path d="M10 9H8"/>',
             '<path d="M16 13H8"/>', '<path d="M16 17H8"/>'],
}

NODES = {}


def _r(nid):
    """Return (left, top, right, bottom, centre-x, centre-y) for a node."""
    x, y = NODES[nid][0], NODES[nid][1]
    return x, y, x + NW, y + NH, x + NW / 2, y + NH / 2


def gutter_x(c):
    """Centre of the gutter to the right of column c."""
    return COL[c] + NW + GUTTER / 2


def gutter_y(r):
    """Centre of the gap below row r."""
    return ROW[r] + NH + (ROW[1] - ROW[0] - NH) / 2


# ------------------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------------------
# The serverless endpoint is the subject (amber) and sits in the middle, called
# from the notebook on its left and from the MCP router above it. The training
# path runs along the bottom row (training job, then bucket), and the MCP sign-in
# pieces sit on the right of the top two rows. Edge labels carry the step
# numbers from the README.
#
# id: (x, y, colour key, icon, title, subtitle[, "dashed"])
NODES.update({
    "claude":   (COL[0], ROW[0], "navy", "users", "Claude",
                 "claude.ai or Claude Desktop"),
    "api":      (COL[1], ROW[0], "purple", "route", "HTTP API + router",
                 "Lambda: OAuth proxy + MCP"),
    "cognito":  (COL[2], ROW[0], "purple", "key", "Cognito user pool",
                 "Hosted UI, open sign-up"),
    "nb":       (COL[0], ROW[1], "blue", "terminal", "Notebook instance",
                 "ml.t3.medium; or demo.py"),
    "ep":       (COL[1], ROW[1], "amber", "monitor", "Serverless endpoint",
                 "<prefix>-endpoint, 2048 MB"),
    "ddb":      (COL[2], ROW[1], "store", "database", "DynamoDB",
                 "OAuth state table"),
    "s3":       (COL[1], ROW[2], "store", "database", "S3 bucket",
                 "data, code, training-output"),
    "tj":       (COL[0], ROW[2], "navy", "layers", "Training job",
                 "ml.m5.large, sklearn, then gone"),
})

cl, api, cg, nb, ep, ddb, s3, tj = (_r(k) for k in
    ("claude", "api", "cognito", "nb", "ep", "ddb", "s3", "tj"))

# id: (path, colour key, label, label_x, label_y, anchor[, "dashed"])
EDGES = {
    "e_data":   (f"M{nb[2]},{nb[5] + 35} H{gutter_x(0)} V{s3[5] - 22} H{s3[0] - 5}", "store",
                 "1 upload", gutter_x(0) - 8, gutter_y(1) + 6, "end"),
    "e_train":  (f"M{nb[4]},{nb[3]} V{tj[1] - 5}", "line",
                 "2 start job", nb[4] + 14, gutter_y(1) + 6, "start"),
    "e_chan":   (f"M{s3[0]},{s3[5]} H{tj[2] + 5}", "store",
                 "train / test", gutter_x(0), s3[5] + 16, "middle"),
    "e_art":    (f"M{tj[2]},{s3[5] + 30} H{s3[0] - 5}", "store",
                 "3 model.tar.gz", gutter_x(0), s3[5] + 48, "middle"),
    "e_model":  (f"M{s3[4]},{s3[1]} V{ep[3] + 5}", "store",
                 "4 ModelDataUrl", s3[4] + 14, gutter_y(1) + 6, "start"),
    "e_inv":    (f"M{nb[2]},{nb[5]} H{ep[0] - 5}", "blue",
                 "5 invoke", gutter_x(0), nb[5] - 12, "middle"),
    "e_call":   (f"M{cl[2]},{cl[5]} H{api[0] - 5}", "line",
                 "6 tools/call", gutter_x(0), cl[5] - 12, "middle"),
    "e_inv2":   (f"M{api[4]},{api[3]} V{ep[1] - 5}", "blue",
                 "7 InvokeEndpoint", api[4] + 14, gutter_y(0) + 6, "start"),
    "e_cog":    (f"M{api[2]},{api[5]} H{cg[0] - 5}", "purple",
                 "sign-in", gutter_x(1), api[5] - 12, "middle", "dashed"),
    "e_ddb":    (f"M{api[2]},{api[5] + 30} H{gutter_x(1)} V{ddb[5]} H{ddb[0] - 5}", "store",
                 "state", gutter_x(1) + 8, gutter_y(0) + 6, "start"),
}

ALT = ("A SageMaker notebook instance, or demo.py from a terminal, uploads a CSV and "
       "the training source to an S3 bucket and starts a managed training job, which "
       "reads the train and test channels on an ml.m5.large in the AWS scikit-learn "
       "container, writes model.tar.gz back to S3 and shuts down. A serverless "
       "endpoint loads that artifact through its ModelDataUrl, and the notebook calls "
       "it with IAM-signed InvokeEndpoint requests. Claude reaches the same endpoint "
       "through a remote MCP connector: an HTTP API in front of a router Lambda that "
       "proxies OAuth to a Cognito user pool, keeps OAuth state in DynamoDB, and on "
       "tools/call invokes the endpoint with its own role.")

# ------------------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------------------
from html import escape

FONT = "-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"


def node_svg(nid, t):
    """Render one service box: rounded panel, icon, title, subtitle."""
    x, y, key, icon, title, sub, *opt = NODES[nid]
    title, sub = escape(title), escape(sub)
    colour = t[key]
    dash = ' stroke-dasharray="10 7"' if opt and opt[0] == "dashed" else ""
    scale = 1.25
    isz = 24 * scale
    ix, iy = x + 20, y + (NH - isz) / 2
    tx = x + 20 + isz + 14
    return "".join([
        f'<rect x="{x}" y="{y}" width="{NW}" height="{NH}" rx="10" ry="10" '
        f'fill="{t["panel"]}" stroke="{colour}" stroke-width="2"{dash}/>',
        f'<g transform="translate({ix},{iy}) scale({scale})" fill="none" '
        f'stroke="{colour}" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round">' + "".join(ICONS[icon]) + "</g>",
        f'<text x="{tx}" y="{y + NH/2 - 4}" font-family="{FONT}" '
        f'font-size="{TITLE_PX}" font-weight="600" fill="{t["text"]}">{title}</text>',
        f'<text x="{tx}" y="{y + NH/2 + 22}" font-family="{FONT}" '
        f'font-size="{SUB_PX}" fill="{t["muted"]}">{sub}</text>',
    ])


def edge_svg(eid, t):
    """Render one arrow plus its label (on a background chip, so lines that
    pass behind a label never cut through its text)."""
    d, key, label, lx, ly, anchor, *opt = EDGES[eid]
    label = escape(label)
    colour = t[key]
    dash = ' stroke-dasharray="8 6"' if opt and opt[0] == "dashed" else ""
    out = (f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2"{dash} '
           f'marker-end="url(#a_{eid})"/>')
    if label:
        out += (f'<text x="{lx}" y="{ly}" text-anchor="{anchor}" '
                f'font-family="{FONT}" font-size="{EDGE_PX}" font-weight="600" '
                f'fill="{colour}" stroke="{t["bg"]}" stroke-width="6" '
                f'paint-order="stroke">{label}</text>')
    return out


def build(t):
    """Assemble the diagram for one theme."""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" role="img" aria-label="{ALT}">', "<defs>"]
    # One marker per edge: a shared marker cannot carry per-edge colour.
    for eid, spec in EDGES.items():
        out.append(
            f'<marker id="a_{eid}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{t[spec[1]]}"/></marker>')
    out.append("</defs>")
    out.append(f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>')
    for eid in EDGES:
        out.append(edge_svg(eid, t))
    for nid in NODES:
        out.append(node_svg(nid, t))
    out.append("</svg>")
    return "".join(out)


def main():
    for name, theme in THEMES.items():
        path = os.path.join(HERE, f"architecture-{name}.svg")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(build(theme))
        print(f"  architecture-{name}.svg")


if __name__ == "__main__":
    main()
