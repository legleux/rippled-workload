# XRPL Explorer: Embeddable Ledger Stream Widget

## Problem

The XRPL Explorer at `custom.xrpl.org` renders a live ledger stream that's useful to embed in external dashboards. Currently, the only way to embed it is via a full-page iframe:

```html
<iframe src="https://custom.xrpl.org/localhost:6006"></iframe>
```

This renders the **entire** explorer page — navbar, legend toggle, metrics panel, footer — when the embedding application only wants the **ledger list** (the scrolling list of closed ledgers with their transaction counts).

### Current workaround

To isolate the ledger list, we use CSS to offset and clip the iframe:

```css
.explorer-viewport {
    position: relative;
    width: 100%;
    height: 400px;
    overflow: hidden;          /* clip everything outside the viewport */
}
.explorer-viewport iframe {
    position: absolute;
    top: -385px;               /* crop navbar, legend, metrics header */
    left: 0;
    width: 100%;
    height: calc(100% + 600px); /* oversize so content fills the viewport */
    border: none;
}
```

This is fragile — any layout change in the explorer shifts the pixel offsets and breaks the embed. Cross-origin restrictions prevent injecting CSS into the iframe to hide elements directly.

## Proposed Changes to the Explorer

### Option A: Query parameter for embed mode (recommended)

Add a `?embed=ledgers` query parameter that renders only the ledger list component, stripped of all chrome:

```
https://custom.xrpl.org/localhost:6006?embed=ledgers
```

**What embed mode would render:**
- The live-updating ledger list (`<div class="ledger-list">` or equivalent)
- Transparent or dark background (no page chrome)
- No navbar
- No legend / "Show legend" toggle
- No metrics panel
- No footer
- No padding/margins around the component

**What embed mode would NOT change:**
- WebSocket connection behavior (still connects to the specified node)
- Ledger data rendering (same content, same updates)
- Click behavior (clicking a ledger could still open the full explorer in a new tab)

**Implementation sketch** (React):

```jsx
// In the router or top-level layout
const params = new URLSearchParams(window.location.search);
const embedMode = params.get('embed');

if (embedMode === 'ledgers') {
    return <LedgerList node={node} />;
}

// Otherwise render full page
return (
    <Layout>
        <Navbar />
        <Legend />
        <LedgerList node={node} />
        <Metrics />
        <Footer />
    </Layout>
);
```

**CSS for embed mode:**

```css
/* When in embed mode, make the component fill its container */
body.embed-mode {
    margin: 0;
    padding: 0;
    background: transparent;
    overflow: hidden;
}
```

**Embedding would then be:**

```html
<iframe src="https://custom.xrpl.org/localhost:6006?embed=ledgers"
        style="width:100%;height:400px;border:none"></iframe>
```

No offset hacks, no clipping, no fragility.

### Option B: CORS headers for same-origin CSS injection

Add CORS headers (`Access-Control-Allow-Origin: *` or a configurable allowlist) so embedding pages can access `contentDocument` and inject CSS to hide elements.

**Pros:** More flexible — embedders can choose what to show/hide.
**Cons:** Security implications of allowing cross-origin DOM access. More complex for embedders. Fragile (depends on class names not changing).

**Not recommended** — Option A is simpler and safer.

### Option C: Standalone web component / npm package

Export the ledger list as a standalone web component or npm package that can be imported directly without an iframe:

```html
<script src="https://custom.xrpl.org/components/ledger-list.js"></script>
<xrpl-ledger-list node="localhost:6006"></xrpl-ledger-list>
```

**Pros:** Best DX, no iframe overhead, fully styleable.
**Cons:** Significant engineering effort, versioning/distribution concerns.

**Good long-term goal** but Option A is the pragmatic first step.

## Suggested embed parameter values

Beyond `?embed=ledgers`, future values could include:

| Parameter | Renders |
|---|---|
| `?embed=ledgers` | Ledger list only |
| `?embed=transactions` | Transaction feed only |
| `?embed=network` | Network topology / node status |

## Impact on this project

Once `?embed=ledgers` is supported, the workload dashboard changes from:

```css
/* 8 lines of fragile offset/clip CSS */
.explorer-viewport { ... overflow: hidden; }
.explorer-viewport iframe { position: absolute; top: -385px; height: calc(100% + 600px); }
```

To:

```html
<iframe src="https://custom.xrpl.org/localhost:6006?embed=ledgers"
        style="width:100%;height:400px;border:none"></iframe>
```
