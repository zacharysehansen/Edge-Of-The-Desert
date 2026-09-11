// check_frontend.mjs — load the frontend the way a BROWSER does, and render it.
//
// This is the fifth Phase 3 gate, and for a long time it was the only one with no
// command: a DOM shim that had to be rewritten from scratch every time. Committing it
// mattered more than it looked, because the other four gates all run under Node and
// Node is exactly where this class of defect hides.
//
// It caught a live one. frontend/structural.js and frontend/catalog.js both ended with
//
//     if (process?.argv?.includes('--dump')) { ... }
//
// Optional chaining short-circuits a property whose value is null or undefined. It does
// NOT protect an identifier that was never declared, and in a browser `process` is
// undeclared, so that line threw ReferenceError at module load. structural.js died,
// and models.js and ui.js died importing it: the page drew its controls and then never
// loaded a single model. Every other gate passed throughout, on a frontend that could
// not start in the only environment it ships to.
//
// So this check does three things the others cannot:
//   1. deletes the Node-only globals, which reproduces the browser exactly — verified:
//      after `delete globalThis.process`, a bare `process` throws ReferenceError while
//      `typeof process` stays safe, which is the difference that matters;
//   2. stubs the DOM, so ui.js can build its panels and be read back;
//   3. stubs onnxruntime-web, which the browser supplies via index.html's importmap.
//
// It does not replace opening the page. It replaces *not* opening the page.
//
// Run:  node scripts/phase3/check_frontend.mjs

import { register } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const stub = pathToFileURL(path.join(here, '_ort_stub.mjs')).href;

register(`data:text/javascript,
export async function resolve(spec, ctx, next) {
    if (spec.startsWith('onnxruntime-web')) return { url: ${JSON.stringify(stub)}, shortCircuit: true };
    return next(spec, ctx);
}`);

// ── the DOM, in about twenty lines ───────────────────────────────────────────
const make = (tag) => {
    const el = {
        tagName: tag, children: [], style: {}, dataset: {}, className: '', title: '',
        classList: { add() {}, remove() {}, toggle() {} }, _text: '',
        setAttribute() {}, removeAttribute() {}, addEventListener() {},
        appendChild(c) { this.children.push(c); return c; },
        append(...c) { this.children.push(...c); },
        insertBefore(c) { this.children.push(c); return c; },
        querySelector: () => null, querySelectorAll: () => [], remove() {},
    };
    Object.defineProperty(el, 'textContent', {
        get: () => el._text,
        set: (v) => { el._text = String(v); el.children.length = 0; },
    });
    Object.defineProperty(el, 'innerHTML', {
        get: () => el._text, set: (v) => { el._text = String(v); },
    });
    return el;
};
const byId = new Map();
globalThis.document = {
    createElement: make,
    createTextNode: (t) => { const n = make('#text'); n.textContent = t; return n; },
    getElementById: (id) => {
        if (!byId.has(id)) byId.set(id, make('div'));
        return byId.get(id);
    },
    querySelector: () => null, querySelectorAll: () => [], addEventListener() {},
    body: make('body'), documentElement: make('html'),
};
globalThis.window = {
    addEventListener() {},
    matchMedia: () => ({ matches: false, addEventListener() {} }),
};
// Models are fetched over HTTP from /model/ in the browser. Refusing here keeps the
// check about module loading and rendering; inference itself is covered headlessly by
// slider_sensitivity.py against the real ONNX files.
globalThis.fetch = async () => { throw new Error('fetch is stubbed in check_frontend'); };

// ── make Node look like a browser ────────────────────────────────────────────
const NODE_ONLY = ['process', 'require', '__dirname', '__filename', 'Buffer', 'global'];
const saved = {};
for (const key of NODE_ONLY) {
    if (key in globalThis) { saved[key] = globalThis[key]; delete globalThis[key]; }
}
const restore = () => { for (const [k, v] of Object.entries(saved)) globalThis[k] = v; };

const fail = (msg, err) => {
    restore();
    console.error(`\nFAIL  ${msg}`);
    if (err) console.error(`      ${err.constructor.name}: ${err.message}`);
    if (err instanceof ReferenceError) {
        console.error('\nA ReferenceError at module load means the frontend references a');
        console.error("Node-only global unguarded. Use `typeof x !== 'undefined'`, never");
        console.error('`x?.` — optional chaining does not guard an undeclared identifier.');
    }
    globalThis.process?.exit?.(1);
    throw err ?? new Error(msg);
};

console.log('=== frontend, loaded as a browser sees it ===\n');

// Imported one at a time so a failure names the module that actually broke.
for (const mod of ['state.js', 'structural.js', 'catalog.js', 'models.js']) {
    try {
        await import(`../../frontend/${mod}`);
        console.log(`  ok    ${mod}`);
    } catch (err) { fail(`${mod} does not load in a browser`, err); }
}

let ui;
try {
    ui = await import('../../frontend/ui.js');
    console.log('  ok    ui.js');
} catch (err) { fail('ui.js does not load in a browser', err); }

try {
    await ui.init();
} catch (err) {
    // init() reaching the network is expected here; a ReferenceError is not.
    if (err instanceof ReferenceError) fail('ui.init() threw a ReferenceError', err);
}

const walk = (el, out = []) => {
    if (el._text && el._text.trim()) out.push(el._text.trim());
    for (const child of el.children ?? []) walk(child, out);
    return out;
};
let rendered = 0;
const panels = [];
for (const [id, el] of byId) {
    const text = walk(el);
    rendered += text.length;
    if (text.length) panels.push(`${id} (${text.length})`);
}

restore();

if (rendered === 0) {
    console.error('\nFAIL  ui.init() rendered nothing. The modules loaded but the page is blank.');
    process.exit(1);
}
console.log(`\n  rendered ${rendered} text nodes across ${panels.length} panels: ${panels.join(', ')}`);
console.log('\nFrontend loads and renders with no Node-only globals available.');
