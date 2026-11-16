// Cloudflare Pages worker for the runs browser. Public and read-only (Sam, 2026-11-08): no token needed.
// The same API as scripts/runs_browser.py, answered from the shards scripts/build_site.py writes.
class ApiError extends Error {
  constructor(status, code, message) { super(message); this.status = status; this.code = code; }
}
const HEAD = { "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff" };
const send = (body, type, status = 200, extra = {}) =>
  new Response(body, { status, headers: { "Content-Type": type + "; charset=utf-8", ...HEAD, ...extra } });
const sendJson = (v, status = 200) => send(JSON.stringify(v), "application/json", status);

function fnv(word) {
  let h = 0x811c9dc5;
  for (const c of new TextEncoder().encode(word)) h = Math.imul(h ^ c, 0x01000193) >>> 0;
  return h;
}
const wordsOf = s => [...new Set((s.toLowerCase().match(/[\p{L}\p{N}]+/gu)) || [])];

function same(a, b) {
  const x = new TextEncoder().encode(a), y = new TextEncoder().encode(b);
  return x.length === y.length && crypto.subtle.timingSafeEqual(x, y);
}
function authorised(request, url, token) {
  const given = [...url.searchParams.getAll("token")];
  const auth = request.headers.get("Authorization") || "";
  if (auth.startsWith("Bearer ")) given.push(auth.slice(7));
  for (const part of (request.headers.get("Cookie") || "").split(";")) {
    const [k, v] = part.trim().split("=");
    if (k === "runs_token" && v) given.push(v);
  }
  return given.some(g => same(g, token));
}

function intParam(q, name, dflt, low, high) {
  if (!q.has(name)) return dflt;
  const v = Number(q.get(name));
  if (!Number.isInteger(v) || q.get(name).trim() === "") throw new ApiError(400, "invalid_parameter", `${name} must be an integer.`);
  if (v < low || v > high) throw new ApiError(400, "invalid_parameter", `${name} must be from ${low} to ${high}.`);
  return v;
}
function paging(total, q, dflt, max) {
  const limit = intParam(q, "limit", dflt, 1, max), offset = intParam(q, "offset", 0, 0, 1e9), end = offset + limit;
  return { total, offset, limit, next_offset: end < total ? end : null };
}
function checkParams(q, allowed) {
  const bad = [...new Set(q.keys())].filter(k => k !== "token" && !allowed.includes(k)).sort();
  if (bad.length) throw new ApiError(400, "unknown_parameter", `Unknown parameter ${bad[0]}. Allowed: ${[...allowed].sort().join(", ") || "none"}.`);
}

function store(env, origin) {
  const raw = async p => {
    const r = await env.ASSETS.fetch(new Request(origin + "/d/" + p));
    return r.ok ? r : null;
  };
  return {
    json: async p => { const r = await raw(p); return r ? r.json() : null; },
    text: async p => { const r = await raw(p); return r ? r.text() : null; },
  };
}
function safeId(id) { return /^[A-Za-z0-9._~-]+$/.test(id) && !id.startsWith("."); }
async function getRun(s, rid) {
  const run = safeId(rid) ? await s.json(`r/${rid}/run.json`) : null;
  if (!run) throw new ApiError(404, "run_not_found", `No run '${rid}'. List runs at /api/v1/runs.`);
  return run;
}
async function dataRun(s, rid) {
  const run = await getRun(s, rid);
  if (run.kind === "log") throw new ApiError(400, "is_a_log", "This run is a log. Read it at /api/v1/runs/{run_id}/log.");
  return run;
}

async function texts(s, rid, q) {
  checkParams(q, ["collection", "q", "filter", "limit", "offset", "max_chars"]);
  await dataRun(s, rid);
  const coll = await s.json(`r/${rid}/coll.json`);
  if (!coll.default) return { run_id: rid, collection: null, rows: [], total: 0, offset: 0, limit: 0, next_offset: null };
  const name = q.get("collection") || coll.default;
  if (!(name in coll.names)) throw new ApiError(400, "invalid_parameter", `Unknown collection. Available: ${Object.keys(coll.names).join(", ")}.`);
  const base = `r/${rid}/c/${coll.names[name]}`;
  const index = await s.json(`${base}/index.json`);
  const conds = [];
  for (const raw of q.getAll("filter")) {
    const i = raw.indexOf(":");
    const facet = i > 0 ? index.facets.find(f => f.name === raw.slice(0, i)) : null;
    if (!facet) throw new ApiError(400, "invalid_parameter", `filter must be name:value. Names: ${index.facets.map(f => f.name).join(", ") || "none"}.`);
    conds.push([index.cols[facet.position], facet.values.indexOf(raw.slice(i + 1))]);
  }
  const maxChars = intParam(q, "max_chars", 0, 0, 1e7);
  let ids = null;
  for (const w of wordsOf(q.get("q") || "")) {
    const post = (await s.json(`${base}/p${fnv(w) % index.post}.json`))[w] || [];
    if (ids === null) ids = post;
    else { const keep = new Set(post); ids = ids.filter(x => keep.has(x)); }
  }
  if (conds.length) {
    const ok = i => conds.every(([col, v]) => v >= 0 && col[i] === v);
    if (ids === null) { ids = []; for (let i = 0; i < index.n; i++) if (ok(i)) ids.push(i); }
    else ids = ids.filter(ok);
  }
  const total = ids === null ? index.n : ids.length;
  const meta = paging(total, q, 100, 500);
  const wanted = [];
  for (let i = meta.offset; i < Math.min(total, meta.offset + meta.limit); i++) wanted.push(ids === null ? i : ids[i]);
  const numbers = [...new Set(wanted.map(i => Math.floor(i / 100)))];
  const shards = new Map(await Promise.all(numbers.map(async n => [n, await s.json(`${base}/s${n}.json`)])));
  const rows = wanted.map(i => {
    const r = shards.get(Math.floor(i / 100))[i % 100];
    return { ...r, text: maxChars ? r.text.slice(0, maxChars) : r.text, truncated: !!(maxChars && r.text.length > maxChars) };
  });
  return { run_id: rid, collection: name, rows, ...meta };
}

async function results(s, rid, q) {
  checkParams(q, ["q", "limit", "offset"]);
  await dataRun(s, rid);
  const needle = (q.get("q") || "").toLowerCase();
  const all = (await s.json(`r/${rid}/results.json`)).filter(r => r.key.toLowerCase().includes(needle));
  const meta = paging(all.length, q, 200, 2000);
  return { run_id: rid, rows: all.slice(meta.offset, meta.offset + meta.limit), ...meta };
}

const PART = 4000000;
async function field(s, rid, key, q) {
  checkParams(q, ["prefix", "offset", "limit_chars"]);
  await dataRun(s, rid);
  const f = (await s.json(`r/${rid}/fields.json`))[key];
  if (!f) throw new ApiError(404, "field_not_found", `No field '${key}'. See fields in /api/v1/runs/${rid}.`);
  const offset = intParam(q, "offset", 0, 0, 1e10), limit = intParam(q, "limit_chars", 200000, 1, 2000000);
  if (q.has("prefix")) {
    if (f.total > 500000) throw new ApiError(400, "field_too_large", "prefix works on fields up to 500,000 characters here. Page through this one with offset and limit_chars.");
    const value = JSON.parse(await s.text(`r/${rid}/f/${f.i}/0.txt`));
    if (Array.isArray(value) || typeof value !== "object" || value === null) throw new ApiError(400, "invalid_parameter", "prefix works only on object fields.");
    const text = JSON.stringify(Object.fromEntries(Object.entries(value).filter(([k]) => k.startsWith(q.get("prefix")))), null, 1);
    return { run_id: rid, key, json_text: text.slice(offset, offset + limit), total_chars: text.length, offset, next_offset: offset + limit < text.length ? offset + limit : null };
  }
  const end = Math.min(f.total, offset + limit);
  let text = "";
  for (let n = Math.floor(offset / PART); n * PART < end; n++) {
    const part = await s.text(`r/${rid}/f/${f.i}/${n}.txt`);
    text += part.slice(Math.max(0, offset - n * PART), end - n * PART);
  }
  return { run_id: rid, key, json_text: text, total_chars: f.total, offset, next_offset: end < f.total ? end : null };
}

async function log(s, rid, q) {
  checkParams(q, ["tail_chars"]);
  const run = await getRun(s, rid);
  if (run.kind !== "log") throw new ApiError(400, "not_a_log", "This run is not a log.");
  const l = await s.json(`r/${rid}/log.json`);
  const tail = intParam(q, "tail_chars", 200000, 1, 2000000);
  return { run_id: rid, text: l.text.slice(-tail), total_chars: l.total_chars, truncated: l.total_chars > tail };
}

async function search(s, q) {
  checkParams(q, ["q", "iteration", "limit", "offset"]);
  const raw = (q.get("q") || "").trim();
  if (raw.length < 2) throw new ApiError(400, "invalid_parameter", "q must be at least 2 characters.");
  const only = q.get("iteration");
  if (only && !(safeId(only) && await s.json(`it/${only}.json`))) throw new ApiError(404, "iteration_not_found", `No iteration '${only}'.`);
  const runs = await s.json("runs.json");
  let found = null;
  for (const w of wordsOf(raw)) {
    const post = (await s.json(`s/${fnv(w) % 256}.json`))[w] || {};
    if (found === null) found = new Map(Object.entries(post));
    else {
      const next = new Map();
      for (const [k, v] of found) if (k in post) next.set(k, Math.min(v, post[k]));
      found = next;
    }
  }
  let hits = [...(found || new Map())].map(([i, n]) => ({ ...runs[Number(i)], matches: n }));
  if (only) hits = hits.filter(r => r.iteration_id === only);
  hits.sort((a, b) => b.matches - a.matches || (a.name < b.name ? -1 : 1));
  const meta = paging(hits.length, q, 50, 200);
  return { query: raw, rows: hits.slice(meta.offset, meta.offset + meta.limit), ...meta };
}

async function listRuns(s, q) {
  checkParams(q, ["iteration", "kind", "role", "model", "limit", "offset"]);
  let rows = await s.json("runs.json");
  for (const [key, f] of [["iteration", "iteration_id"], ["kind", "kind"], ["role", "role"], ["model", "model"]])
    if (q.has(key)) rows = rows.filter(r => r[f] === q.get(key));
  const meta = paging(rows.length, q, 200, 1000);
  return { rows: rows.slice(meta.offset, meta.offset + meta.limit), ...meta };
}

async function route(s, path, q) {
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  if (!parts.length) return { name: "Knob Ladder Runs API", version: "1.0.0", guide: "/llms.txt", openapi: "/openapi.json", start: "/api/v1/experiments" };
  const [head, a, b, c] = parts;
  if (head === "experiments" && parts.length === 1) {
    checkParams(q, ["track"]);
    let rows = await s.json("experiments.json");
    if (q.has("track")) rows = rows.filter(r => r.track === q.get("track"));
    return { rows, total: rows.length };
  }
  if (head === "experiments" && parts.length === 2) {
    const e = safeId(a) ? await s.json(`exp/${a}.json`) : null;
    if (!e) throw new ApiError(404, "experiment_not_found", "See /api/v1/experiments.");
    return e;
  }
  if (head === "iterations" && parts.length === 2) {
    const it = safeId(a) ? await s.json(`it/${a}.json`) : null;
    if (!it) throw new ApiError(404, "iteration_not_found", "See /api/v1/experiments.");
    return it;
  }
  if (head === "search" && parts.length === 1) return search(s, q);
  if (head === "runs" && parts.length === 1) return listRuns(s, q);
  if (head === "runs" && parts.length === 2) { checkParams(q, []); return getRun(s, a); }
  if (head === "runs" && parts.length === 3 && b === "results") return results(s, a, q);
  if (head === "runs" && parts.length === 3 && b === "texts") return texts(s, a, q);
  if (head === "runs" && parts.length === 3 && b === "log") return log(s, a, q);
  if (head === "runs" && parts.length === 4 && b === "fields") return field(s, a, c, q);
  if (head === "runs" && (parts.length === 3 || parts.length === 4) && b === "conversations") {
    checkParams(q, []);
    if (!safeId(a)) throw new ApiError(404, "run_not_found", "See /api/v1/runs.");
    if (parts.length === 3) return (await s.json(`r/${a}/conv/index.json`)) || { rows: [], total: 0 };
    const chunk = /^\d+$/.test(c) ? await s.json(`r/${a}/conv/c${Math.floor(+c / 20)}.json`) : null;
    const conv = chunk ? chunk[+c % 20] : null;
    if (!conv) throw new ApiError(404, "conversation_not_found", "See /api/v1/runs/{run_id}/conversations.");
    return conv;
  }
  throw new ApiError(404, "not_found", "See /openapi.json for the routes.");
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";
    try {
      if (request.method !== "GET" && request.method !== "HEAD") throw new ApiError(405, "method_not_allowed", "This API is read-only.");
      const s = store(env, url.origin);
      if (path === "/" && url.searchParams.has("token"))
        return new Response(null, { status: 303, headers: { Location: "/", ...HEAD } });  // old token links still work
      if (path === "/") {  // the story page; old /#run= and /#i= links are forwarded to /runs by the page itself
        const story = await s.text("story.html");
        if (story) return send(story, "text/html");
      }
      if (path === "/" || path === "/runs") return send((await s.text("page.txt")).replace("__TOKEN__", "null"), "text/html");
      if (path === "/notebook") {
        const nb = await s.text("notebook.html");
        if (nb) return send(nb, "text/html");
      }
      if (path === "/llms.txt") return send((await s.text("guide.txt")).replaceAll("__BASE__", url.origin).replace(/^Auth:.*$/m, "Auth: none. This API is public and read-only."), "text/plain");
      if (path === "/openapi.json") return send((await s.text("openapi.json")).replaceAll("__BASE__", url.origin), "application/json");
      if (path === "/api/v1" || path.startsWith("/api/v1/")) return sendJson(await route(s, path.slice(7), url.searchParams));
      throw new ApiError(404, "not_found", "See /llms.txt and /openapi.json.");
    } catch (e) {
      if (e instanceof ApiError) return sendJson({ error: { code: e.code, message: e.message } }, e.status);
      console.log(String(e));
      return sendJson({ error: { code: "internal_error", message: "The server hit an error reading this." } }, 500);
    }
  },
};
