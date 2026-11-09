import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { createServer, request } from "node:http";
import { fileURLToPath } from "node:url";
import test from "node:test";

const root = fileURLToPath(new URL(".", import.meta.url));
const jpeg = Buffer.from([255, 216, 0, 128, 42, 255, 217]);
const audio = { source: "hardware", status: "recognized", stale: false, sequence: 42,
  notes: [{ note_name: "A4", frequency_hz: 440, confidence: .9 }],
  config: { sample_rate: 48000, fft_size: 4096 }, spectrum: { bins: [0, 2, 1] } };
const vision = { status: "streaming", source: "camera", sequence: 42, frame_available: true };
const metadata = { "X-Vision-Sequence": "42", "X-Vision-Captured-Ms": "1234.5",
  "X-Vision-Source": "camera", "X-Vision-Stream": "fixture-stream",
  "X-Vision-Tracking": JSON.stringify({ tracker_id: "fixture", status: "tracking" }),
  "X-Vision-Fretboard": JSON.stringify({ status: "tracking", sequence: 42 }) };

function getResponse(url, { method = "GET", body, headers = {} } = {}) {
  return new Promise((resolve, reject) => {
    const outgoing = request(url, { method, agent: false, signal: AbortSignal.timeout(6000),
      maxHeaderSize: 128 * 1024,
      headers: body === undefined ? headers : { ...headers, "Content-Length": body.length } }, async response => {
      try {
        const chunks = [];
        for await (const chunk of response) chunks.push(chunk);
        resolve({ status: response.statusCode, headers: response.headers, body: Buffer.concat(chunks) });
      } catch (error) { reject(error); }
    });
    outgoing.once("error", reject);
    outgoing.end(body);
  });
}

test("PC JPEG transport forwards bytes without PC recognition or recompression", { timeout: 15000 }, async t => {
  const calls = [];
  const fixture = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    calls.push({ headers: request.headers, body: Buffer.concat(chunks) });
    const planes = request.headers['x-vision-encoding'] === 'identity';
    response.writeHead(200, { "Content-Type": planes ? 'application/x-fretlens-planes' : "image/jpeg", "X-Vision-Backend": "pl",
      "X-Vision-Source": "pc_camera", "X-Vision-Token": "jpeg-1", "X-Vision-Processing-Ms": "42",
      "X-Vision-Fretboard": JSON.stringify({ status: "tracking", fingertip_neural: { preview: { jpeg_base64: "a".repeat(32000) } } }) });
    response.end(planes ? Buffer.alloc(320*240*2,73) : jpeg);
  });
  await new Promise((resolve, reject) => fixture.once("error", reject).listen(8084, "127.0.0.1", resolve));
  t.after(() => new Promise(resolve => fixture.close(resolve)));
  const base = await startProxy(t, "http://127.0.0.1:8084", { bind: "127.0.0.1" });
  const result = await getResponse(`${base}/api/vision/pc/process`, { method: "POST", body: jpeg,
    headers: { "Content-Type": "image/jpeg", "X-Vision-Token": "jpeg-1" } });
  assert.equal(result.status, 200);
  assert.deepEqual(calls[0].body, jpeg);
  assert.equal(calls[0].headers["x-vision-encoding"], "jpeg");
  assert.equal(calls[0].headers["content-type"], "application/octet-stream");
  assert.equal(result.headers["x-vision-processing-ms"], "42");
  assert.equal(JSON.parse(result.headers["x-vision-fretboard"]).fingertip_neural.preview.jpeg_base64.length, 32000);
  const rejected = await getResponse(`${base}/api/vision/pc/process`, { method: "POST", body: jpeg,
    headers: { "Content-Type": "image/jpeg", "X-Vision-Options": JSON.stringify({ transport_roi: {x:0,y:0,width:1,height:1} }) } });
  assert.equal(rejected.status, 400);
  assert.equal(calls.length, 1);
  const raw = Buffer.alloc(640*480*4,123);
  const rawResult = await getResponse(`${base}/api/vision/pc/process`, {method:'POST',body:raw,
    headers:{'Content-Type':'application/octet-stream','X-Vision-Token':'raw-1',
      'X-Vision-Transport':'identity','X-Vision-Options':JSON.stringify({preview_encoding:'planes'})}});
  assert.equal(rawResult.status,200);
  assert.deepEqual(calls[1].body,raw);
  assert.equal(calls[1].headers['x-vision-encoding'],'identity');
  assert.equal(rawResult.headers['content-type'],'application/x-fretlens-planes');
  assert.deepEqual(rawResult.body,Buffer.alloc(320*240*2,73));
});

async function startProxy(t, boardUrl, { entry = "server.mjs", bind = "127.0.0.2" } = {}) {
  const child = spawn(process.execPath, [entry], { cwd: root, windowsHide: true,
    env: { ...process.env, PORT: "0", FPGA_API_URL: boardUrl, FPGA_VISION_URL: boardUrl,
      FPGA_BIND_ADDRESS: bind }, stdio: ["ignore", "pipe", "pipe"] });
  t.after(async () => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    const exited = once(child, "exit");
    child.kill();
    await exited;
  });
  return new Promise((resolve, reject) => {
    let output = "", errors = "";
    const timer = setTimeout(() => reject(new Error(`Proxy startup timed out: ${errors}`)), 5000);
    child.stderr.on("data", chunk => { errors += chunk; });
    child.stdout.on("data", chunk => {
      output += chunk;
      const match = output.match(/http:\/\/127\.0\.0\.1:(\d+)/);
      if (match) { clearTimeout(timer); resolve(`http://127.0.0.1:${match[1]}`); }
    });
    child.once("error", error => { clearTimeout(timer); reject(error); });
    child.once("exit", code => { clearTimeout(timer); reject(new Error(`Proxy exited ${code}: ${errors}`)); });
  });
}

test("invalid URI returns 400 without terminating the web service", { timeout: 10000 }, async t => {
  const base = await startProxy(t, "http://127.0.0.1:9");
  for (const path of ["/%ZZ", "/%E0%A4%A", "/%00"]) {
    assert.equal((await getResponse(base + path)).status, 400);
    assert.equal((await getResponse(base + "/")).status, 200);
  }
});

function assertOffline(response, kind) {
  assert.equal(response.headers["cache-control"], "no-store");
  assert.equal(response.headers["content-type"], "application/json; charset=utf-8");
  const payload = JSON.parse(response.body);
  assert.equal(typeof payload.error, "string");
  assert.ok(payload.error.length > 0);
  if (kind === "audio") {
    assert.equal(response.status, 200);
    assert.deepEqual(Object.keys(payload).sort(), ["error", "notes", "source", "stale", "status"]);
    assert.equal(payload.source, "hardware");
    assert.equal(payload.status, "hardware_unavailable");
    assert.equal(payload.stale, true);
    assert.deepEqual(payload.notes, []);
  } else {
    assert.equal(response.status, 503);
    assert.deepEqual(Object.keys(payload).sort(), ["error", "source", "status"]);
    assert.equal(payload.status, "offline");
    assert.equal(payload.source, null);
  }
}

test("board proxy binds upstream requests and preserves its contracts", { timeout: 20000 }, async t => {
  const calls = [];
  let mode = "ok";
  const fixture = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const body = Buffer.concat(chunks);
    calls.push({ url: request.url, method: request.method, headers: request.headers,
      address: request.socket.remoteAddress, body });
    if (mode === "disconnect") { request.socket.destroy(); return; }
    if (mode === "truncated") {
      response.writeHead(200, { "Content-Type": "application/json", "Content-Length": "100" });
      response.write("{");
      setImmediate(() => response.destroy());
      return;
    }
    if (mode === "timeout") {
      if (request.url === "/api/audio") return;
      response.writeHead(200, { "Content-Type": "image/jpeg" });
      response.flushHeaders();
      const timer = setInterval(() => response.write("x"), 40);
      response.once("close", () => clearInterval(timer));
      return;
    }
    if (mode === "http-error") {
      response.writeHead(502, { "Content-Type": "application/json", ...metadata });
      response.end('{"error":"upstream unavailable"}');
      return;
    }
    if (mode === "invalid-json") { response.end("not JSON"); return; }
    if (request.url.startsWith("/vision/frame.jpg")) {
      response.writeHead(200, { "Content-Type": "image/jpeg", ...metadata, "X-Unrelated": "private" });
      response.end(jpeg);
    } else if (request.url.startsWith("/api/vision/tracking")) {
      response.writeHead(201, { "Content-Type": "application/json" });
      response.end(body);
    } else {
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(JSON.stringify(request.url === "/api/audio" ? audio : vision));
    }
  });
  fixture.listen(0, "127.0.0.1");
  await once(fixture, "listening");
  t.after(async () => {
    const closed = new Promise(resolve => fixture.close(resolve));
    fixture.closeAllConnections();
    await closed;
  });
  const boardUrl = `http://127.0.0.1:${fixture.address().port}`;
  const proxy = await startProxy(t, boardUrl);

  await t.test("JSON audio and vision requests use the selected source address", async () => {
    for (const [path, payload] of [["/api/audio", audio], ["/api/vision/status", vision]]) {
      const response = await getResponse(proxy + path);
      assert.equal(response.status, 200);
      assert.deepEqual(JSON.parse(response.body), payload);
      assert.equal(response.headers["cache-control"], "no-store");
      assert.equal(calls.at(-1).address, "127.0.0.2");
      assert.equal(calls.at(-1).method, "GET");
    }
  });

  await t.test("frame bytes, query and all six metadata headers survive forwarding", async () => {
    const path = "/vision/frame.jpg?tracking=fixture%2Fid&auto=1";
    const response = await getResponse(proxy + path);
    assert.equal(response.status, 200);
    assert.deepEqual(response.body, jpeg);
    assert.equal(response.headers["content-type"], "image/jpeg");
    assert.equal(response.headers["cache-control"], "no-store");
    for (const [key, value] of Object.entries(metadata)) assert.equal(response.headers[key.toLowerCase()], value);
    assert.equal(response.headers["x-unrelated"], undefined);
    assert.equal(calls.at(-1).url, path);
    assert.equal(calls.at(-1).address, "127.0.0.2");
  });

  await t.test("both tracking POST routes preserve body bytes and JSON content type", async () => {
    const body = Buffer.from('{ "tracker_id": "fixture", "sequence": 42 }');
    for (const path of ["/api/vision/tracking", "/api/vision/tracking/stop"]) {
      const response = await getResponse(proxy + path + "?client=fixture", { method: "POST", body });
      assert.equal(response.status, 201);
      assert.deepEqual(response.body, body);
      assert.equal(calls.at(-1).method, "POST");
      assert.equal(calls.at(-1).url, path + "?client=fixture");
      assert.equal(calls.at(-1).headers["content-type"], "application/json");
      assert.equal(calls.at(-1).headers["content-length"], String(body.length));
      assert.equal(calls.at(-1).address, "127.0.0.2");
      assert.deepEqual(calls.at(-1).body, body);
    }
  });

  await t.test("tracking accepts 4096 bytes and rejects 4097 before contacting the board", async () => {
    const body = Buffer.from(JSON.stringify({ padding: "x".repeat(4096 - Buffer.byteLength('{"padding":""}')) }));
    assert.equal(body.length, 4096);
    assert.equal((await getResponse(proxy + "/api/vision/tracking", { method: "POST", body })).status, 201);
    const count = calls.length;
    const response = await getResponse(proxy + "/api/vision/tracking", {
      method: "POST", body: Buffer.concat([body, Buffer.from(" ")])
    });
    assert.equal(response.status, 413);
    assert.equal(response.body.toString(), "Body too large");
    assert.equal(calls.length, count);
  });

  await t.test("vision method restrictions reject requests before contacting the board", async () => {
    const count = calls.length;
    for (const [path, method] of [["/api/vision/status", "POST"], ["/vision/frame.jpg", "POST"],
      ["/api/vision/tracking", "GET"], ["/api/vision/tracking/stop", "GET"]]) {
      assert.equal((await getResponse(proxy + path, { method })).status, 405);
    }
    assert.equal(calls.length, count);
  });

  await t.test("upstream HTTP errors retain vision status and the audio fallback schema", async () => {
    mode = "http-error";
    assertOffline(await getResponse(proxy + "/api/audio"), "audio");
    const response = await getResponse(proxy + "/api/vision/status");
    assert.equal(response.status, 502);
    assert.deepEqual(JSON.parse(response.body), { error: "upstream unavailable" });
    assert.equal(response.headers["x-vision-sequence"], "42");
    mode = "invalid-json";
    assertOffline(await getResponse(proxy + "/api/audio"), "audio");
  });

  await t.test("connection resets and truncated bodies produce the existing offline responses", async () => {
    for (mode of ["disconnect", "truncated"]) {
      assertOffline(await getResponse(proxy + "/api/audio"), "audio");
      assertOffline(await getResponse(proxy + "/vision/frame.jpg"), "vision");
    }
  });

  await t.test("the 2.5 second deadline covers missing headers and a trickling response body", async () => {
    mode = "timeout";
    const started = performance.now();
    const [audioResponse, frameResponse] = await Promise.all([
      getResponse(proxy + "/api/audio"), getResponse(proxy + "/vision/frame.jpg")
    ]);
    assertOffline(audioResponse, "audio");
    assertOffline(frameResponse, "vision");
    const elapsed = performance.now() - started;
    assert.ok(elapsed >= 2000 && elapsed < 5500, `Unexpected deadline: ${elapsed}ms`);
  });

  await t.test("later requests recover after upstream failures", async () => {
    mode = "ok";
    assert.deepEqual(JSON.parse((await getResponse(proxy + "/api/audio")).body), audio);
    assert.deepEqual((await getResponse(proxy + "/vision/frame.jpg")).body, jpeg);
  });

  await t.test("the board launcher honors URL, port and source-address overrides", async t => {
    const launcher = await startProxy(t, boardUrl, { entry: "serve_board.mjs" });
    assert.deepEqual(JSON.parse((await getResponse(launcher + "/api/audio")).body), audio);
    assert.equal(calls.at(-1).address, "127.0.0.2");
  });

  await t.test("the general server permits an omitted source address", async t => {
    const unbound = await startProxy(t, boardUrl, { bind: "" });
    assert.deepEqual(JSON.parse((await getResponse(unbound + "/api/audio")).body), audio);
    assert.equal(calls.at(-1).address, "127.0.0.1");
  });
});
