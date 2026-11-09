import { createServer, request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { requestBoard } from "./board_proxy.mjs";
import { deflate } from "node:zlib";
import { promisify } from "node:util";

const compressFrame = promisify(deflate);

const root = process.cwd();
const port = Number(process.env.PORT || 8000);
const boardApi = new URL(process.env.FPGA_API_URL || "http://192.168.2.99:8081");
const visionApi = new URL(process.env.FPGA_VISION_URL || "http://192.168.2.99:8082");
const plVisionApi = new URL(process.env.FPGA_PL_VISION_URL || visionApi.href);
if (!process.env.FPGA_PL_VISION_URL) plVisionApi.port = "8083";
const localAddress = process.env.FPGA_BIND_ADDRESS || undefined;
const types = { ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".wasm": "application/wasm", ".json": "application/json", ".md": "text/plain; charset=utf-8" };
const server = createServer(async (request, response) => {
  let pathname;
  try {
    pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    if (pathname.includes("\0")) throw new URIError("Invalid path");
  } catch {
    response.writeHead(400); response.end("Invalid path"); return;
  }
  if (pathname === "/api/vision/pc/process" || pathname === "/api/vision/pc/status") {
    try {
      const processing = pathname.endsWith("process");
      if (processing && process.env.FPGA_BENCHMARK_ONLY === "1" &&
          !String(request.headers["x-vision-token"] || "").startsWith("bench-")) {
        response.writeHead(429); response.end(); return;
      }
      if (request.method !== (processing ? "POST" : "GET")) { response.writeHead(405); response.end(); return; }
      let body, frameEncoding = "identity";
      if (processing) {
        const jpegInput = request.headers["content-type"] === "image/jpeg";
        if (!jpegInput && request.headers["content-type"] !== "application/octet-stream") { response.writeHead(400); response.end(); return; }
        const metadata = JSON.parse(request.headers["x-vision-options"] || "null");
        const roi = metadata?.transport_roi;
        let expected = 1228800;
        if (roi !== undefined && roi !== null) {
          if (jpegInput) { response.writeHead(400); response.end(); return; }
          if (Object.keys(roi).sort().join(",") !== "height,width,x,y" ||
              ![roi.x,roi.y,roi.width,roi.height].every(Number.isInteger) ||
              roi.x < 0 || roi.y < 0 || roi.width < 1 || roi.height < 1 ||
              roi.x+roi.width > 640 || roi.y+roi.height > 480) {
            response.writeHead(400); response.end(); return;
          }
          expected = roi.width*roi.height*4;
        }
        const chunks = []; let size = 0;
        for await (const chunk of request) {
          size += chunk.length;
          if (size > 1228800) { response.writeHead(413); response.end(); return; }
          chunks.push(chunk);
        }
        if (jpegInput ? size < 4 : size !== expected) { response.writeHead(400); response.end(); return; }
        body = Buffer.concat(chunks);
        // Lossless transport only: the board still computes gray/Sobel in PL.
        // Retain raw transport for noisy frames that do not compress enough.
        if (jpegInput) frameEncoding = "jpeg";
        else if (process.env.FPGA_FRAME_DEFLATE !== "0" && request.headers['x-vision-transport'] !== 'identity') {
          const compressed = await compressFrame(body, { level: 1 });
          if (compressed.length < body.length * .90) {
            body = compressed; frameEncoding = "deflate";
          }
        }
      }
      const target = new URL(visionApi); target.port = "8084"; target.pathname = pathname;
      const upstream = await requestBoard(target, { method: request.method, body, localAddress,
        timeout: processing ? 30000 : 6500,
        headers: processing ? { "Content-Type": "application/octet-stream", "X-Vision-Token": request.headers["x-vision-token"] || "",
          "X-Vision-Encoding": frameEncoding,
          ...(request.headers["x-vision-options"] ? { "X-Vision-Options": request.headers["x-vision-options"] } : {}) } : {} });
      const headers = { "Content-Type": upstream.headers["content-type"] || "application/json", "Cache-Control": "no-store" };
      for (const key of ["x-vision-backend", "x-vision-source", "x-vision-token", "x-vision-processing-ms", "x-vision-fretboard"])
        if (upstream.headers[key] !== undefined) headers[key] = upstream.headers[key];
      response.writeHead(upstream.status, headers); response.end(upstream.body);
    } catch { response.writeHead(503); response.end('{"error":"PC to PL bridge unavailable"}'); }
    return;
  }
  const isPlVision = pathname === "/api/vision/pl/status" || pathname === "/vision/pl.jpg";
  if (isPlVision || pathname === "/api/vision/status" || pathname === "/vision/frame.jpg" ||
      pathname === "/vision/stream.mjpeg" ||
      pathname === "/api/vision/tracking" || pathname === "/api/vision/tracking/stop" || pathname === "/api/vision/roi") {
    try {
      const isTracking = pathname.startsWith("/api/vision/tracking") || pathname === "/api/vision/roi";
      if ((isTracking && request.method !== "POST") || (!isTracking && request.method !== "GET")) {
        response.writeHead(405); response.end("Method not allowed"); return;
      }
      if (pathname === "/vision/stream.mjpeg") {
        const target = new URL(pathname + new URL(request.url, "http://localhost").search, visionApi);
        const upstream = (target.protocol === "https:" ? httpsRequest : httpRequest)(target, {
          method: "GET", localAddress
        });
        upstream.setTimeout(8000, () => upstream.destroy(new Error("Board stream connect timeout")));
        upstream.once("response", (remote) => {
          upstream.setTimeout(0);
          response.writeHead(remote.statusCode || 502, remote.headers);
          remote.pipe(response);
        });
        upstream.once("error", () => {
          if (!response.headersSent) response.writeHead(503, {
            "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store"
          });
          if (!response.writableEnded) response.end(response.headersSent ? undefined :
            '{"status":"offline","error":"板卡视频流未连接"}');
        });
        response.once("close", () => upstream.destroy());
        upstream.end();
        return;
      }
      let body;
      if (isTracking) {
        const chunks = []; let length = 0;
        for await (const chunk of request) {
          length += chunk.length;
          if (length > 4096) { response.writeHead(413); response.end("Body too large"); return; }
          chunks.push(chunk);
        }
        body = Buffer.concat(chunks);
      }
      const query = new URL(request.url, "http://localhost").search;
      const upstream = await requestBoard(new URL(pathname + query, isPlVision ? plVisionApi : visionApi), {
        method: request.method, body, localAddress,
        ...(isTracking ? { headers: { "Content-Type": "application/json" } } : {})
      });
      const headers = { "Content-Type": upstream.headers["content-type"] || "application/octet-stream", "Cache-Control": "no-store" };
      for (const key of ["X-Vision-Sequence", "X-Vision-Captured-Ms", "X-Vision-Source", "X-Vision-Stream", "X-Vision-Tracking", "X-Vision-Fretboard"]) {
        if (upstream.headers[key.toLowerCase()] !== undefined) headers[key] = upstream.headers[key.toLowerCase()];
      }
      if (isPlVision) for (const key of ["X-Vision-Backend", "X-Vision-Processing-Ms"]) {
        if (upstream.headers[key.toLowerCase()] !== undefined) headers[key] = upstream.headers[key.toLowerCase()];
      }
      response.writeHead(upstream.status, headers); response.end(upstream.body);
    } catch {
      response.writeHead(503, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" });
      response.end(JSON.stringify(isPlVision ?
        { backend: "pl", status: "error", fresh: false, error: "板卡 PL 图像服务未连接" } :
        { status: "offline", error: "板卡视频服务未连接", source: null }));
    }
    return;
  }
  if (pathname === "/api/audio") {
    try {
      const upstream = await requestBoard(new URL("/api/audio", boardApi), { localAddress });
      if (upstream.status < 200 || upstream.status >= 300) throw new Error(`Board HTTP ${upstream.status}`);
      const payload = JSON.parse(upstream.body.toString("utf8"));
      response.writeHead(200, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" });
      response.end(JSON.stringify(payload));
    } catch {
      response.writeHead(200, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" });
      response.end(JSON.stringify({ source: "hardware", status: "hardware_unavailable", stale: true,
        notes: [], error: "板卡音频服务未连接，请检查网线与 PS 服务" }));
    }
    return;
  }
  const relative = pathname === "/" ? "index.html" : pathname.slice(1);
  const target = normalize(join(root, relative));
  if (target !== root && !target.startsWith(root + "/") && !target.startsWith(root + "\\")) { response.writeHead(403); response.end("Forbidden"); return; }
  try { const body = await readFile(target); response.writeHead(200, { "Content-Type": types[extname(target)] || "application/octet-stream", "Cache-Control": "no-store" }); response.end(body); } catch { response.writeHead(404); response.end("Not found"); }
});
server.listen(port, "127.0.0.1", () => console.log(`FretLens web server: http://127.0.0.1:${server.address().port}`));
