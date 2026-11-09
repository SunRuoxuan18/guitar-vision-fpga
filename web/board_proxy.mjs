import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";

export function requestBoard(url, { method = "GET", body, headers = {}, localAddress, timeout = 2500 } = {}) {
  return new Promise((resolve, reject) => {
    const request = (url.protocol === "https:" ? httpsRequest : httpRequest)(url, {
      method, localAddress, signal: AbortSignal.timeout(timeout),
      // Four-tip metadata includes a bounded 208px crop preview. Combined
      // tracking/preview headers can exceed Node's default 16 KiB limit.
      maxHeaderSize: 128 * 1024,
      headers: body === undefined ? headers : { ...headers, "Content-Length": body.length }
    }, response => {
      const chunks = [];
      response.on("data", chunk => chunks.push(chunk));
      response.once("error", reject);
      response.once("end", () => resolve({
        status: response.statusCode, headers: response.headers, body: Buffer.concat(chunks)
      }));
    });
    request.on("socket", socket => socket.setNoDelay(true));
    request.once("error", reject);
    request.end(body);
  });
}
