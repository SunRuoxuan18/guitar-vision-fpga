import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";

const { chromium } = createRequire(import.meta.url)(process.env.PLAYWRIGHT_MODULE || "playwright");
const source = await readFile(new URL("vision.js", import.meta.url), "utf8");
const browser = await chromium.launch({ channel: "msedge", headless: true });
try {
  const page = await browser.newPage();
  await page.route("**/vision.js*", route => route.fulfill({ contentType: "text/javascript",
    body: source.replace(/\}\)\(\);\s*$/, `window.publishFeedbackFixture = (reason, synthetic = false) => {
      V.source = synthetic ? "synthetic" : "camera";
      V.result = { positions: [], reason, receivedAt: performance.now() };
      publish(performance.now());
    };\n})();`) }));
  await page.goto(process.argv[2] || "http://127.0.0.1:8001/");
  await page.evaluate(() => {
    document.getElementById("pcPlEnabled").checked = false;
    window.FretAudioSnapshot = () => null;
    window.publishFeedbackFixture("hand_outside_measured_lattice");
  });
  assert.match(await page.locator("#visualEvidence").textContent(), /不在已测品丝范围/);
  await page.evaluate(() => {
    window.feedbackTime = performance.now();
    window.FretAudioSnapshot = () => ({ stale: false, receivedAt: window.feedbackTime,
      age_seconds: 0, ageReportedAt: window.feedbackTime,
      visualFeedback: "手指投影与当前音高不一致，联合判断待确认" });
    window.publishFeedbackFixture(null);
    window.publishFeedbackFixture(null);
  });
  assert.match(await page.locator("#visualEvidence").textContent(), /不一致/);
  await page.evaluate(() => {
    window.feedbackTime = performance.now() - 800;
    window.publishFeedbackFixture("hand_outside_measured_lattice");
  });
  assert.match(await page.locator("#visualEvidence").textContent(), /不在已测品丝范围/);
  for (const invalid of ["stale", "server_age", "future", "nonfinite"]) {
    await page.evaluate(mode => {
      window.FretAudioSnapshot = () => ({ stale: mode === "stale",
        receivedAt: performance.now() + (mode === "future" ? 1000 : -10),
        age_seconds: mode === "server_age" ? 2.5 : mode === "nonfinite" ? NaN : 0,
        visualFeedback: "手指位置与当前音高吻合" });
      window.publishFeedbackFixture("hand_outside_measured_lattice");
    }, invalid);
    assert.match(await page.locator("#visualEvidence").textContent(), /不在已测品丝范围/, invalid);
  }
  await page.evaluate(() => {
    window.feedbackTime = performance.now();
    window.FretAudioSnapshot = () => ({ stale: false, receivedAt: window.feedbackTime,
      visualFeedback: "手指投影与当前音高不一致" });
    window.publishFeedbackFixture(null, true);
  });
  assert.match(await page.locator("#visualEvidence").textContent(), /SYNTHETIC/);
  assert.doesNotMatch(await page.locator("#visualEvidence").textContent(), /不一致/);
  console.log("PASS: board feedback survives video updates; expired and synthetic results stay separate");
} finally {
  await browser.close();
}
