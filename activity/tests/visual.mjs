import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer } from "vite";

const outputDirectory = new URL("../test-results/", import.meta.url);
const server = await createServer({
  configFile: fileURLToPath(new URL("../vite.config.js", import.meta.url)),
  server: {
    host: "127.0.0.1",
    port: 0,
    strictPort: false,
  },
});

let browser;
try {
  await server.listen();
  const address = server.httpServer?.address();
  if (!address || typeof address === "string") {
    throw new Error("Vite did not expose a local TCP port.");
  }
  browser = await chromium.launch({ headless: true });
  await mkdir(outputDirectory, { recursive: true });

  for (const scenario of [
    { name: "desktop", width: 1280, height: 800 },
    { name: "mobile", width: 390, height: 844 },
  ]) {
    const page = await browser.newPage({
      viewport: { width: scenario.width, height: scenario.height },
      deviceScaleFactor: 1,
    });
    const errors = [];
    page.on("console", (message) => {
      if (message.type() === "error" || message.type() === "warning") {
        errors.push(`${message.type()}: ${message.text()}`);
      }
    });
    page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));

    await page.goto(
      `http://127.0.0.1:${address.port}/?preview=1`,
      { waitUntil: "networkidle" },
    );
    await page.evaluate(() => document.fonts.ready);
    const before = await page.locator("#elapsed").textContent();
    await page.waitForTimeout(1_100);
    const after = await page.locator("#elapsed").textContent();
    const geometry = await page.evaluate(() => ({
      overflow: document.documentElement.scrollWidth > window.innerWidth,
      fatalHidden: document.querySelector("#fatal")?.hidden === true,
      title: document.querySelector("#title")?.textContent,
      nextCount: document.querySelectorAll("#up-next > li").length,
    }));

    if (errors.length) {
      throw new Error(`${scenario.name} emitted console errors:\n${errors.join("\n")}`);
    }
    if (geometry.overflow) {
      throw new Error(`${scenario.name} has horizontal overflow.`);
    }
    if (!geometry.fatalHidden) {
      throw new Error(`${scenario.name} displayed the fatal-error panel.`);
    }
    if (geometry.title !== "Primary Colors" || geometry.nextCount !== 3) {
      throw new Error(`${scenario.name} did not render the complete fixture.`);
    }
    if (before === after) {
      throw new Error(`${scenario.name} progress did not advance client-side.`);
    }
    await page.screenshot({
      path: fileURLToPath(
        new URL(`activity-${scenario.name}.png`, outputDirectory),
      ),
      fullPage: true,
    });
    await page.close();
  }
} finally {
  await browser?.close();
  await server.close();
}
