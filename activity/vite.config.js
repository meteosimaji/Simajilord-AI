import { defineConfig } from "vite";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  base: "/",
  build: {
    outDir: fileURLToPath(
      new URL("../src/simajilord/activity/static", import.meta.url),
    ),
    emptyOutDir: true,
    sourcemap: false,
  },
});
