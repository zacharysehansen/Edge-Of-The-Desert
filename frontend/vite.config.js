import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const repoRoot = resolve(fileURLToPath(new URL(".", import.meta.url)), "..");

export default defineConfig({
  base: "./",
  server: {
    host: "0.0.0.0",
    port: 8000,
    fs: {
      allow: [repoRoot],
    },
  },
  preview: {
    host: "0.0.0.0",
    port: 8000,
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
