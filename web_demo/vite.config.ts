import { defineConfig } from "vite";
import { resolve } from "node:path";

/**
 * Multi-page setup:
 *   - ``/``            index.html   live interactive demo (article embed target)
 *   - ``/verify.html`` verify.html  TS↔JAX parity smoke test (dev tool only)
 *
 * Production base is ``/assets/sodc-demo/`` so that when the bundle is copied
 * to ``gabrielbena.github.io/assets/sodc-demo/`` all asset references resolve
 * correctly.  ``import.meta.env.BASE_URL`` in TS source picks this up
 * automatically at build time (``"/"`` in dev, ``"/assets/sodc-demo/"`` in prod).
 *
 * Filenames are kept deterministic (no content hash) so the HTML embed tag in
 * the blog post never needs updating after a rebuild.
 */
export default defineConfig({
  base: "/assets/sodc-demo/",
  build: {
    outDir: "dist",
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        verify: resolve(__dirname, "verify.html"),
      },
      output: {
        // Strip the content hash so the embed <script src="…/main.js"> is stable.
        entryFileNames: "[name].js",
        chunkFileNames: "[name].js",
        assetFileNames: "[name][extname]",
      },
    },
  },
});
