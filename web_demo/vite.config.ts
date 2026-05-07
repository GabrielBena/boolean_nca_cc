import { defineConfig } from "vite";
import { resolve } from "node:path";

/**
 * Multi-page setup:
 *   - ``/``           index.html   live interactive demo (the article embed target)
 *   - ``/verify.html`` verify.html  TS↔JAX parity smoke test (dev tool)
 *
 * Vite picks both up automatically in dev (any HTML at the root is served);
 * for production builds we have to enumerate them in ``rollupOptions.input``.
 */
export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        verify: resolve(__dirname, "verify.html"),
      },
    },
  },
});
