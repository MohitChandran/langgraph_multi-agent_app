/**
 * Vite config: the dev server and build tool for the React app.
 *
 * The proxy forwards every request starting with /api to the FastAPI server.
 * The browser only ever talks to localhost:5173 (same origin), so we don't need
 * CORS headers on the API, and the frontend code can use relative URLs like "/api/db/ask".
 */
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
