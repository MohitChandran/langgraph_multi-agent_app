/**
 * Entry point: mounts the React app into <div id="root"> in index.html.
 *
 * BrowserRouter keeps the URL in sync with the app (/c/<chat id>), so reloads, bookmarks
 * and the Back button work. The Vite dev server answers every unknown path with
 * index.html, so a reload on /c/<id> loads the app, which then reads the id from the URL.
 *
 * StrictMode runs extra checks in development (e.g. renders components twice to
 * surface side-effect bugs). It has no effect in the production build.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";
import "./index.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
