/**
 * Entry point: mounts the React app into <div id="root"> in index.html.
 *
 * StrictMode runs extra checks in development (e.g. renders components twice to
 * surface side-effect bugs). It has no effect in the production build.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
