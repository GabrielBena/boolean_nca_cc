#!/usr/bin/env node
// Copy the built demo into the GitHub-Pages repo's assets/sodc-demo/.
// Robust to where that repo lives:
//   1. $SODC_DEPLOY_DEST  — explicit destination dir (wins if set)
//   2. else auto-find the `gabrielbena.github.io` checkout among home-based
//      candidates (survives the repo moving / being on a different machine).
// Fails loudly with instructions if none is found — never silently writes the
// wrong place. Run AFTER `npm run build` (package.json `deploy` chains them).
import { execSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const home = os.homedir();
const explicit = process.env.SODC_DEPLOY_DEST;

let dest;
if (explicit) {
  dest = path.resolve(explicit);
} else {
  const candidates = [
    path.join(home, "code/writing/gabrielbena.github.io"),
    path.join(home, "code/PhD/gabrielbena.github.io"),
    path.join(home, "code/gabrielbena.github.io"),
  ];
  const root = candidates.find((p) => fs.existsSync(p));
  if (!root) {
    console.error(
      "[deploy] Could not find the gabrielbena.github.io Pages repo.\n" +
        "  Looked in:\n    " + candidates.join("\n    ") + "\n" +
        "  Fix: set SODC_DEPLOY_DEST=/abs/path/to/gabrielbena.github.io/assets/sodc-demo",
    );
    process.exit(1);
  }
  dest = path.join(root, "assets", "sodc-demo");
}

if (!fs.existsSync("dist")) {
  console.error("[deploy] no dist/ — run `npm run build` first.");
  process.exit(1);
}

fs.rmSync(dest, { recursive: true, force: true });
fs.mkdirSync(dest, { recursive: true });
for (const f of fs.readdirSync("dist").filter((f) => f.endsWith(".js"))) {
  fs.copyFileSync(path.join("dist", f), path.join(dest, f));
}
fs.cpSync("public/weights", path.join(dest, "weights"), { recursive: true });

// Helpful: show what the Pages repo now sees so the user can commit it there.
const repo = path.dirname(path.dirname(dest));
console.log(`[deploy] copied dist/*.js + public/weights → ${dest}`);
try {
  const status = execSync("git status --short assets/sodc-demo", { cwd: repo }).toString().trim();
  console.log(`[deploy] in ${repo}:\n${status || "  (no changes)"}\n` +
    "[deploy] commit + push there to publish.");
} catch {
  /* repo may not be a git checkout; copy still done */
}
