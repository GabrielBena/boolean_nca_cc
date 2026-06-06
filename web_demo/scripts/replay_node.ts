/**
 * Node-side smoke runner for the TS port.
 *
 * Reads ``public/weights/<task>_random_damage.json`` and
 * ``public/weights/<task>_trajectory.json`` from disk, hydrates the
 * TS implementation, and runs the replay parity test — same code path
 * the browser exercises in ``main.ts``, just without ``fetch``.
 *
 * Run with::
 *
 *     node --experimental-strip-types scripts/replay_node.ts
 *
 * (Node 22 LTS has native ``.ts`` support via strip-types; no transpilation
 * needed for our straight ESM modules.)
 */

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

import { parseWeightsFromJson } from "../src/weights.ts";
import { runReplayFromJson } from "../src/replay.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
// Optional argv overrides (resolved against this script's dir):
//   node --experimental-strip-types scripts/replay_node.ts [weights.json] [trajectory.json]
const WEIGHTS_PATH = resolve(
  HERE,
  process.argv[2] ?? "../public/weights/reverse_random_damage.json",
);
const TRAJECTORY_PATH = resolve(
  HERE,
  process.argv[3] ?? "../public/weights/reverse_trajectory.json",
);

async function main() {
  console.log(`[ts-replay] reading ${WEIGHTS_PATH}`);
  const weightsRaw = JSON.parse(await readFile(WEIGHTS_PATH, "utf8"));
  const weights = parseWeightsFromJson(weightsRaw);
  console.log(
    `[ts-replay] parsed weights: ${weights.header.modelKind}, ` +
      `D=${weights.header.attentionDim}, H=${weights.header.numHeads}, ` +
      `dtype=${weights.header.tensorDtype}`,
  );

  console.log(`[ts-replay] reading ${TRAJECTORY_PATH}`);
  const trajRaw = JSON.parse(await readFile(TRAJECTORY_PATH, "utf8"));
  const t0 = performance.now();
  const result = runReplayFromJson(weights, trajRaw);
  const dt = performance.now() - t0;

  console.log(`[ts-replay] replay ran in ${dt.toFixed(0)} ms over ${result.nTicks} ticks`);
  console.log(
    [
      "",
      `  step  hard_acc(jax)  hard_acc(ts)        Δ      Δ logits      Δ hidden       Δ loss`,
      `  ----  -------------  -------------  -------  ------------  ------------  -----------`,
      ...result.perTick.map(
        (r) =>
          `  ${String(r.step).padStart(4)}  ${r.hardAccJax.toFixed(4).padStart(13)}  ` +
          `${r.hardAccTs.toFixed(4).padStart(13)}  ${r.hardAccDelta.toFixed(4).padStart(7)}  ` +
          `${r.maxAbsLogitsDelta.toExponential(2).padStart(12)}  ` +
          `${r.maxAbsHiddenDelta.toExponential(2).padStart(12)}  ` +
          `${r.maxAbsLossDelta.toExponential(2).padStart(11)}`,
      ),
      "",
    ].join("\n"),
  );

  console.log(`[ts-replay] ${result.message}`);
  process.exit(result.pass ? 0 : 1);
}

void main().catch((err) => {
  console.error("[ts-replay] FAILED with exception:", err);
  process.exit(1);
});
