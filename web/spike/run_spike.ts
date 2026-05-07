/**
 * Phase 0 jax-js sanity spike.
 *
 * Exercises the primitives listed under each SPEC.md section's "jax-js op deps".
 * Goal: a clear pass/fail per section so we can decide go/no-go on the jax-js path.
 *
 * Reference-counting note: jax-js uses manual refcounting (no JS destructors).
 * Every reused Array must be passed as `x.ref` for all but its final use.
 *
 * Run: pnpm spike
 */

import * as jax from "@jax-js/jax";
import * as optax from "@jax-js/optax";

const { numpy: np, nn, random, grad, jit } = jax;
type Arr = jax.Array;

type Result = {
  section: string;
  status: "pass" | "fail";
  detail: string;
  ops: string[];
};

const results: Result[] = [];

const record = (section: string, status: "pass" | "fail", detail: string, ops: string[]) =>
  results.push({ section, status, detail, ops });

// helper: get scalar from a 0-d Array
const scalar = (a: Arr): number => Number(a.js());

// ---------------------------------------------------------------------------
// §CIRCUIT-FORWARD primitives — tiny circuit, 1 layer, arity=2, 2 gates, 4 cases
// ---------------------------------------------------------------------------
async function testCircuitForward() {
  const opsExercised: string[] = [];
  try {
    const x = np.array(
      [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ],
      { dtype: jax.DType.f32 },
    );
    const wires = np.array([[0], [1]], { dtype: jax.DType.i32 });
    const logits = np.array([[[1, -1, -1, 1], [-1, -1, -1, 1]]], { dtype: jax.DType.f32 });

    const sig = nn.sigmoid(logits);
    opsExercised.push("nn.sigmoid");

    const evenIdx = np.array([0, 2], { dtype: jax.DType.i32 });
    const oddIdx = np.array([1, 3], { dtype: jax.DType.i32 });
    const lutsEven = np.take(sig.ref, evenIdx, -1);
    const lutsOdd = np.take(sig, oddIdx, -1);
    opsExercised.push("np.take(axis=-1)");

    const wires0 = np.take(wires.ref, np.array([0], { dtype: jax.DType.i32 }), 0);
    const w0Flat = np.reshape(wires0, [1]);
    const x0 = np.take(x.ref, w0Flat, 1);
    opsExercised.push("np.reshape", "np.take(axis=1)");

    const x0_b = np.expandDims(x0, -1);
    opsExercised.push("np.expandDims");

    const one = np.array(1.0, { dtype: jax.DType.f32 });
    const reduced1 = np.add(
      np.multiply(np.subtract(one.ref, x0_b.ref), lutsEven),
      np.multiply(x0_b, lutsOdd),
    );
    opsExercised.push("np.add", "np.multiply", "np.subtract");

    if (reduced1.shape.join(",") !== "4,2,2") {
      throw new Error(`expected (4,2,2), got (${reduced1.shape.join(",")})`);
    }

    const evenIdx2 = np.array([0], { dtype: jax.DType.i32 });
    const oddIdx2 = np.array([1], { dtype: jax.DType.i32 });
    const lutsEven2 = np.take(reduced1.ref, evenIdx2, -1);
    const lutsOdd2 = np.take(reduced1, oddIdx2, -1);

    const wires1 = np.take(wires, np.array([1], { dtype: jax.DType.i32 }), 0);
    const w1Flat = np.reshape(wires1, [1]);
    const x1 = np.take(x, w1Flat, 1);
    const x1_b = np.expandDims(x1, -1);
    const reduced2 = np.add(
      np.multiply(np.subtract(one, x1_b.ref), lutsEven2),
      np.multiply(x1_b, lutsOdd2),
    );

    const out = np.reshape(reduced2, [4, 2]);
    if (out.shape.join(",") !== "4,2") {
      throw new Error(`expected (4,2), got (${out.shape.join(",")})`);
    }

    const minVal = scalar(np.min(out.ref));
    const maxVal = scalar(np.max(out));
    opsExercised.push("np.min", "np.max");
    if (minVal < -1e-6 || maxVal > 1 + 1e-6) {
      throw new Error(`output not in [0,1]: min=${minVal}, max=${maxVal}`);
    }

    record(
      "§CIRCUIT-FORWARD",
      "pass",
      `forward pass produced shape (4,2) values in [${minVal.toFixed(3)}, ${maxVal.toFixed(3)}]`,
      opsExercised,
    );
  } catch (e) {
    record("§CIRCUIT-FORWARD", "fail", String(e), opsExercised);
  }
}

// ---------------------------------------------------------------------------
// §CIRCUIT-FORWARD grad — confirm grad flows through the float-only primitives.
// We deliberately exercise grad on a function that has the same float ops as
// the LUT reduction (sigmoid + arithmetic + reshape + reduce) WITHOUT integer
// gathers. Reason: jax-js's grad tracing currently loses int-dtype info on
// intermediate take indices computed from a non-grad input (e.g., wires
// gathered into w0). The actual circuit port (Phase 1) will hoist all integer
// intermediates outside the grad-d function — confirmed working pattern.
// ---------------------------------------------------------------------------
// Two grad tests so we can isolate whether the issue is grad-through-take
// specifically or something more fundamental.

async function testGradFloatOnly() {
  // Pure-float pipeline: sigmoid + arithmetic + reshape + reduce. No takes.
  const opsExercised: string[] = [];
  try {
    const target = np.array([0.0, 1.0, 1.0, 0.0], { dtype: jax.DType.f32 });
    const xCol0 = np.array([0, 0, 1, 1], { dtype: jax.DType.f32 });
    const xCol1 = np.array([0, 1, 0, 1], { dtype: jax.DType.f32 });
    const one = np.array(1.0, { dtype: jax.DType.f32 });

    // Direct LUT-arity-2 evaluation without take/slicing — pure float ops.
    // For each of 4 cases (x0, x1), output = sig[0]*(1-x0)*(1-x1) + sig[1]*(1-x0)*x1 + sig[2]*x0*(1-x1) + sig[3]*x0*x1.
    // We reshape sig (shape (4,)) and broadcast against case-major masks.
    const forward = (logits: Arr) => {
      const sig = nn.sigmoid(logits); // (4,)
      const xz = np.subtract(one.ref, xCol0.ref); // (4,)
      const xo = xCol0.ref; // (4,)
      const yz = np.subtract(one.ref, xCol1.ref); // (4,)
      const yo = xCol1.ref;

      // m[i] for i in 0..3 — 4 mask vectors, each shape (4,).
      const m0 = np.multiply(xz.ref, yz.ref); // (1-x0)*(1-x1)
      const m1 = np.multiply(xz, yo.ref); // (1-x0)*x1
      const m2 = np.multiply(xo.ref, yz); // x0*(1-x1)
      const m3 = np.multiply(xo, yo); // x0*x1

      const s0 = np.take(sig.ref, np.array(0, { dtype: jax.DType.i32 }));
      const s1 = np.take(sig.ref, np.array(1, { dtype: jax.DType.i32 }));
      const s2 = np.take(sig.ref, np.array(2, { dtype: jax.DType.i32 }));
      const s3 = np.take(sig, np.array(3, { dtype: jax.DType.i32 }));

      const pred = np.add(
        np.add(np.multiply(s0, m0), np.multiply(s1, m1)),
        np.add(np.multiply(s2, m2), np.multiply(s3, m3)),
      ); // (4,)
      const diff = np.subtract(pred, target.ref);
      return np.mean(np.power(np.abs(diff), 4));
    };

    const logits = np.array([2.0, -2.0, -2.0, 2.0], { dtype: jax.DType.f32 });

    const lossVal = forward(logits.ref);
    const lossNum = scalar(lossVal);
    opsExercised.push("nn.sigmoid", "np.take(scalar)", "np.add", "np.multiply", "np.subtract", "np.mean", "np.power", "np.abs");

    const gradFn = grad(forward);
    opsExercised.push("grad");
    const g = gradFn(logits);

    if (g.shape.join(",") !== "4") {
      throw new Error(`grad shape ${g.shape} != logits shape 4`);
    }

    const gMin = scalar(np.min(g.ref));
    const gMax = scalar(np.max(g));
    if (!isFinite(gMin) || !isFinite(gMax)) {
      throw new Error(`non-finite grad: min=${gMin}, max=${gMax}`);
    }

    record(
      "grad (float-only pipeline)",
      "pass",
      `loss=${lossNum.toFixed(4)}, grad shape ${g.shape.join("×")}, range [${gMin.toFixed(4)}, ${gMax.toFixed(4)}]`,
      opsExercised,
    );
  } catch (e) {
    record("grad (float-only pipeline)", "fail", String(e), opsExercised);
  }
}

async function testGradWithIntTake() {
  // Same forward, but using np.take with array (1-d) indices instead of scalar.
  // This is the "stride 2" pattern needed for the actual circuit forward port.
  // Currently expected to FAIL — documents jax-js's grad+take limitation.
  const opsExercised: string[] = [];
  try {
    const target = np.array([0.0, 1.0, 1.0, 0.0], { dtype: jax.DType.f32 });
    const evenIdx = np.array([0, 2], { dtype: jax.DType.i32 });
    const oddIdx = np.array([1, 3], { dtype: jax.DType.i32 });

    const forward = (logits: Arr) => {
      const sig = nn.sigmoid(logits); // (4,)
      const evens = np.take(sig.ref, evenIdx.ref, 0); // (2,)
      const odds = np.take(sig, oddIdx.ref, 0); // (2,)
      const sum2 = np.add(evens, odds); // (2,)
      const meanVal = np.mean(sum2);
      const tgt = np.mean(target.ref);
      return np.power(np.subtract(meanVal, tgt), 2);
    };

    const logits = np.array([1.0, 2.0, 3.0, 4.0], { dtype: jax.DType.f32 });
    const gradFn = grad(forward);
    opsExercised.push("grad", "np.take(array indices, axis)", "nn.sigmoid");
    const g = gradFn(logits);

    if (g.shape.join(",") !== "4") {
      throw new Error(`grad shape ${g.shape} != logits shape 4`);
    }
    const gMax = scalar(np.max(np.abs(g)));
    if (!isFinite(gMax)) throw new Error(`non-finite grad: max=${gMax}`);

    record("grad (with take[i32 array indices])", "pass", `grad shape ${g.shape.join("×")}, |max|=${gMax.toFixed(4)}`, opsExercised);
  } catch (e) {
    record("grad (with take[i32 array indices])", "fail", String(e), opsExercised);
  }
}

// ---------------------------------------------------------------------------
// §LOSS-L4 / §LOSS-BCE primitives
// ---------------------------------------------------------------------------
async function testLossPrimitives() {
  const opsExercised: string[] = [];
  try {
    const pred = np.array([0.1, 0.9, 0.8, 0.2], { dtype: jax.DType.f32 });
    const y = np.array([0, 1, 1, 0], { dtype: jax.DType.f32 });

    const res = np.subtract(pred.ref, y.ref);
    const l4 = np.mean(np.power(np.abs(res), 4));
    opsExercised.push("np.subtract", "np.abs", "np.power", "np.mean");

    // BCE form: y*softplus(-z) + (1-y)*softplus(z), z = logit(pred)
    const z = jax.scipySpecial.logit(pred.ref);
    opsExercised.push("scipySpecial.logit");
    const sp = nn.softplus;
    const one = np.array(1.0, { dtype: jax.DType.f32 });
    const bce = np.mean(
      np.add(
        np.multiply(y.ref, sp(np.negative(z.ref))),
        np.multiply(np.subtract(one, y.ref), sp(z)),
      ),
    );
    opsExercised.push("nn.softplus", "np.negative");

    const predRound = np.round(pred);
    const correct = np.equal(predRound, y);
    const acc = np.mean(np.astype(correct, jax.DType.f32));
    opsExercised.push("np.round", "np.equal", "np.astype");

    record(
      "§LOSS-L4 / §LOSS-BCE",
      "pass",
      `L4=${scalar(l4).toFixed(4)}, BCE=${scalar(bce).toFixed(4)}, acc=${scalar(acc).toFixed(2)}`,
      opsExercised,
    );
  } catch (e) {
    record("§LOSS-L4 / §LOSS-BCE", "fail", String(e), opsExercised);
  }
}

// ---------------------------------------------------------------------------
// §ADAMW — single optimizer step via @jax-js/optax
// ---------------------------------------------------------------------------
async function testOptaxOptimizer(
  name: string,
  makeOpt: () => optax.GradientTransformation,
  passesParams: boolean,
) {
  const opsExercised: string[] = [];
  try {
    // No .js() before the optimizer — jax-js .js() appears to consume.
    // We hard-code "before" since the test inputs are constants.
    const before = [1.0, -2.0, 0.5];
    let params = np.array([1.0, -2.0, 0.5], { dtype: jax.DType.f32 });
    const grads = np.array([0.1, -0.2, 0.05], { dtype: jax.DType.f32 });

    const opt = makeOpt();
    opsExercised.push(`optax.${name}`);

    let state = opt.init(params.ref);
    const [updates, _newState] = passesParams
      ? opt.update(grads, state, params.ref)
      : opt.update(grads, state);
    state = _newState;

    params = optax.applyUpdates(params, updates);
    opsExercised.push("opt.init", "opt.update", "optax.applyUpdates");

    if (params.shape.join(",") !== "3") {
      throw new Error(`shape mismatch after update: ${params.shape}`);
    }

    const after = params.js() as number[];
    const delta = before.map((b, i) => after[i] - b);

    record(
      `optax.${name}`,
      "pass",
      `params Δ=[${delta.map((x) => x.toFixed(4)).join(",")}]`,
      opsExercised,
    );
  } catch (e) {
    record(`optax.${name}`, "fail", String(e), opsExercised);
  }
}

// Hand-rolled AdamW one step — fallback pattern for the actual port.
async function testHandRolledAdamW() {
  const opsExercised: string[] = [];
  try {
    let params = np.array([1.0, -2.0, 0.5], { dtype: jax.DType.f32 });
    const grads = np.array([0.1, -0.2, 0.05], { dtype: jax.DType.f32 });
    let m = np.zeros(params.shape, { dtype: jax.DType.f32 });
    let v = np.zeros(params.shape, { dtype: jax.DType.f32 });

    const lr = 1.0,
      b1 = 0.8,
      b2 = 0.8,
      wd = 0.1,
      eps = 1e-8;
    const t = 1;

    const before = [1.0, -2.0, 0.5];

    const b1Arr = np.array(b1, { dtype: jax.DType.f32 });
    const b2Arr = np.array(b2, { dtype: jax.DType.f32 });
    const oneMinusB1 = np.array(1 - b1, { dtype: jax.DType.f32 });
    const oneMinusB2 = np.array(1 - b2, { dtype: jax.DType.f32 });
    const epsArr = np.array(eps, { dtype: jax.DType.f32 });
    const lrArr = np.array(lr, { dtype: jax.DType.f32 });
    const wdArr = np.array(lr * wd, { dtype: jax.DType.f32 });
    const bc1 = np.array(1 - Math.pow(b1, t), { dtype: jax.DType.f32 });
    const bc2 = np.array(1 - Math.pow(b2, t), { dtype: jax.DType.f32 });

    // m = b1*m + (1-b1)*g
    m = np.add(np.multiply(b1Arr.ref, m), np.multiply(oneMinusB1.ref, grads.ref));
    // v = b2*v + (1-b2)*g^2
    v = np.add(np.multiply(b2Arr.ref, v), np.multiply(oneMinusB2.ref, np.power(grads, 2)));
    // m_hat = m / (1 - b1^t)
    const mHat = np.divide(m.ref, bc1);
    const vHat = np.divide(v.ref, bc2);
    const denom = np.add(np.sqrt(vHat), epsArr);
    const adamUpdate = np.multiply(np.negative(lrArr.ref), np.divide(mHat, denom));
    const wdUpdate = np.multiply(np.negative(wdArr), params.ref);
    params = np.add(np.add(params, adamUpdate), wdUpdate);
    opsExercised.push("np.add", "np.multiply", "np.divide", "np.sqrt", "np.power", "np.negative");

    const after = params.js() as number[];
    const delta = before.map((b, i) => after[i] - b);

    record(
      "hand-rolled AdamW (1 step)",
      "pass",
      `params Δ=[${delta.map((x) => x.toFixed(4)).join(",")}]`,
      opsExercised,
    );
  } catch (e) {
    record("hand-rolled AdamW (1 step)", "fail", String(e), opsExercised);
  }
}

// ---------------------------------------------------------------------------
// §DAMAGE — random shuffle + boolean mask + where
// ---------------------------------------------------------------------------
async function testDamage() {
  const opsExercised: string[] = [];
  try {
    const totalNodes = 8;
    const nKnock = 3;

    const key = random.key(42);
    opsExercised.push("random.key");
    const u = random.uniform(key, [totalNodes]);
    opsExercised.push("random.uniform");
    const order = np.argsort(u);
    opsExercised.push("np.argsort");

    const knockIndices = np.take(order, np.arange(nKnock), 0);
    opsExercised.push("np.arange");

    // Pattern: True where index appears in knockIndices.
    // Use broadcasting: idxRange[:, None] == knockIndices[None, :]  -> (8, 3)
    // Then any along axis -1 -> (8,)
    const idxRange = np.arange(totalNodes);
    const idxRangeExp = np.expandDims(idxRange, -1);
    const knockExp = np.expandDims(knockIndices, 0);
    const eqMatrix = np.equal(idxRangeExp, knockExp);
    const pattern = np.any(eqMatrix, -1);
    opsExercised.push("np.equal (broadcast)", "np.any");

    const zero = np.array(0.0, { dtype: jax.DType.f32 });
    const one = np.array(1.0, { dtype: jax.DType.f32 });
    const mask = np.where(pattern.ref, zero, one);
    opsExercised.push("np.where");

    const patternJs = pattern.js() as boolean[] | number[] | Uint8Array;
    const knocked = Array.from(patternJs as Iterable<unknown>).filter((b) => Boolean(b)).length;
    if (knocked !== nKnock) throw new Error(`expected ${nKnock} knocked, got ${knocked} (pattern=${JSON.stringify(patternJs)})`);

    const maskJs = mask.js() as number[];
    record(
      "§DAMAGE",
      "pass",
      `pattern=[${Array.from(patternJs as Iterable<unknown>).map((b) => (b ? "1" : "0")).join(",")}], mask=[${(maskJs as number[]).map((v) => v.toFixed(0)).join(",")}]`,
      opsExercised,
    );
  } catch (e) {
    record("§DAMAGE", "fail", String(e), opsExercised);
  }
}

// ---------------------------------------------------------------------------
// §SA-FORWARD primitives — multi-head attention, softmax, layernorm, gelu
// ---------------------------------------------------------------------------
async function testSaPrimitives() {
  const opsExercised: string[] = [];
  try {
    const nNodes = 4;
    const nHeads = 2;
    const headDim = 8;

    const key = random.key(0);
    // random.split returns a single Array with leading axis [num, ...].
    const keys = random.split(key, 3);
    opsExercised.push("random.split");

    const k0 = np.take(keys.ref, np.array(0, { dtype: jax.DType.i32 }), 0);
    const k1 = np.take(keys.ref, np.array(1, { dtype: jax.DType.i32 }), 0);
    const k2 = np.take(keys, np.array(2, { dtype: jax.DType.i32 }), 0);

    const q = random.normal(k0, [1, nNodes, nHeads, headDim]);
    const k = random.normal(k1, [1, nNodes, nHeads, headDim]);
    const v = random.normal(k2, [1, nNodes, nHeads, headDim]);
    opsExercised.push("random.normal");

    const attn = nn.dotProductAttention(q, k, v);
    opsExercised.push("nn.dotProductAttention");
    if (attn.shape.join(",") !== `1,${nNodes},${nHeads},${headDim}`) {
      throw new Error(`bad attention shape: ${attn.shape}`);
    }

    const xKey = random.key(1);
    const x = random.normal(xKey, [4, 16]);
    const normed = nn.standardize(x.ref, -1);
    opsExercised.push("nn.standardize");
    if (normed.shape.join(",") !== "4,16") throw new Error(`standardize shape mismatch`);

    const g = nn.gelu(x.ref);
    opsExercised.push("nn.gelu");
    if (g.shape.join(",") !== "4,16") throw new Error(`gelu shape mismatch`);

    const sm = nn.softmax(x, -1);
    opsExercised.push("nn.softmax");
    // Verify a single row sums to ~1.
    const row0 = np.take(sm, np.array(0, { dtype: jax.DType.i32 }), 0);
    const rowSum = scalar(np.sum(row0));
    if (Math.abs(rowSum - 1.0) > 1e-3) throw new Error(`softmax rows don't sum to 1: ${rowSum}`);

    record(
      "§SA-FORWARD primitives",
      "pass",
      `attention out ${attn.shape.join("×")}, standardize ok, gelu ok, softmax row sum ${rowSum.toFixed(4)}`,
      opsExercised,
    );
  } catch (e) {
    record("§SA-FORWARD primitives", "fail", String(e), opsExercised);
  }
}

// ---------------------------------------------------------------------------
// Performance — repeated jit'd matmul timing
// ---------------------------------------------------------------------------
async function testPerf() {
  try {
    const a0 = random.normal(random.key(7), [256, 256]);
    const b = random.normal(random.key(8), [256, 256]);

    const matmulFn = jit((x: Arr, y: Arr) => np.matmul(x, y));
    // warmup (compile)
    let out = matmulFn(a0, b.ref);
    await jax.blockUntilReady(out);

    const t0 = performance.now();
    const N = 20;
    for (let i = 0; i < N; i++) {
      out = matmulFn(out, b.ref);
    }
    await jax.blockUntilReady(out);
    const t1 = performance.now();
    b.dispose();

    const msPerCall = (t1 - t0) / N;
    record(
      "perf (256×256 matmul, jit)",
      "pass",
      `${msPerCall.toFixed(2)} ms/call (${N} iterations)`,
      ["jit", "np.matmul", "blockUntilReady"],
    );
  } catch (e) {
    record("perf (256×256 matmul, jit)", "fail", String(e), []);
  }
}

// ---------------------------------------------------------------------------
async function main() {
  console.log("== jax-js spike — initializing ==");
  const t0 = performance.now();
  const devices = await jax.init();
  console.log(`init ok in ${(performance.now() - t0).toFixed(0)}ms — devices: ${devices.map((d) => d.toString()).join(", ")}`);

  await testCircuitForward();
  await testGradFloatOnly();
  await testGradWithIntTake();
  await testLossPrimitives();
  await testOptaxOptimizer("sgd(0.1)", () => optax.sgd(0.1), false);
  await testOptaxOptimizer("adam(1.0, b1=b2=0.8)", () => optax.adam(1.0, { b1: 0.8, b2: 0.8, eps: 1e-8 }), false);
  await testOptaxOptimizer("adamw(1.0, b1=b2=0.8, wd=0.1)", () => optax.adamw(1.0, { b1: 0.8, b2: 0.8, weightDecay: 0.1, eps: 1e-8 }), true);
  await testHandRolledAdamW();
  await testDamage();
  await testSaPrimitives();
  await testPerf();

  console.log("");
  console.log("== Spike Results ==");
  let pass = 0;
  let fail = 0;
  for (const r of results) {
    const tag = r.status === "pass" ? "PASS" : "FAIL";
    console.log(`[${tag}] ${r.section}`);
    console.log(`       ${r.detail}`);
    if (r.ops.length) console.log(`       ops: ${r.ops.join(", ")}`);
    if (r.status === "pass") pass++;
    else fail++;
  }
  console.log("");
  console.log(`Summary: ${pass} pass, ${fail} fail`);

  const fs = await import("node:fs/promises");
  await fs.mkdir(new URL("./results", import.meta.url), { recursive: true });
  await fs.writeFile(
    new URL("./results/spike_results.json", import.meta.url),
    JSON.stringify({ devices: devices.map((d) => d.toString()), results, pass, fail }, null, 2),
  );

  if (fail > 0) process.exit(1);
}

main().catch((e) => {
  console.error("FATAL:", e);
  process.exit(2);
});
