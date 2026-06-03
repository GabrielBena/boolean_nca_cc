/**
 * Demo state controller.
 *
 * One ``Controller`` owns the entire mutable side of the live demo:
 * weights (immutable, loaded once), topology (rebuilt only on Shuffle Wires),
 * state (mutated by the TMT every tick), scratch buffers, rng, step counter,
 * and the static x / y task data.
 *
 * The four user actions all funnel through methods here, each returning a
 * fresh snapshot. The UI layer (``main.ts``) doesn't touch the math
 * primitives directly — it just calls ``ctrl.tick()`` /
 * ``ctrl.reset()`` / ``ctrl.shuffle()`` / ``ctrl.shotgun()`` /
 * ``ctrl.damage(idx)`` and consumes ``Snapshot`` objects.
 */

import {
  type CircuitState,
  type LayerSize,
  type Topology,
  applyShotgun,
  buildTopology,
  circuitForward,
  damageGate,
  genWires,
  initState,
  makeRng,
  resetCircuit,
  shuffleWires,
  zeroHidden,
  type Rng,
} from "./circuit";
import {
  type Scratch,
  computeResiduals,
  makeScratch,
  tick as runTick,
} from "./tmt";
import type { TMTWeights } from "./weights";

export interface ControllerSnapshot {
  step: number;
  hardAccuracy: number;
  softAccuracy: number;
  predHard: Float32Array;
  predSoft: Float32Array;
  damaged: number;
  /** Hard accuracy of the displayed text reconstruction (text mode); null in
   *  "cases" mode. hardAccuracy/softAccuracy always reflect the base inference
   *  on the diverse batch. */
  textAccuracy: number | null;
  /** ``[caseN, outputBits]`` flat array, useful for redraws between ticks. */
  caseN: number;
  /** How many leading cases the UI should render (= caseN in "cases" mode; the
   *  text-column count in "text" mode, where the batch also carries hidden filler). */
  displayCaseN: number;
  outputBits: number;
}

export interface ControllerOptions {
  caseN: number;
  xData: Float32Array; // [caseN * inputBits]
  yData: Float32Array; // [caseN * outputBits]
  inputBits: number;
  outputBits: number;
  layerSizes: LayerSize[];
  rngSeed?: number;
  noiseScale?: number;
  /** Pre-computed wires from the training run (fixed-wires models).
   *  When present, topology is built from these instead of generating via RNG. */
  precomputedWires?: Int32Array[];
  /** Ranked pool of pre-screened random topologies, best-first (rank-0 = the
   *  crispest). When present, the initial topology is pool[0] and ``shuffle()``
   *  walks the pool in order, falling back to fresh random wirings once the pool
   *  is exhausted. Each entry is per-gate-layer wires, same shape as
   *  ``precomputedWires``. */
  topologyPool?: Int32Array[][];
  /** Text-reverse display columns (the demo's "Text" mode). When present, the
   *  controller can switch to a batch of ``[textCols + random filler]`` and
   *  display only the text slice (the filler keeps the node-loss residual healthy
   *  so the circuit settles — sparse text alone starves it). */
  textXData?: Float32Array; // [textCaseN * inputBits]
  textYData?: Float32Array; // [textCaseN * outputBits]
  textCaseN?: number;
}

export type DisplayMode = "cases" | "text";

export class Controller {
  readonly weights: TMTWeights;
  readonly arity: number;
  readonly hiddenDim: number;
  readonly maxNeighbors: number;
  /** Active BATCH size (cases the circuit runs on; includes hidden filler in text mode). */
  caseN: number;
  readonly inputBits: number;
  readonly outputBits: number;
  xData: Float32Array;
  yData: Float32Array;
  /** Leading cases the UI renders (= caseN in cases-mode; text-col count in text-mode). */
  displayCaseN: number;
  displayMode: DisplayMode = "cases";
  readonly layerSizes: LayerSize[];
  /** Text-reverse display columns. The TMT ALWAYS runs on the diverse batch
   *  (xData/yData) for the residual; these are forwarded for display only. */
  private _textCols?: { x: Float32Array; y: Float32Array; n: number };

  topology: Topology;
  state: CircuitState;
  scratch: Scratch;
  rng: Rng;
  step = 0;
  /** When the user has manually damaged a specific gate (clicks), we keep
   *  a rolling list so the rendering can highlight the most recent click. */
  lastDamagedFlatIdx: number | null = null;
  private _precomputedWires: Int32Array[] | undefined;
  private _topologyPool: Int32Array[][] | undefined;
  /** Index of the NEXT pool topology ``shuffle()`` will show (rank-0 shows at boot). */
  private _poolCursor = 0;
  private _originalTopology: Topology;

  constructor(weights: TMTWeights, opts: ControllerOptions) {
    this.weights = weights;
    this.arity = weights.header.arity;
    this.hiddenDim = weights.header.circuitHiddenDim;
    this.maxNeighbors = weights.header.maxNeighbors;
    this.caseN = opts.caseN;
    this.xData = opts.xData;
    this.yData = opts.yData;
    this.displayCaseN = opts.caseN;
    this.inputBits = opts.inputBits;
    this.outputBits = opts.outputBits;
    this.layerSizes = opts.layerSizes;
    if (opts.textXData && opts.textYData && opts.textCaseN) {
      this._textCols = { x: opts.textXData, y: opts.textYData, n: opts.textCaseN };
    }
    this.rng = makeRng(opts.rngSeed ?? 44);
    this._precomputedWires = opts.precomputedWires;
    this._topologyPool = opts.topologyPool;

    if (this._topologyPool && this._topologyPool.length > 0) {
      // Curated pool: boot on rank-0 (crispest); shuffle walks down from rank-1.
      this._originalTopology = this._topologyFromWires(this._topologyPool[0]);
      this._poolCursor = 1;
    } else {
      this._originalTopology = this._freshTopology();
    }
    this.topology = this._originalTopology;
    this.state = initState(
      this.topology,
      { arity: this.arity, hiddenDim: this.hiddenDim, noiseScale: opts.noiseScale ?? 0.1 },
      this.rng,
    );
    this.scratch = makeScratch(weights, this.topology.nNodes);
    // Initial residual computation — gives the visualisation a tick-0 frame.
    computeResiduals(this.state, this.topology, this.xData, this.yData, this.caseN, this.arity);
  }

  // ------------------------------------------------------------------
  // Public actions (each returns a snapshot)
  // ------------------------------------------------------------------

  tick(): ControllerSnapshot {
    const aux = runTick(
      this.state,
      this.topology,
      this.weights,
      this.scratch,
      this.xData,
      this.yData,
      this.caseN,
      this.arity,
    );
    this.step++;
    return this._snapshot(aux);
  }

  reset(): ControllerSnapshot {
    this.topology = this._originalTopology;
    // Restart the pool walk (rank-0 is showing again).
    if (this._topologyPool) this._poolCursor = 1;
    this.state = resetCircuit(this.topology, this.rng, this.arity, this.hiddenDim);
    this.step = 0;
    this.lastDamagedFlatIdx = null;
    const aux = computeResiduals(
      this.state,
      this.topology,
      this.xData,
      this.yData,
      this.caseN,
      this.arity,
    );
    return this._snapshot(aux);
  }

  shuffle(): ControllerSnapshot {
    if (this._topologyPool && this._poolCursor < this._topologyPool.length) {
      // Walk the curated pool best-first; every shuffle lands a real (pre-vetted)
      // random topology, shown crispest-first.
      this.topology = this._topologyFromWires(this._topologyPool[this._poolCursor]);
      this._poolCursor++;
    } else {
      // Pool exhausted (or absent) → genuinely fresh random wiring. Bottomless,
      // just no longer best-first.
      this.topology = shuffleWires(
        this.topology,
        this.rng,
        this.arity,
        this.hiddenDim,
        this.maxNeighbors,
      );
    }
    zeroHidden(this.state);
    // Mirror DemoSession._restart_tmt_generator: hidden + loss reset to 0,
    // logits + gateMask carry over.
    this.state.loss.fill(0);
    const aux = computeResiduals(
      this.state,
      this.topology,
      this.xData,
      this.yData,
      this.caseN,
      this.arity,
    );
    return this._snapshot(aux);
  }

  /** Knock out ``nGates`` random hidden gates. Returns the snapshot plus
   *  the chosen flat indices so the renderer can flash them. */
  shotgun(nGates: number = 5): { snapshot: ControllerSnapshot; chosen: Int32Array } {
    const lutDim = 1 << this.arity;
    const chosen = applyShotgun(this.state, this.topology, nGates, this.rng, lutDim);
    const aux = computeResiduals(
      this.state,
      this.topology,
      this.xData,
      this.yData,
      this.caseN,
      this.arity,
    );
    return { snapshot: this._snapshot(aux), chosen };
  }

  /** Click-to-damage a single gate by its flat index. ``null`` if the
   *  index is ineligible (input / output layer) or already damaged. */
  damage(flatIdx: number): ControllerSnapshot | null {
    const lutDim = 1 << this.arity;
    if (!damageGate(this.state, this.topology, flatIdx, lutDim)) return null;
    this.lastDamagedFlatIdx = flatIdx;
    const aux = computeResiduals(
      this.state,
      this.topology,
      this.xData,
      this.yData,
      this.caseN,
      this.arity,
    );
    return this._snapshot(aux);
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  /** True when this model ships text-reverse columns (enables the Text toggle). */
  get hasTextMode(): boolean {
    return this._textCols !== undefined;
  }

  /** Switch the displayed case-set. The TMT ALWAYS runs on the diverse batch
   *  (constant cost, healthy residual); "text" just forwards the SAME evolved
   *  circuit on the text columns for display — no re-settle, no extra batch. */
  setDisplayMode(mode: DisplayMode): ControllerSnapshot | null {
    if (mode === "text" && !this._textCols) return null;
    this.displayMode = mode;
    this.displayCaseN = mode === "text" && this._textCols ? this._textCols.n : this.caseN;
    // Recompute the (unchanged) diverse residual snapshot + re-render the display.
    const aux = computeResiduals(
      this.state, this.topology, this.xData, this.yData, this.caseN, this.arity,
    );
    return this._snapshot(aux);
  }

  /** Display x / y for the active mode (text columns in text mode; the diverse
   *  batch otherwise) — display-only; the TMT always runs on the diverse batch. */
  get displayXData(): Float32Array {
    return this.displayMode === "text" && this._textCols ? this._textCols.x : this.xData;
  }
  get displayYData(): Float32Array {
    return this.displayMode === "text" && this._textCols ? this._textCols.y : this.yData;
  }

  /** Forward the CURRENT circuit on display cases (soft+hard preds + accuracy)
   *  WITHOUT touching the TMT residual state — pure display, ~1ms. */
  private _forwardDisplay(x: Float32Array, y: Float32Array, n: number) {
    const softActs = circuitForward(this.topology, this.state, x, n, false, this.arity);
    const hardActs = circuitForward(this.topology, this.state, x, n, true, this.arity);
    const predSoft = softActs[softActs.length - 1];
    const predHard = hardActs[hardActs.length - 1];
    let matches = 0;
    const len = n * this.outputBits;
    for (let i = 0; i < len; i++) if (predHard[i] === y[i]) matches++;
    return { predSoft, predHard, acc: matches / len };
  }

  /** Build a Topology from an explicit per-gate-layer wire list (curated pool). */
  private _topologyFromWires(poolWires: Int32Array[]): Topology {
    const wires: Int32Array[] = [];
    const wiresShape: [number, number][] = [];
    for (let li = 1; li < this.layerSizes.length; li++) {
      const [outN, gs] = this.layerSizes[li];
      wires.push(poolWires[li - 1]);
      wiresShape.push([this.arity, outN / gs]);
    }
    return buildTopology(this.layerSizes, wires, wiresShape, {
      arity: this.arity,
      hiddenDim: this.hiddenDim,
      maxNeighbors: this.maxNeighbors,
    });
  }

  private _freshTopology(): Topology {
    const wires: Int32Array[] = [];
    const wiresShape: [number, number][] = [];
    for (let li = 1; li < this.layerSizes.length; li++) {
      const [outN, gs] = this.layerSizes[li];
      const inN = this.layerSizes[li - 1][0];
      if (this._precomputedWires) {
        // Use the exact wires from the training run (fixed-wires models).
        wires.push(this._precomputedWires[li - 1]);
        wiresShape.push([this.arity, outN / gs]);
      } else {
        const w = genWires(this.rng, inN, outN, this.arity, gs);
        wires.push(w.data);
        wiresShape.push(w.shape);
      }
    }
    return buildTopology(this.layerSizes, wires, wiresShape, {
      arity: this.arity,
      hiddenDim: this.hiddenDim,
      maxNeighbors: this.maxNeighbors,
    });
  }

  private _snapshot(aux: { hardAccuracy: number; softAccuracy: number; predHard: Float32Array; predSoft: Float32Array }): ControllerSnapshot {
    // In text mode, the displayed prediction is the SAME evolved circuit forwarded
    // on the text columns (cheap, no TMT) — the residual/aux above stays on the
    // diverse batch. hardAccuracy then reflects the text reconstruction.
    let { predHard, predSoft } = aux;
    let displayCaseN = this.caseN;
    let textAccuracy: number | null = null;
    if (this.displayMode === "text" && this._textCols) {
      const t = this._textCols;
      const d = this._forwardDisplay(t.x, t.y, t.n);
      predHard = d.predHard; // strips show the text reconstruction
      predSoft = d.predSoft;
      textAccuracy = d.acc; // separate readout; base hard/soft stay on the diverse batch
      displayCaseN = t.n;
    }
    return {
      step: this.step,
      hardAccuracy: aux.hardAccuracy, // base inference (diverse), unchanged
      softAccuracy: aux.softAccuracy,
      textAccuracy,
      predHard,
      predSoft,
      damaged: this.countDamaged(),
      caseN: this.caseN,
      displayCaseN,
      outputBits: this.outputBits,
    };
  }

  countDamaged(): number {
    let n = 0;
    for (let i = 0; i < this.state.gateMask.length; i++) if (this.state.gateMask[i] === 0) n++;
    return n;
  }

  /** Total number of hidden gates (eligible for damage). */
  totalHiddenGates(): number {
    let n = 0;
    for (let li = 1; li < this.topology.layerSizes.length - 1; li++) {
      n += this.topology.layerSizes[li][0];
    }
    return n;
  }
}
