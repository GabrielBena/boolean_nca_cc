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
  /** ``[caseN, outputBits]`` flat array, useful for redraws between ticks. */
  caseN: number;
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
}

export class Controller {
  readonly weights: TMTWeights;
  readonly arity: number;
  readonly hiddenDim: number;
  readonly maxNeighbors: number;
  readonly caseN: number;
  readonly inputBits: number;
  readonly outputBits: number;
  readonly xData: Float32Array;
  readonly yData: Float32Array;
  readonly layerSizes: LayerSize[];

  topology: Topology;
  state: CircuitState;
  scratch: Scratch;
  rng: Rng;
  step = 0;
  /** When the user has manually damaged a specific gate (clicks), we keep
   *  a rolling list so the rendering can highlight the most recent click. */
  lastDamagedFlatIdx: number | null = null;

  constructor(weights: TMTWeights, opts: ControllerOptions) {
    this.weights = weights;
    this.arity = weights.header.arity;
    this.hiddenDim = weights.header.circuitHiddenDim;
    this.maxNeighbors = weights.header.maxNeighbors;
    this.caseN = opts.caseN;
    this.xData = opts.xData;
    this.yData = opts.yData;
    this.inputBits = opts.inputBits;
    this.outputBits = opts.outputBits;
    this.layerSizes = opts.layerSizes;
    this.rng = makeRng(opts.rngSeed ?? 44);

    this.topology = this._freshTopology();
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
    this.topology = shuffleWires(
      this.topology,
      this.rng,
      this.arity,
      this.hiddenDim,
      this.maxNeighbors,
    );
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

  private _freshTopology(): Topology {
    const wires: Int32Array[] = [];
    const wiresShape: [number, number][] = [];
    for (let li = 1; li < this.layerSizes.length; li++) {
      const [outN, gs] = this.layerSizes[li];
      const inN = this.layerSizes[li - 1][0];
      const w = genWires(this.rng, inN, outN, this.arity, gs);
      wires.push(w.data);
      wiresShape.push(w.shape);
    }
    return buildTopology(this.layerSizes, wires, wiresShape, {
      arity: this.arity,
      hiddenDim: this.hiddenDim,
      maxNeighbors: this.maxNeighbors,
    });
  }

  private _snapshot(aux: { hardAccuracy: number; softAccuracy: number; predHard: Float32Array; predSoft: Float32Array }): ControllerSnapshot {
    return {
      step: this.step,
      hardAccuracy: aux.hardAccuracy,
      softAccuracy: aux.softAccuracy,
      predHard: aux.predHard,
      predSoft: aux.predSoft,
      damaged: this.countDamaged(),
      caseN: this.caseN,
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
