import { createHash, randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { delimiter } from "node:path";
import { createInterface } from "node:readline";

const REQUEST_SCHEMA = "hermes.bitagent_sidecar_request.v1";
const RESPONSE_SCHEMA = "hermes.bitagent_sidecar_response.v1";
const MAX_LINE_BYTES = 1024 * 1024;
const ALLOWED_AUTHORITIES = new Set([
  "no_effect",
  "candidate_only_or_capability_material_no_execution",
]);

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonicalize(value[key])]),
    );
  }
  return value;
}

function canonicalHash(value) {
  return createHash("sha256")
    .update(JSON.stringify(canonicalize(value)), "utf8")
    .digest("hex");
}

export class HermesRoleMeshSidecar {
  constructor({
    cwd,
    python = "python",
    roleConfig = "configs/bitagent_bonsai_role_adapters_v1.json",
    protocol = "configs/bitagent_hermes_control_mesh_v0.json",
    timeoutMs = 5000,
  }) {
    if (!cwd) throw new Error("cwd is required");
    this.cwd = cwd;
    this.python = python;
    this.roleConfig = roleConfig;
    this.protocol = protocol;
    this.timeoutMs = timeoutMs;
    this.child = null;
    this.pending = new Map();
    this.stderr = "";
  }

  start() {
    if (this.child) return;
    this.stderr = "";
    const args = [
      "-m",
      "agent.bitagent_sidecar_v1",
      "--role-config",
      this.roleConfig,
      "--protocol",
      this.protocol,
      "serve",
    ];
    const env = {
      ...process.env,
      PYTHONPATH: process.env.PYTHONPATH
        ? `${this.cwd}/src${delimiter}${process.env.PYTHONPATH}`
        : `${this.cwd}/src`,
    };
    this.child = spawn(this.python, args, {
      cwd: this.cwd,
      env,
      shell: false,
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"],
    });
    const lines = createInterface({ input: this.child.stdout });
    lines.on("line", (line) => this.#receive(line));
    this.child.stderr.on("data", (chunk) => {
      this.stderr = (this.stderr + chunk.toString("utf8")).slice(-4000);
    });
    this.child.once("exit", (code, signal) => {
      const diagnostic = this.stderr.trim()
        ? ` stderr=${this.stderr.trim().slice(-500)}`
        : "";
      const error = new Error(
        `Hermes sidecar exited code=${code} signal=${signal ?? ""}${diagnostic}`,
      );
      for (const entry of this.pending.values()) {
        clearTimeout(entry.timer);
        entry.reject(error);
      }
      this.pending.clear();
      this.child = null;
    });
  }

  request(payload) {
    this.start();
    const requestId = payload.request_id || randomUUID();
    const value = {
      ...payload,
      schema: REQUEST_SCHEMA,
      request_id: requestId,
    };
    const line = JSON.stringify(value);
    if (Buffer.byteLength(line, "utf8") > MAX_LINE_BYTES) {
      return Promise.reject(new Error("Hermes sidecar request exceeds 1 MiB"));
    }
    if (this.pending.has(requestId)) {
      return Promise.reject(new Error(`duplicate request_id ${requestId}`));
    }
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(requestId);
        reject(new Error(`Hermes sidecar timeout for ${requestId}`));
      }, this.timeoutMs);
      this.pending.set(requestId, {
        resolve,
        reject,
        timer,
        operation: value.operation,
      });
      this.child.stdin.write(`${line}\n`, "utf8", (error) => {
        if (!error) return;
        clearTimeout(timer);
        this.pending.delete(requestId);
        reject(error);
      });
    });
  }

  async close() {
    const child = this.child;
    if (!child) return;
    child.stdin.end();
    await new Promise((resolve) => {
      const timer = setTimeout(() => {
        if (this.child === child) child.kill();
        resolve();
      }, 2000);
      child.once("exit", () => {
        clearTimeout(timer);
        resolve();
      });
    });
  }

  #receive(line) {
    if (Buffer.byteLength(line, "utf8") > MAX_LINE_BYTES) return;
    let value;
    try {
      value = JSON.parse(line);
    } catch {
      return;
    }
    if (value.schema !== RESPONSE_SCHEMA) return;
    const entry = this.pending.get(value.request_id);
    if (!entry) return;
    clearTimeout(entry.timer);
    this.pending.delete(value.request_id);
    if (value.operation !== entry.operation) {
      entry.reject(new Error("Hermes sidecar operation identity mismatch"));
      return;
    }
    if (value.ok) {
      if (!ALLOWED_AUTHORITIES.has(value.authority)) {
        entry.reject(new Error("Hermes sidecar authority boundary mismatch"));
        return;
      }
      if (!value.result || typeof value.result !== "object") {
        entry.reject(new Error("Hermes sidecar result contract mismatch"));
        return;
      }
      const expectedAuthority =
        entry.operation === "materialize_capability_request"
          ? "candidate_only_or_capability_material_no_execution"
          : "no_effect";
      if (value.authority !== expectedAuthority) {
        entry.reject(new Error("Hermes sidecar operation authority mismatch"));
        return;
      }
      const expectedHash = canonicalHash({
        request_id: value.request_id,
        operation: value.operation,
        result: value.result,
      });
      if (value.response_sha256 !== expectedHash) {
        entry.reject(new Error("Hermes sidecar response hash mismatch"));
        return;
      }
      entry.resolve(value);
    } else {
      if (value.authority !== "no_effect") {
        entry.reject(new Error("Hermes sidecar error authority mismatch"));
        return;
      }
      const error = new Error(value.error?.detail || "Hermes sidecar rejected request");
      error.code = value.error?.code || "sidecar_error";
      entry.reject(error);
    }
  }
}
