export type SidecarOperation =
  | "packetize"
  | "validate"
  | "materialize_capability_request";

export type SidecarRequest = {
  request_id?: string;
  operation: SidecarOperation;
  task_card: Record<string, unknown>;
  workflow_state: Record<string, unknown>;
  tool_contracts?: Array<Record<string, unknown>>;
  current_evidence?: Array<Record<string, unknown>>;
  replay_candidates?: Array<Record<string, unknown>>;
  candidate?: Record<string, unknown>;
};

export type SidecarResponse = {
  schema: "hermes.bitagent_sidecar_response.v1";
  request_id: string;
  operation: SidecarOperation;
  ok: true;
  authority:
    | "no_effect"
    | "candidate_only_or_capability_material_no_execution";
  result: Record<string, unknown>;
  response_sha256: string;
};

export declare class HermesRoleMeshSidecar {
  constructor(options: {
    cwd: string;
    python?: string;
    roleConfig?: string;
    protocol?: string;
    timeoutMs?: number;
  });
  start(): void;
  request(payload: SidecarRequest): Promise<SidecarResponse>;
  close(): Promise<void>;
}
