// Types for fllm-agent.mjs (the client of `fllm-rs agent --rpc`).
export class AgentError extends Error {
  code: number;
}

export type Category = 'edit' | 'command' | 'network' | 'memory' | 'all';

export interface AgentEvent {
  session: string;
  type: 'text' | 'tool_call' | 'tool_result';
  text?: string;
  id?: string;
  name?: string;
  args?: Record<string, unknown>;
  ok?: boolean;
}

export interface ApprovalRequest {
  session: string;
  call_id: string;
  tool: string;
  category: string;
  args: Record<string, any>;
}

export interface PromptResult {
  answer: string;
  status: 'done' | 'out_of_steps' | 'cancelled';
  steps: number;
  edited: string[];
  tool_calls: { tool: string; permission: string; status: string; detail: string }[];
}

export interface SessionOptions {
  allow?: Category[];
  plan?: boolean;
  model?: string;
  system?: string;
  maxSteps?: number;
  allowRules?: string[];
  denyRules?: string[];
  askRules?: string[];
}

export class Session {
  readonly id: string;
  prompt(
    text: string,
    options?: {
      onEvent?: (event: AgentEvent) => void;
      approve?: (call: ApprovalRequest) => boolean | { allow: boolean; always?: boolean } | Promise<boolean | { allow: boolean; always?: boolean }>;
    },
  ): Promise<PromptResult>;
  cancel(): void;
  history(): Promise<unknown[]>;
  close(): Promise<unknown>;
}

export class Agent {
  info: { name: string; version: string; protocol: number; methods: string[]; categories: string[] };
  static start(options?: { exe?: string; url?: string; apiKey?: string; model?: string; extraArgs?: string[] }): Promise<Agent>;
  newSession(folder: string, options?: SessionOptions): Promise<Session>;
  tools(): Promise<{ name: string; description: string; category: string }[]>;
  request(method: string, params?: unknown): Promise<any>;
  notify(method: string, params?: unknown): void;
  close(): Promise<void>;
}
