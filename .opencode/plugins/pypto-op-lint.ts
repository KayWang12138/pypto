import type { Plugin } from "@opencode-ai/plugin";

type HookOutput = {
  hookSpecificOutput?: {
    additionalContext?: string;
  };
};

const DELIVERY_GATE_REMINDER =
  "[pypto-op-lint] 若准备结束当前阶段，请在算子目录显式运行 `python3 .agents/hooks/pypto-op-lint/pypto_op_lint.py --check-gate --op-dir <算子目录> --stage <当前阶段>` 做交付自检。";

// NOTE: OpenCode plugin API 不支持 stop/session-end hook，因此无法像 Claude Code
// (.agents/settings.json Stop hook) 那样在 agent 结束时自动阻断。当前只能通过
// post-bash 后的文字提醒引导用户手动运行 --check-gate。

function appendMessage(output: { output?: string }, message: string): void {
  output.output = `${output.output ?? ""}\n\n${message}`.trim();
}

function formatPluginError(scope: string, error: unknown): string {
  const detail = error instanceof Error ? error.message : String(error);
  return `[pypto-op-lint plugin-error] ${scope} 自动检查失败：${detail}`;
}

function extractAdditionalContext(raw: string): string {
  if (!raw.trim()) return "";

  try {
    const parsed = JSON.parse(raw) as HookOutput;
    return parsed.hookSpecificOutput?.additionalContext ?? "";
  } catch (error) {
    return formatPluginError("post-edit 输出解析", error);
  }
}

export const PyptoOpLintPlugin: Plugin = async (input) => {
  const $ = input.$;
  const lint = `python3 ${input.directory}/.agents/hooks/pypto-op-lint/pypto_op_lint.py`;

  return {
    "tool.execute.after": async ({ tool, args }, output) => {
      const filePath: string = args?.file_path ?? args?.path ?? "";
      if (
        filePath.endsWith("_impl.py") ||
        filePath.endsWith("_golden.py") ||
        /test_\w+\.py$/.test(filePath)
      ) {
        try {
          const raw = await $`${lint} --hook post-edit`.stdin(
            JSON.stringify({ tool_input: { file_path: filePath } }),
          );
          const context = extractAdditionalContext(await raw.text());
          if (context) appendMessage(output, context);
        } catch (error) {
          appendMessage(output, formatPluginError("post-edit", error));
        }
        return;
      }

      if (
        (tool === "bash" || tool === "shell") &&
        /python3?\s+.*test_\w+\.py/.test(args?.command ?? "")
      ) {
        try {
          const currentOutput = output as {
            output?: string;
            stderr?: string;
            exit_code?: number;
            code?: number;
          };
          const raw = await $`${lint} --hook post-bash`.stdin(
            JSON.stringify({
              tool_input: { command: args?.command ?? "" },
              tool_result: {
                stdout: currentOutput.output ?? "",
                stderr: currentOutput.stderr ?? "",
                exit_code:
                  typeof currentOutput.exit_code === "number"
                    ? currentOutput.exit_code
                    : typeof currentOutput.code === "number"
                      ? currentOutput.code
                      : 0,
              },
            }),
          );
          const context = extractAdditionalContext(await raw.text());
          if (context) appendMessage(output, context);
          appendMessage(output, DELIVERY_GATE_REMINDER);
        } catch (error) {
          appendMessage(output, formatPluginError("post-bash", error));
        }
      }

      return;
    },

    "tool.execute.before": async ({ tool }, output) => {
      if (tool !== "write" && tool !== "edit" && tool !== "multiedit") return;

      const filePath = output.args?.file_path ?? "";
      if (!filePath.endsWith("_impl.py")) return;

      // pre-edit: 纯 lint 检查（无副作用）
      try {
        await $`${lint} --hook pre-edit`.stdin(
          JSON.stringify({ tool_input: { file_path: filePath } }),
        );
      } catch (error) {
        appendMessage(output, formatPluginError("pre-edit", error));
      }

      // pre-edit-backup: Stage 6 git 备份（独立于 lint）
      try {
        await $`${lint} --hook pre-edit-backup`.stdin(
          JSON.stringify({ tool_input: { file_path: filePath } }),
        );
      } catch (error) {
        appendMessage(output, formatPluginError("pre-edit-backup", error));
      }
    },
  };
};
