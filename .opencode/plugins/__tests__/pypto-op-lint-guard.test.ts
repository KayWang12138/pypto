import { expect, test } from "bun:test";
import { $ } from "bun";
import { PyptoOpLintPlugin } from "../pypto-op-lint";

test("blocks direct state write via write tool", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);

  const before = plugin["tool.execute.before"];
  await expect(
    before?.(
      { tool: "write" } as never,
      { args: { file_path: "/tmp/qat/.orchestrator_state.json", content: "{}" } } as never,
    ),
  ).rejects.toThrow("禁止直接修改 .orchestrator_state.json");
});

test("blocks direct state write via bash tool", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);

  const before = plugin["tool.execute.before"];
  await expect(
    before?.(
      { tool: "bash" } as never,
      { args: { command: "echo '{}' > /tmp/qat/.orchestrator_state.json" } } as never,
    ),
  ).rejects.toThrow("禁止通过 bash/shell 直接写入");
});

test("no watcher handler registered", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);

  expect(plugin["file.watcher.updated"]).toBeUndefined();
});

test("allows readonly bash commands mentioning state file", async () => {
  const plugin = await PyptoOpLintPlugin({
    $,
    client: { app: { log: async () => {} } },
    directory: process.cwd(),
    worktree: process.cwd(),
    project: {},
  } as never);

  const before = plugin["tool.execute.before"];

  // cat、ls、stat、grep 等只读命令不应被拦截
  const readonlyCommands = [
    "cat /tmp/qat/.orchestrator_state.json",
    "ls -la /tmp/qat/.orchestrator_state.json",
    "stat /tmp/qat/.orchestrator_state.json",
    "grep current_stage /tmp/qat/.orchestrator_state.json",
    "test -f /tmp/qat/.orchestrator_state.json",
  ];

  for (const cmd of readonlyCommands) {
    // 只读命令不应抛出异常
    await expect(
      before?.(
        { tool: "bash" } as never,
        { args: { command: cmd } } as never,
      ),
    ).resolves.toBeUndefined();
  }
});
