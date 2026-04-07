#!/bin/bash

# 移除set -e，使用显式错误处理以避免意外退出
set +e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HOME_DIR=$(eval echo ~)
TARGET_DIR="${HOME_DIR}/skill_monitor_test"
PYPTO_DIR="${TARGET_DIR}/pypto_${TIMESTAMP}"

# opencode工具路径（从~/.opencode查找）
OPENCODE_BIN="${HOME_DIR}/.opencode/bin/opencode"

echo "========================================"
echo "Skill Monitor Test Script"
echo "========================================"
echo "  Timestamp: ${TIMESTAMP}"
echo "  Target Dir: ${TARGET_DIR}"
echo "  PyPTO Dir: ${PYPTO_DIR}"
echo "========================================"
echo ""

echo "Step 0: Creating target directory..."
mkdir -p "${TARGET_DIR}"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create target directory"
    exit 1
fi

echo "Step 1: Cloning pypto repository..."
cd "${TARGET_DIR}" || exit 1
git clone https://gitcode.com/cann/pypto.git
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to clone pypto repository"
    exit 1
fi

echo "Step 2: Renaming to pypto_${TIMESTAMP}..."
mv pypto "pypto_${TIMESTAMP}"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to rename pypto directory"
    exit 1
fi

echo "Step 3: Verifying required resources..."
# 检查opencode工具
if [[ ! -f "${OPENCODE_BIN}" ]]; then
    echo "[WARNING] opencode not found at ${OPENCODE_BIN}"
    echo "[WARNING] Will try to find opencode in PATH..."
    if command -v opencode &> /dev/null; then
        OPENCODE_BIN="opencode"
        echo "[INFO] Found opencode in PATH"
    else
        echo "[ERROR] Cannot find opencode tool"
        echo "[ERROR] Please install opencode or set OPENCODE_BIN environment variable"
        exit 1
    fi
else
    echo "  ✓ Found opencode at ${OPENCODE_BIN}"
fi

# 检查pypto仓库中的必需skill
echo "  Checking required skills in pypto repository..."
REQUIRED_SKILLS=("pypto-skill-reviewer" "skill-validation-prompt")
for skill in "${REQUIRED_SKILLS[@]}"; do
    skill_dir="${PYPTO_DIR}/.agents/skills/${skill}"
    if [[ ! -d "${skill_dir}" ]]; then
        echo "[ERROR] Required skill ${skill} not found in cloned pypto repository"
        exit 1
    fi
    echo "  ✓ Found ${skill}"
done

echo "Done! PyPTO directory: ${PYPTO_DIR}"
echo ""

echo "Step 4: Scanning skills..."
cd "${PYPTO_DIR}"

# 激活conda环境（可选）
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate agent_conda 2>/dev/null || echo "[WARNING] Failed to activate agent_conda, proceeding without it"
else
    echo "[WARNING] conda not found, proceeding without it"
fi

SKILL_LIST=()
for skill_dir in ./.agents/skills/pypto*; do
    if [[ -f "${skill_dir}/SKILL.md" ]] && [[ ! "${skill_dir}" =~ "pass" ]]; then
        skill_name=$(basename "${skill_dir}")
        SKILL_LIST+=("${skill_name}")
    fi
done

TOTAL_SKILLS=${#SKILL_LIST[@]}
echo "Found ${TOTAL_SKILLS} skills:"
printf '  - %s\n' "${SKILL_LIST[@]}"

# 创建状态跟踪目录
mkdir -p "${PYPTO_DIR}/.task_status"

echo ""
echo "========================================"
echo "Task 1 Round 1: Batch generating validation prompts for all skills"
echo "========================================"

SKILL_STR=$(IFS=,; echo "${SKILL_LIST[@]}")

echo "  Starting Task 1 Round 1 (timeout: 30 minutes)..."
START_TIME=$(date +%s)

timeout 1800 "${OPENCODE_BIN}" run "对所有以下skill调用/skill-validation-prompt：${SKILL_STR}

重要要求：
1. 对每个skill，先创建目录：mkdir -p ./skill_check_report/{skill名}
2. 在对应目录下生成名为 validation.md 的文件（不是其他名字）
3. 文件路径必须是：./skill_check_report/{skill名}/validation.md
4. 批量处理所有skill，不要遗漏任何一个
5. 不要在当前工作目录下直接生成文件" --model zhipuai-coding-plan/glm-5 > "${PYPTO_DIR}/task1_batch.log" 2>&1

TASK1_EXIT_CODE=$?
ELAPSED_TIME=$(( $(date +%s) - START_TIME ))

if [ $TASK1_EXIT_CODE -eq 124 ]; then
    echo "  [TIMEOUT] Task 1 Round 1 exceeded 30 minutes, terminated"
elif [ $TASK1_EXIT_CODE -ne 0 ]; then
    echo "  [ERROR] Task 1 Round 1 failed with exit code $TASK1_EXIT_CODE"
else
    echo "  ✓ Task 1 Round 1 completed successfully in ${ELAPSED_TIME}s"
fi
echo ""

# Task 1 Round 2: Retry skills with missing validation.md
echo "========================================"
echo "Task 1 Round 2: Checking for missing validation.md files"
echo "========================================"

MISSING_VALIDATION_SKILLS=()
for skill_name in "${SKILL_LIST[@]}"; do
    validation_file="${PYPTO_DIR}/skill_check_report/${skill_name}/validation.md"
    if [[ ! -f "${validation_file}" ]]; then
        MISSING_VALIDATION_SKILLS+=("${skill_name}")
        echo "  - Missing: ${skill_name}"
    fi
done

MISSING_COUNT=${#MISSING_VALIDATION_SKILLS[@]}
if [ $MISSING_COUNT -eq 0 ]; then
    echo "  ✓ All skills have validation.md files"
else
    echo ""
    echo "  Found ${MISSING_COUNT} skills with missing validation.md"
    echo "  Starting Task 1 Round 2 to generate them..."
    
    MISSING_SKILL_STR=$(IFS=,; echo "${MISSING_VALIDATION_SKILLS[@]}")
    START_TIME=$(date +%s)
    
    timeout 1800 "${OPENCODE_BIN}" run "对所有以下skill调用/skill-validation-prompt：${MISSING_SKILL_STR}

重要要求：
1. 对每个skill，先创建目录：mkdir -p ./skill_check_report/{skill名}（如果已存在则忽略）
2. 在对应目录下生成名为 validation.md 的文件（不是其他名字）
3. 文件路径必须是：./skill_check_report/{skill名}/validation.md
4. 批量处理所有skill，不要遗漏任何一个
5. 不要在当前工作目录下直接生成文件" --model zhipuai-coding-plan/glm-5 > "${PYPTO_DIR}/task1_batch_round2.log" 2>&1
    
    TASK1_R2_EXIT_CODE=$?
    ELAPSED_TIME=$(( $(date +%s) - START_TIME ))
    
    if [ $TASK1_R2_EXIT_CODE -eq 124 ]; then
        echo "  [TIMEOUT] Task 1 Round 2 exceeded 30 minutes, terminated"
    elif [ $TASK1_R2_EXIT_CODE -ne 0 ]; then
        echo "  [ERROR] Task 1 Round 2 failed with exit code $TASK1_R2_EXIT_CODE"
    else
        echo "  ✓ Task 1 Round 2 completed successfully in ${ELAPSED_TIME}s"
    fi
    
    # Final check: report which skills still missing validation.md
    STILL_MISSING=0
    for skill_name in "${MISSING_VALIDATION_SKILLS[@]}"; do
        validation_file="${PYPTO_DIR}/skill_check_report/${skill_name}/validation.md"
        if [[ ! -f "${validation_file}" ]]; then
            echo "  [WARNING] ${skill_name}: validation.md still missing after Round 2"
            STILL_MISSING=$((STILL_MISSING + 1))
        fi
    done
    
    if [ $STILL_MISSING -gt 0 ]; then
        echo "  [INFO] ${STILL_MISSING} skills still lack validation.md, Task 2 will be skipped for them"
    fi
fi
echo ""

echo "========================================"
echo "Task 2 & 4: Processing each skill (review + execute validation)"
echo "========================================"
echo ""

CURRENT_SKILL=0
for skill_name in "${SKILL_LIST[@]}"; do
    CURRENT_SKILL=$((CURRENT_SKILL + 1))
    echo "========================================"
    echo "[$CURRENT_SKILL/${TOTAL_SKILLS}] Processing: ${skill_name}"
    echo "========================================"
    
    validation_file="./skill_check_report/${skill_name}/validation.md"
    validation_content=""
    TASK2_SKIPPED=false
    
    # Check if validation.md exists
    if [[ -f "${validation_file}" ]]; then
        validation_content=$(cat "${validation_file}")
        echo "  ✓ Found validation.md for ${skill_name}"
        
        # Task 2: Generate review.md (requires validation.md)
        setsid "${OPENCODE_BIN}" run "/pypto-skill-reviewer，检查以下skill: ${skill_name}。跳过全部skill的语义检查，只进行静态检查。

重要要求：
1. 先创建目录：mkdir -p ./skill_check_report/${skill_name}（如果已存在则忽略）
2. 在该目录下生成名为 review.md 的文件（不是其他名字）
3. 文件路径必须是：./skill_check_report/${skill_name}/review.md
4. 不要在当前工作目录下直接生成文件" --model zhipuai-coding-plan/glm-5 > "${PYPTO_DIR}/task2_${skill_name}.log" 2>&1 &
        TASK2_PID=$!
        TASK2_PGID=$(ps -o pgid= -p $TASK2_PID | tr -d ' ')
        echo "  Task2 PID: $TASK2_PID (review.md)"
        sleep 1
    else
        echo "  [WARNING] ${skill_name}: validation.md not found, skipping Task 2"
        TASK2_SKIPPED=true
        TASK2_PID="SKIP"
        echo "SKIPPED_NO_VALIDATION" > "${PYPTO_DIR}/.task_status/${skill_name}.task2_status"
    fi
    
    # Task 4: Generate execution_report.md (works even without validation.md)
    if [[ -f "${validation_file}" ]]; then
        # With validation.md: use validation content
        TASK4_PROMPT="${validation_content}

重要：请将本任务的产物生成在 ./skill_check_report/${skill_name}/execution_report.md"
    else
        # Without validation.md: execute validation based on skill itself
        TASK4_PROMPT="对skill ${skill_name} 进行验证执行。

重要要求：
1. 根据skill定义，设计合理的验证场景和测试案例
2. 在目录 ./skill_check_report/${skill_name}/ 下生成名为 execution_report.md 的文件
3. execution_report.md应包含：验证执行过程、遇到的问题、验证结果等关键信息
4. 文件路径必须是：./skill_check_report/${skill_name}/execution_report.md
5. 不要在当前工作目录下直接生成文件"
    fi
    
    setsid "${OPENCODE_BIN}" run "${TASK4_PROMPT}" --model zhipuai-coding-plan/glm-5 > "${PYPTO_DIR}/task4_${skill_name}.log" 2>&1 &
    TASK4_PID=$!
    TASK4_PGID=$(ps -o pgid= -p $TASK4_PID | tr -d ' ')
    
    echo "  Task4 PID: $TASK4_PID (execution_report.md)"
    
    if [ "$TASK2_SKIPPED" = true ]; then
        echo "  Waiting for Task 4 to complete (timeout: 15 minutes)..."
    else
        echo "  Waiting for both tasks to complete (timeout: 15 minutes)..."
    fi
    
    TIMEOUT_SECONDS=900
    START_TIME=$(date +%s)
    TIMEOUT_OCCURRED=false
    
    while true; do
        TASK2_RUNNING=false
        TASK4_RUNNING=false
        
        # Check Task 2 status
        if [ "$TASK2_SKIPPED" = true ]; then
            TASK2_RUNNING=false
        else
            if kill -0 $TASK2_PID 2>/dev/null; then
                TASK2_RUNNING=true
            fi
        fi
        
        # Check Task 4 status (always runs)
        if kill -0 $TASK4_PID 2>/dev/null; then
            TASK4_RUNNING=true
        fi
        
        CURRENT_TIME=$(date +%s)
        ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
        
        if [ $ELAPSED_TIME -ge $TIMEOUT_SECONDS ]; then
            echo "  [TIMEOUT] Tasks exceeded 15 minutes for ${skill_name}"
            if [ "$TASK2_SKIPPED" = false ] && [ "$TASK2_RUNNING" = true ]; then
                echo "    Killing Task2 PGID $TASK2_PGID..."
                kill -9 -$TASK2_PGID 2>/dev/null || true
                wait $TASK2_PID 2>/dev/null || true
            fi
            if [ "$TASK4_RUNNING" = true ]; then
                echo "    Killing Task4 PGID $TASK4_PGID..."
                kill -9 -$TASK4_PGID 2>/dev/null || true
                wait $TASK4_PID 2>/dev/null || true
            fi
            TIMEOUT_OCCURRED=true
            break
        fi
        
        if [ "$TASK2_RUNNING" = false ] && [ "$TASK4_RUNNING" = false ]; then
            if [ "$TASK2_SKIPPED" = true ]; then
                echo "  ✓ Task 4 completed for ${skill_name} (${ELAPSED_TIME}s, Task 2 was skipped)"
            else
                echo "  ✓ Both tasks completed for ${skill_name} (${ELAPSED_TIME}s)"
            fi
            break
        fi
        
        if [ $((ELAPSED_TIME % 60)) -eq 0 ] && [ $ELAPSED_TIME -gt 0 ]; then
            if [ "$TASK2_SKIPPED" = true ]; then
                echo "  [Progress] ${skill_name}: ${ELAPSED_TIME}s elapsed, Task4: $TASK4_RUNNING (Task2 skipped)"
            else
                echo "  [Progress] ${skill_name}: ${ELAPSED_TIME}s elapsed, Task2: $TASK2_RUNNING, Task4: $TASK4_RUNNING"
            fi
        fi
        
        sleep 5
    done
    
    if [ "$TIMEOUT_OCCURRED" = true ]; then
        if [ "$TASK2_SKIPPED" = true ]; then
            echo "  [WARNING] ${skill_name} timeout (Task 2 was skipped, Task 4 timeout)"
        else
            echo "  [WARNING] Skipping ${skill_name} due to timeout"
        fi
        echo "TIMEOUT" > "${PYPTO_DIR}/.task_status/${skill_name}.status"
    else
        echo "SUCCESS" > "${PYPTO_DIR}/.task_status/${skill_name}.status"
    fi
    
    echo ""
done

echo ""
echo "========================================"
echo "Task 3: Summarizing reports..."
echo "========================================"
echo "  Starting Task 3 (timeout: 30 minutes)..."
START_TIME=$(date +%s)

timeout 1800 "${OPENCODE_BIN}" run "对./skill_check_report下各个skill的两个检查报告(execution_report.md和review.md)进行如下操作：汇总信息 -> 归纳优化点 -> 优化点去重 -> 优化点重要性排序 -> 优化点修改建议生成 -> 生成清单md文件。

重要说明：部分skill可能因超时或其他原因导致报告文件缺失（execution_report.md或review.md），这没关系。请先扫描./skill_check_report下各个skill目录，检查哪些报告文件存在，哪些缺失。对于缺失的报告，在清单md文件中单独标注为'报告缺失'即可，不要报错或中断处理。只对实际存在的报告文件进行汇总和分析。" --model zhipuai-coding-plan/glm-5

TASK3_EXIT_CODE=$?
ELAPSED_TIME=$(( $(date +%s) - START_TIME ))

if [ $TASK3_EXIT_CODE -eq 124 ]; then
    echo "  [TIMEOUT] Task 3 exceeded 30 minutes, terminated"
elif [ $TASK3_EXIT_CODE -ne 0 ]; then
    echo "  [ERROR] Task 3 failed with exit code $TASK3_EXIT_CODE"
else
    echo "  ✓ Task 3 completed successfully in ${ELAPSED_TIME}s"
fi

echo ""
echo "========================================"
echo "All tasks completed"
echo "========================================"
echo "  Work directory: ${PYPTO_DIR}"
echo "  Log files: ${PYPTO_DIR}/*.log"
echo "  Reports: ${PYPTO_DIR}/skill_check_report/"
echo "  Status: ${PYPTO_DIR}/.task_status/"
echo "========================================"