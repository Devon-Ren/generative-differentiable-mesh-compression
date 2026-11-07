#!/bin/bash
SSH_CONFIG="$HOME/.ssh/config"
ACCOUNT="torch_pr_38_general"
HOST_ALIAS="cpu-torch"

# 12h 站位（轻量）
PARTITION="cpu_short"  # 若你必须用 cs，就改成 "cs"
CPUS=2                  # 连接/轻量调试足够
MEM="8G"
HOURS=6
SLEEP_SEC=$((HOURS*3600))

echo "🔐 检查 SSH 连接..."
if ! ssh -o ControlPath="$HOME/.ssh/ansible-%r@%h:%p" -O check torch 2>/dev/null; then
  echo "🔑 需要身份验证。请完成 MFA..."
  ssh torch "echo '✅ 认证成功，连接已建立'"
else
  echo "✅ 现有连接仍有效，跳过 MFA。"
fi

echo "🚀 正在申请 CPU 节点（${HOURS}h 站位，${CPUS} CPU, ${MEM}）..."
JOB_ID=$(ssh torch "sbatch \
  --account=$ACCOUNT \
  --partition=$PARTITION \
  --cpus-per-task=$CPUS \
  --mem=$MEM \
  --time=${HOURS}:00:00 \
  --job-name=${HOST_ALIAS} \
  --wrap 'sleep ${SLEEP_SEC}' \
  --parsable")

if [ -z "$JOB_ID" ]; then
  echo "❌ 提交失败，请检查账户名/分区: $ACCOUNT / $PARTITION"
  exit 1
fi

echo "✅ Job: $JOB_ID"
echo "⏳ 等待分配..."

while true; do
  INFO=$(ssh torch "squeue -j $JOB_ID -h -o '%T %N'" 2>/dev/null)
  STATE=$(echo "$INFO" | awk '{print $1}')
  NODE=$(echo "$INFO" | awk '{print $2}')

  [ -z "$STATE" ] && echo "❌ Job 不存在" && exit 1
  if [ "$STATE" = "RUNNING" ] && [ "$NODE" != "(null)" ]; then break; fi

  echo "   状态: $STATE..."
  sleep 3
done

echo "✅ 节点已分配: $NODE"

# 仅修改 Host cpu-torch 段落内 HostName（避免误伤其它 Host）
if [[ "$OSTYPE" == "darwin"* ]]; then
  sed -i '' -E "/^Host[[:space:]]+${HOST_ALIAS}\$/,/^Host[[:space:]]+/ s/^[[:space:]]*HostName[[:space:]]+.*/  HostName ${NODE}.hpc.nyu.edu/" "$SSH_CONFIG"
else
  sed -i -E "/^Host[[:space:]]+${HOST_ALIAS}\$/,/^Host[[:space:]]+/ s/^[[:space:]]*HostName[[:space:]]+.*/  HostName ${NODE}.hpc.nyu.edu/" "$SSH_CONFIG"
fi

echo "✅ 已更新 $SSH_CONFIG"
echo "🎉 VS Code 连接: ${HOST_ALIAS}"
echo "💡 结束站位: ssh torch 'scancel $JOB_ID'"
