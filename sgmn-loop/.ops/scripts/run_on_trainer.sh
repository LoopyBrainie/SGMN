#!/bin/bash
# 用法: run_on_trainer.sh '<远端命令>'    在 trainer 上执行（自动走正确的 SSH 路由）
exec ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new \
  -o ProxyCommand="$HOME/.sandbox/experiment/1107/sgmn-loop/.ops/scripts/tr.sh %h" \
  -i "$HOME/.ssh/Training.pem" -n \
  ec2-user@trainer.loopybrainie.aws "${1:?需要远端命令}"
