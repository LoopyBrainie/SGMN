#!/bin/bash
# 绕过 FlClash 的策略路由（table 2022）—— 仅在**本次连接**上绑定真实出口源地址。
# 不改任何系统路由 / 不 kill 用户的代理进程 / 无需 root。
exec ssh -b 10.29.221.115 \
  -i "$HOME/.ssh/Bastion.pem" \
  -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new \
  -W "$1:22" ec2-user@13.238.13.194
