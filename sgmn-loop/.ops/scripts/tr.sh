#!/bin/bash
# SGMN loop · 到 trainer 的标准入口（Cycle 189 起）
#
# 背景（DEF-189-01）：本机 FlClash 代理的 TUN 策略路由（table 2022）把**所有**外网目的地
#   改走 `198.18.0.2 dev FlClash`，而其上游节点已死 ⇒ 任何 TCP 建连后立刻被切断
#   （SSH: kex 阶段被关；HTTPS: SSL unexpected eof）。ICMP 不受影响 ⇒ ping 一切正常，
#   极易误判成「远端机器挂了」。Cycle 186-188 正是据此**误判 trainer 宕机**。
#
# 正确做法：仅对**外层一跳**（bastion）绑定真实出口源地址 10.29.221.115（wlp1s0），
#   内核会据此选中 `via 10.29.0.1 dev wlp1s0` 而**绕开** FlClash 策略表；
#   内层一跳（bastion → trainer 10.0.140.43）在 AWS VPC 内，不经过本机代理。
#   **不改系统路由、不动用户代理、无需 root**，只作用于本次连接。
#
# 注意：源地址随 WiFi 重连可能变化 ⇒ 失效时用 `ip route show default` 取新的 src 填入。
SRC="${SGMN_LAN_SRC:-10.29.221.115}"
BASTION_IP="${SGMN_BASTION_IP:-13.238.13.194}"
exec ssh -b "$SRC" \
  -i "$HOME/.ssh/Bastion.pem" \
  -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new \
  -W "$1:22" "ec2-user@$BASTION_IP"
