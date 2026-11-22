#!/bin/bash
# DIR-61 正式轮（rev 1.1）：Stage A 已 8/8 PASS 之后才允许执行本脚本
set -e
cd /home/ec2-user/sgmn-loop
P=/opt/pytorch/bin/python
$P dir61/R1_init61.py --out dir61/out/DIR61r11_formal_init.json
for v in d2r1 d4r1 d4r2; do
  $P dir61/esc61.py --variant $v --out dir61/out/DIR61r11_formal_esc_$v.json | tail -2
done
