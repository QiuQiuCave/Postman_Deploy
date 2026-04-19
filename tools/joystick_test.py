"""
按下手柄任意按键/摇杆，打印对应编号和 JoystickButton 枚举名（若有匹配）。
用法：python tools/joystick_test.py
退出：Ctrl+C
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pygame
from common.joystick import JoystickButton

BUTTON_NAMES = {b.value: b.name for b in JoystickButton}

AXIS_DEADZONE = 0.2   # 摇杆偏移超过此值才打印，避免噪声刷屏
HAT_IGNORE = (0, 0)   # 十字键归零不打印


def main():
    pygame.init()
    pygame.joystick.init()

    count = pygame.joystick.get_count()
    if count == 0:
        print("未检测到手柄，请先接好并执行 sudo modprobe xpad")
        sys.exit(1)

    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"已连接手柄：{js.get_name()}")
    print(f"  按键数：{js.get_numbuttons()}  摇杆轴数：{js.get_numaxes()}  hat 数：{js.get_numhats()}")
    print("-" * 50)
    print("按下按键 / 拨动摇杆 / 十字键，将实时打印编号。Ctrl+C 退出。")
    print("-" * 50)

    prev_axes = [0.0] * js.get_numaxes()
    prev_buttons = [0] * js.get_numbuttons()
    prev_hats = [(0, 0)] * js.get_numhats()

    try:
        while True:
            pygame.event.pump()

            # 按键：按下时打印（边沿触发）
            for i in range(js.get_numbuttons()):
                cur = js.get_button(i)
                if cur and not prev_buttons[i]:
                    name = BUTTON_NAMES.get(i, "未知")
                    print(f"[按键按下]  button_id={i:2d}  名称={name}")
                prev_buttons[i] = cur

            # 摇杆轴：超过死区时打印当前值
            for i in range(js.get_numaxes()):
                cur = js.get_axis(i)
                if abs(cur) > AXIS_DEADZONE and abs(cur - prev_axes[i]) > 0.05:
                    print(f"[摇杆]      axis_id={i:2d}  值={cur:+.3f}")
                prev_axes[i] = cur

            # 十字键 hat
            for i in range(js.get_numhats()):
                cur = js.get_hat(i)
                if cur != HAT_IGNORE and cur != prev_hats[i]:
                    print(f"[十字键]    hat_id ={i:2d}  方向={cur}")
                prev_hats[i] = cur

            pygame.time.wait(16)   # ~60 fps

    except KeyboardInterrupt:
        print("\n退出。")


if __name__ == "__main__":
    main()
