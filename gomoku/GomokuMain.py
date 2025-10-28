import numpy as np
import os
import time
from termcolor import colored
import keyboard


class GomokuGame:
    def __init__(self, board_size=15, win_length=5, show=False):
        self.board_size = board_size
        self.win_length = win_length
        self.board = np.zeros(
            (board_size, board_size), dtype=int
        )  # 0: empty, 1: player1, -1: player2
        self.current_player = 1  # Player 1 starts (you can use 1 and -1 for simplicity)
        self.game_over = False
        self.winner = None
        self.show = show
        self.move_history = []

    def is_valid_move(self, x, y):
        """检查落子是否合法"""
        return (
            0 <= x < self.board_size
            and 0 <= y < self.board_size
            and self.board[y, x] == 0
        )

    def step(self, action):
        """
        执行一步动作（落子）
        action: (x, y) 坐标
        返回: (done, winner)
        """
        if self.game_over:
            return True, self.winner

        x, y = action
        if not self.is_valid_move(x, y):
            # 非法落子，可视为输（或忽略，这里选择判负）
            self.game_over = True
            self.winner = -self.current_player
            return True, self.winner

        # 落子
        self.board[y, x] = self.current_player
        self.move_history.append((x, y))

        # 检查是否获胜
        if self._check_win(x, y):
            self.game_over = True
            self.winner = self.current_player
        elif len(self.move_history) == self.board_size * self.board_size:
            # 棋盘填满，平局
            self.game_over = True
            self.winner = 0  # 0 表示平局

        # 切换玩家
        self.current_player *= -1
        return self.game_over, self.winner

    def _check_win(self, x, y):
        """检查在 (x, y) 落子后是否形成五连"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 横、竖、正斜、反斜
        player = self.board[y, x]

        for dx, dy in directions:
            count = 1  # 包含当前子

            # 正方向
            nx, ny = x + dx, y + dy
            while (
                0 <= nx < self.board_size
                and 0 <= ny < self.board_size
                and self.board[ny, nx] == player
            ):
                count += 1
                nx += dx
                ny += dy

            # 反方向
            nx, ny = x - dx, y - dy
            while (
                0 <= nx < self.board_size
                and 0 <= ny < self.board_size
                and self.board[ny, nx] == player
            ):
                count += 1
                nx -= dx
                ny -= dy

            if count >= self.win_length:
                return True
        return False

    def get_state(self):
        """返回当前棋盘状态（NumPy 数组）"""
        return self.board.copy()

    def get_valid_actions(self):
        """返回所有合法落子位置列表 [(x, y), ...]"""
        empty = np.argwhere(self.board == 0)
        return [(x, y) for y, x in empty]  # 注意：argwhere 返回 [row, col] 即 [y, x]

    def render(self, speed=0.1, other_info=None):
        """渲染棋盘，坐标最多支持两位数（board_size <= 100）"""
        if not self.show:
            return

        if self.board_size > 100:
            raise ValueError(
                "Board size must be <= 100 to support two-digit coordinates."
            )

        os.system("clear" if os.name == "posix" else "cls")

        # 顶部列号：两位宽，右对齐
        print("   ", end="")  # 行号占2字符 + 1空格
        for j in range(self.board_size):
            print(f"{j:2d}", end=" ")
        print()

        # 棋盘内容
        for i in range(self.board_size):
            print(f"{i:2d} ", end="")  # 行号两位
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    print(colored("●", "red"), end="  ")
                elif self.board[i, j] == -1:
                    print(colored("○", "blue"), end="  ")
                else:
                    print(".", end="  ")
            print()

        # 游戏状态
        if self.game_over:
            if self.winner == 1:
                status = "🔴 Player 1 Wins!"
            elif self.winner == -1:
                status = "🔵 Player 2 Wins!"
            else:
                status = "Draw!"
        else:
            status = f"Player {'●' if self.current_player == 1 else '○'}'s turn"

        print(colored(status, "green"))

        if other_info:
            info_str = ", ".join(f"{k}: {v}" for k, v in other_info.items())
            print(colored(info_str, "cyan"))

        time.sleep(speed)

    def reset(self):
        """重置游戏"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []

    def play_with_keyboard(self, speed=0.1):
        """
        使用键盘控制光标移动并落子：
        - 方向键：上下左右移动光标
        - Enter：在当前位置落子（若合法）
        - 'r'：重置游戏
        - 'q'：退出
        """
        if not self.show:
            print("Warning: show=False, but keyboard play requires rendering.")
            self.show = True

        # 初始化光标位置（居中）
        cursor_x, cursor_y = self.board_size // 2, self.board_size // 2

        try:
            while not self.game_over:
                # 渲染时高亮光标位置
                self._render_with_cursor(cursor_x, cursor_y, speed=speed)

                # 等待按键
                event = keyboard.read_event()
                if event.event_type == keyboard.KEY_DOWN:
                    key = event.name

                    if key == "up" and cursor_y > 0:
                        cursor_y -= 1
                    elif key == "down" and cursor_y < self.board_size - 1:
                        cursor_y += 1
                    elif key == "left" and cursor_x > 0:
                        cursor_x -= 1
                    elif key == "right" and cursor_x < self.board_size - 1:
                        cursor_x += 1
                    elif key == "enter":
                        # 尝试落子
                        if self.is_valid_move(cursor_x, cursor_y):
                            self.step((cursor_x, cursor_y))
                            # 落子后光标可留在原地，或自动居中（这里保留原位）
                        # else: 无效位置，什么都不做
                    elif key == "r":
                        self.reset()
                        cursor_x, cursor_y = self.board_size // 2, self.board_size // 2
                    elif key == "q":
                        print("Quit game.")
                        return

            # 游戏结束后再渲染一次
            self._render_with_cursor(cursor_x, cursor_y, speed=0)

            # 等待任意键退出
            print("Press any key to exit...")
            keyboard.read_event()

        except KeyboardInterrupt:
            print("\nGame interrupted.")

    def _render_with_cursor(self, cursor_x, cursor_y, speed=0.1):
        """带光标高亮的渲染"""
        if not self.show:
            return

        os.system("clear" if os.name == "posix" else "cls")

        # 顶部列号
        print("   ", end="")
        for j in range(self.board_size):
            print(f"{j:2d}", end=" ")
        print()

        for i in range(self.board_size):
            print(f"{i:2d} ", end="")
            for j in range(self.board_size):
                cell = self.board[i, j]
                is_cursor = j == cursor_x and i == cursor_y

                if cell == 1:
                    symbol = colored("●", "red")
                elif cell == -1:
                    symbol = colored("○", "blue")
                else:
                    symbol = "."

                # 高亮光标位置：用 [] 包裹，或加背景色（这里用 []）
                if is_cursor:
                    print(f"[{symbol}]", end=" ")
                else:
                    print(f" {symbol} ", end="")
            print()

        # 游戏状态
        if self.game_over:
            if self.winner == 1:
                status = "🔴 Player 1 Wins!"
            elif self.winner == -1:
                status = "🔵 Player 2 Wins!"
            else:
                status = "Draw!"
        else:
            status = f"Player {'●' if self.current_player == 1 else '○'}'s turn (use arrows + Enter)"

        print(colored(status, "green"))
        print(colored("↑↓←→: move, Enter: place, R: reset, Q: quit", "yellow"))

        time.sleep(speed)


if __name__ == "__main__":
    """
    uv run gomoku/GomokuMain.py
    """

    game = GomokuGame(board_size=15, win_length=5, show=True)

    # # 简单对弈（手动输入或AI）
    # while not game.game_over:
    #     game.render()
    #     try:
    #         x = int(input("Enter x: "))
    #         y = int(input("Enter y: "))
    #     except:
    #         print("Invalid input!")
    #         continue

    #     done, winner = game.step((x, y))

    # game.render()
    # print(
    #     "Final result:",
    #     "Player 1 wins" if winner == 1 else "Player 2 wins" if winner == -1 else "Draw",
    # )

    game.play_with_keyboard(speed=0.05)
