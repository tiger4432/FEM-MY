import numpy as np
import matplotlib.pyplot as plt

def create_checkerboard(size):
    """체커보드 패턴 생성"""
    return np.indices((size, size)).sum(axis=0) % 2

def create_spiral(size):
    """나선형 패턴 생성"""
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (np.sin(r * 10 + theta * 5) > 0).astype(int)

def create_wave(size):
    """파동 패턴 생성"""
    x, y = np.meshgrid(np.linspace(0, 2*np.pi, size), np.linspace(0, 2*np.pi, size))
    return (np.sin(x) * np.cos(y) > 0).astype(int)

def create_maze(size):
    """미로 패턴 생성"""
    maze = np.ones((size, size))
    for i in range(1, size-1, 2):
        for j in range(1, size-1, 2):
            maze[i,j] = 0
            if np.random.random() > 0.5:
                maze[i+1,j] = 0
            else:
                maze[i,j+1] = 0
    return maze

def create_custom_pattern(size):
    """사용자 정의 패턴 생성"""
    pattern = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i + j) % 3 == 0 or (i - j) % 4 == 0:
                pattern[i,j] = 1
    return pattern

def plot_pattern(pattern, title):
    plt.figure(figsize=(8, 8))
    plt.imshow(pattern, cmap='binary')
    plt.title(title)
    plt.axis('off')
    plt.show()

# 패턴 생성 및 시각화
size = 50

patterns = [
    (create_checkerboard(size), "Checkerboard Pattern"),
    (create_spiral(size), "Spiral Pattern"),
    (create_wave(size), "Wave Pattern"),
    (create_maze(size), "Maze Pattern"),
    (create_custom_pattern(size), "Custom Pattern")
]

# 모든 패턴 시각화
for pattern, title in patterns:
    plot_pattern(pattern, title) 