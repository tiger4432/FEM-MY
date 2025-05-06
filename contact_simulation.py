import numpy as np
from skfem import *
from skfem.helpers import *
from skfem.models.elasticity import linear_elasticity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# 두 구불구불한 면을 생성하는 함수
def create_rough_surface(size, amplitude, frequency):
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    Z = amplitude * np.sin(2 * np.pi * frequency * X) * np.sin(2 * np.pi * frequency * Y)
    return X, Y, Z

# 3D 메시 생성
def create_mesh(size, amplitude, frequency, thickness=0.1):
    X, Y, Z = create_rough_surface(size, amplitude, frequency)
    
    # 3D 점 생성
    points = []
    for z in [0, thickness]:  # 두 개의 평면
        for i in range(size):
            for j in range(size):
                points.append([X[i,j], Y[i,j], Z[i,j] + z])
    
    points = np.array(points)
    
    # 테트라헤드론 연결성 생성
    tets = []
    n = size * size  # 한 평면의 점 개수
    
    for i in range(size-1):
        for j in range(size-1):
            # 아래 평면의 정사각형
            v1 = i*size + j
            v2 = i*size + j + 1
            v3 = (i+1)*size + j
            v4 = (i+1)*size + j + 1
            
            # 위 평면의 정사각형
            v5 = v1 + n
            v6 = v2 + n
            v7 = v3 + n
            v8 = v4 + n
            
            # 테트라헤드론 생성
            tets.extend([
                [v1, v2, v3, v5],
                [v2, v3, v4, v6],
                [v3, v4, v7, v6],
                [v5, v6, v7, v8]
            ])
    
    return MeshTet(points.T, np.array(tets).T)

# Contact 조건 검사
def check_contact(mesh1, mesh2, tol=1e-6):
    contact_nodes = []
    for i in range(mesh1.p.shape[1]):
        for j in range(mesh2.p.shape[1]):
            if abs(mesh1.p[0,i] - mesh2.p[0,j]) < tol and \
               abs(mesh1.p[1,i] - mesh2.p[1,j]) < tol and \
               abs(mesh1.p[2,i] - mesh2.p[2,j]) < tol:
                contact_nodes.append((i, j))
    return contact_nodes

# Contact 강성 행렬 생성
def create_contact_stiffness(contact_nodes, k_contact):
    n_nodes = len(contact_nodes)
    K_contact = csr_matrix((n_nodes, n_nodes))
    for i, (node1, node2) in enumerate(contact_nodes):
        K_contact[i, i] = k_contact
    return K_contact

# 메인 시뮬레이션 함수
def run_contact_simulation():
    # 파라미터 설정
    size = 5  # 더 작은 크기로 변경
    amplitude = 0.1
    frequency = 2
    thickness = 0.1
    k_contact = 1e6  # Contact 강성
    
    # 두 개의 구불구불한 면 생성
    mesh1 = create_mesh(size, amplitude, frequency, thickness)
    mesh2 = create_mesh(size, amplitude, frequency, thickness)
    
    # 두 번째 면을 약간 위로 이동
    mesh2.p[2] += 0.5
    
    # 재료 특성 설정
    E = 210e9  # Young's modulus (Pa)
    nu = 0.3   # Poisson's ratio
    
    # 요소와 공간 설정
    element = ElementTetP1()
    basis1 = Basis(mesh1, element)
    basis2 = Basis(mesh2, element)
    
    # 강성 행렬 계산
    @BilinearForm
    def bilin(u, v, w):
        eps_u = sym_grad(u)
        eps_v = sym_grad(v)
        return 2.0 * mu * ddot(eps_u, eps_v) + lmbda * tr(eps_u) * tr(eps_v)
    
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    K1 = asm(bilin, basis1)
    K2 = asm(bilin, basis2)
    
    # Contact 조건 검사
    contact_nodes = check_contact(mesh1, mesh2)
    
    # Contact 강성 행렬 생성
    K_contact = create_contact_stiffness(contact_nodes, k_contact)
    
    # 전체 강성 행렬 조립
    K_total = csr_matrix((K1.shape[0] + K2.shape[0], K1.shape[1] + K2.shape[1]))
    K_total[:K1.shape[0], :K1.shape[1]] = K1
    K_total[K1.shape[0]:, K1.shape[1]:] = K2
    
    # Contact 강성 추가
    for i, (node1, node2) in enumerate(contact_nodes):
        K_total[node1, node2] += k_contact
        K_total[node2, node1] += k_contact
    
    # 하중 벡터 생성
    F = np.zeros(K_total.shape[0])
    F[2::3] = -1.0  # z 방향 하중
    
    # 변위 계산
    u = spsolve(K_total, F)
    
    # 변위를 메시에 적용
    mesh1.p += u[:len(mesh1.p[0])].reshape(-1, 3).T
    mesh2.p += u[len(mesh1.p[0]):].reshape(-1, 3).T
    
    # 시각화
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 첫 번째 면
    ax.plot_trisurf(mesh1.p[0], mesh1.p[1], mesh1.p[2], 
                    triangles=mesh1.t.T, alpha=0.5)
    
    # 두 번째 면
    ax.plot_trisurf(mesh2.p[0], mesh2.p[1], mesh2.p[2], 
                    triangles=mesh2.t.T, alpha=0.5)
    
    # Contact 노드 표시
    for node1, node2 in contact_nodes:
        ax.plot([mesh1.p[0,node1], mesh2.p[0,node2]],
                [mesh1.p[1,node1], mesh2.p[1,node2]],
                [mesh1.p[2,node1], mesh2.p[2,node2]], 'r-')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Contact Simulation between Two Rough Surfaces')
    plt.show()

if __name__ == "__main__":
    run_contact_simulation()