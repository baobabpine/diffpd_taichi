import taichi as ti
import taichi.math as tm
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

real = ti.f32
arch = ti.cpu
ti.init(arch=arch, default_fp=real, debug=True)

max_steps = 200
vis_interval = 256
output_vis_interval = 8
steps = 80

vis_resolution = 1024

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(3, dtype=real)
timing = lambda: ti.field(dtype=ti.i32)

loss = scalar()
grad_a = scalar()
grad_b = scalar()
x_ = vec()
v_ = vec()
x_simulate = vec()
store_t = timing()
# a = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
a = scalar()
b = scalar()

imgSize = 1024

clothWid = 4.0
clothHgt = 4.0
clothResX = 31

N = clothResX
NE = (N - 1)*(3*N-1)
num_triangles = (clothResX-1) * (clothResX-1) * 2
num_vertices = clothResX * clothResX
num_spring = NE
indices = ti.field(int, num_triangles * 3)
edgelist = ti.Vector.field(2, dtype=ti.i32, shape=NE)
edge = []
tem = ti.field(int, 3)
idx = ti.field(int, num_triangles * 3)
vertices = ti.Vector.field(3, float, clothResX * clothResX)

pos_pre = ti.Vector.field(3, dtype=ti.f32, shape=(clothResX , clothResX ))
pos = ti.Vector.field(3, dtype=ti.f32, shape=(clothResX , clothResX))
vel = ti.Vector.field(3, dtype=ti.f32, shape=(clothResX , clothResX ))
F = ti.Vector.field(3, dtype=ti.f32, shape=(clothResX , clothResX ))
J = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(clothResX , clothResX ))
fext = ti.Vector.field(3, dtype=ti.f32, shape=(clothResX , clothResX ))

spring = ti.Vector.field(2, ti.i32, NE)
pre_length = ti.field(ti.f32, NE)

KsStruct = ti.field(dtype=ti.f32, shape=())

# dm = ti.ndarray(ti.f32, 3 * num_spring)
dmr = ti.linalg.SparseMatrixBuilder(3 * num_spring, 1, max_num_triplets=1000000)
ym = ti.ndarray(ti.f32, 3 * num_vertices)
mhlr = ti.linalg.SparseMatrixBuilder(3 * num_vertices, 3 * num_vertices, max_num_triplets=1000000)
jmr = ti.linalg.SparseMatrixBuilder(3 * num_vertices, 3 * num_spring, max_num_triplets=1000000)
dproj_dxnew = ti.linalg.SparseMatrixBuilder(3 * num_spring, 3 * num_vertices, max_num_triplets=1000000)
res_right = ti.ndarray(ti.f32, 3*num_vertices)
Lx = ti.ndarray(ti.f32, 3 * num_vertices)

gravity = ti.Vector([0.0, -0.5, 0.0])
ball_centers = ti.Vector.field(3, float, 1)
ball_radius = ti.field(float, shape=(1))
deltaT = ti.field(float, shape=(1))

mass = 0.5
damping = -0.00125
# damping = 0.0

KsStruct = 10

gui = ti.ui.Window('Cloth', (imgSize, imgSize), vsync=True)

canvas = gui.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(20.0, 20.0, 20.0)
camera.lookat(0.0, 7.0, 0.0)
camera.up(0.0, 1.0, 0.0)

learning_rate = 2


def allocate_fields():
    ti.root.dense(ti.i, max_steps).dense(ti.j, num_vertices).place(x_, v_)
    ti.root.dense(ti.i, 1).dense(ti.j, num_vertices).place(x_simulate)
    ti.root.place(loss)
    ti.root.place(grad_a)
    ti.root.place(grad_b)
    ti.root.place(a)
    ti.root.place(b)
    ti.root.place(store_t)
    ti.root.lazy_grad()


@ti.func
def get_length3(v):
    return ti.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


@ti.func
def get_length2(v):
    return ti.sqrt(v.x * v.x + v.y * v.y)


@ti.kernel
def reset_cloth():
    for i, j in pos:
        pos[i, j] = ti.Vector(
            [clothWid * (i / clothResX) - clothWid / 2.0 - 8.0, 10.0, clothHgt * (j / clothResX) - clothHgt / 2.0])
        pos_pre[i, j] = pos[i, j]
        vel[i, j] = ti.Vector([0.0, 0.0, 0.0])
        F[i, j] = ti.Vector([0.0, 0.0, 0.0])

        if i < clothResX-1 and j < clothResX-1:
            tri_id = ((i * (clothResX - 1)) + j) * 2
            indices[tri_id * 3 + 2] = i * clothResX + j
            indices[tri_id * 3 + 1] = (i + 1) * clothResX + j
            indices[tri_id * 3 + 0] = i * clothResX + (j + 1)

            tri_id += 1
            indices[tri_id * 3 + 2] = (i + 1) * clothResX + j + 1
            indices[tri_id * 3 + 1] = i * clothResX + (j + 1)
            indices[tri_id * 3 + 0] = (i + 1) * clothResX + j

    ball_centers[0] = ti.Vector([0.0, 1.0, 0.0])
    ball_radius[0] = 1.0
    deltaT[0] = 0.5


@ti.kernel
def assemble_mhl(mhl: ti.types.sparse_matrix_builder()):
    for i in range(0, NE):
        idx1 = edgelist[i][0]
        idx2 = edgelist[i][1]
        for j in range(0, 3):
            mhl[3 * idx1 + j, 3 * idx1 + j] += KsStruct * deltaT[0] * deltaT[0]
            mhl[3 * idx1 + j, 3 * idx2 + j] -= KsStruct * deltaT[0] * deltaT[0]
            mhl[3 * idx2 + j, 3 * idx1 + j] -= KsStruct * deltaT[0] * deltaT[0]
            mhl[3 * idx2 + j, 3 * idx2 + j] += KsStruct * deltaT[0] * deltaT[0]
    for i in range(0, num_vertices):
        for j in range(0, 3):
            mhl[3 * i + j, 3 * i + j] += mass


@ti.kernel
def assemble_dm(testd: ti.types.sparse_matrix_builder()):
    for i in range(0, NE):
        idx1 = edgelist[i][0]
        idx2 = edgelist[i][1]
        coord0 = ti.Vector([int(idx1 / clothResX), idx1 % clothResX])
        coord1 = ti.Vector([int(idx2 / clothResX), idx2 % clothResX])
        p = pos[coord0] - pos[coord1]
        tmp = 0.0
        for k in range(0, 3):
            tmp += (pos[coord0][k] - pos[coord1][k]) * (pos[coord0][k] - pos[coord1][k])
        tmp = ti.sqrt(tmp)
        pn = tmp
        for j in range(0, 3):
            testd[3*i+j, 0] += pre_length[i] * p[j] * (1 / pn)


@ti.kernel
def assemble_d2(td: ti.types.sparse_matrix_builder(), x: ti.types.ndarray()):
    for i in range(0, NE):
        idx1 = edgelist[i][0]
        idx2 = edgelist[i][1]
        tmp = 0.0
        for j in range(0, 3):
            tmp += (x[3 * idx1 + j] - x[3 * idx2 + j]) * (x[3 * idx1 + j] - x[3 * idx2 + j])
        tmp = ti.sqrt(tmp)
        for j in range(0, 3):
            p = x[3 * idx1 + j] - x[3 * idx2 + j]
            td[3 * i + j, 0] += pre_length[i] * p * (1 / tmp)


@ti.kernel
def assemble_jm(testj: ti.types.sparse_matrix_builder()):
    for i in range(0, NE):
        idx1 = edgelist[i][0]
        idx2 = edgelist[i][1]
        for j in range(0, 3):
            testj[3 * idx1 + j, 3 * i + j] += KsStruct
            testj[3 * idx2 + j, 3 * i + j] -= KsStruct


@ti.kernel
def assemble_ym(testy: ti.types.ndarray()):
    for i in range(0, num_vertices):
        for j in range(0, 3):
            coord = ti.Vector([int(i / clothResX), i % clothResX])
            testy[3*i+j] = pos[coord][j] + deltaT[0] * vel[coord][j] + deltaT[0] * deltaT[0] * (1/mass) * F[coord][j]


@ti.kernel
def update_pos(x: ti.types.ndarray()):
    for i in range(0, num_vertices):
        y = int(i / clothResX)
        z = i % clothResX

        for j in range(0, 3):
            vel[y, z][j] = (x[3 * i + j] - pos[y, z][j]) / deltaT[0]
            pos[y, z][j] = x[3 * i + j]

        for ie in range(0, NE):
            idx1 = edgelist[ie][0]
            idx2 = edgelist[ie][1]
            tmp = 0.0
            p = pos[int(idx1 / clothResX), idx1 % clothResX] - pos[int(idx2 / clothResX), idx2 % clothResX]
            for k in range(0, 3):
                tmp += p[k] * p[k]
            tmp = ti.sqrt(tmp)
            if abs(tmp - pre_length[ie]) > 0.2:
                pos[int(idx2 / clothResX), idx2 % clothResX] = pos[int(idx1 / clothResX), idx1 % clothResX] - p.normalized() * 0.2

        y = int(i / clothResX)
        z = i % clothResX
        # collision
        offcet = pos[y, z] - ball_centers[0]
        dist = ti.sqrt(offcet.x * offcet.x + offcet.y * offcet.y + offcet.z * offcet.z)

        if dist < ball_radius[0]+0.07:
            delta0 = (ball_radius[0] - dist)+0.07
            pos[y, z] += (pos[y, z] - ball_centers[0]).normalized() * delta0
            pos_pre[y, z] = pos[y, z]
            vel[y, z] = [0.0, 0.0, 0.0]
            F[y, z] = [0.0, 0.0, 0.0]
        if pos[y, z].y < 1:
            offset = abs(pos[y, z].y-1)
            pos[y,z].y += offset
            vel[y, z] = [0.0, 0.0, 0.0]
            F[y, z] = [0.0, 0.0, 0.0]


def init_edges():
    for i in range(0, num_triangles):
        tem[0] = indices[3 * i]
        tem[1] = indices[3 * i + 1]
        tem[2] = indices[3 * i + 2]
        ti.algorithms.parallel_sort(tem, values=None)
        edge.append((tem[0], tem[2]))
        edge.append((tem[1], tem[2]))
        edge.append((tem[0], tem[1]))

    edge.sort()

    itr = 0
    if edge[0][0] != edge[1][0] or edge[0][1] != edge[1][1]:
        idx[itr] = 0
        itr = 1

    for i in range(1, num_triangles*3):
        if edge[i][0] != edge[i - 1][0] or edge[i][1] != edge[i - 1][1]:
            idx[itr] = i
            itr = itr + 1

    assert itr == NE

    for i in range(0, itr):
        edgelist[i][0] = edge[idx[i]][0]
        edgelist[i][1] = edge[idx[i]][1]
        idx1 = edgelist[i][0]
        idx2 = edgelist[i][1]
        tmp = 0.0
        p = pos[int(idx1 / clothResX), idx1 % clothResX]-pos[int(idx2 / clothResX), idx2 % clothResX]
        for k in range(0, 3):
            tmp += p[k]*p[k]
        tmp = ti.sqrt(tmp)
        pre_length[i] = tmp


@ti.kernel
def update_verts():
    for i, j in ti.ndrange(clothResX, clothResX):
        vertices[i * clothResX + j] = pos[i, j]


@ti.kernel
def calc_converge(xc: ti.types.ndarray(), x_prevc: ti.types.ndarray()) -> ti.f32:
    tmp = 0.0
    for i in range(3 * num_vertices):
        tmp += abs(xc[i]-x_prevc[i])
    tmp /= (3 * num_vertices)
    return tmp


def apply_pd():
    assemble_dm(dmr)
    dmb = dmr.build()
    assemble_ym(ym)

    x = ti.ndarray(ti.f32, 3 * num_vertices)
    tmp = 100000000
    count = 0
    prev_x = ti.ndarray(ti.f32, 3 * num_vertices)
    while tmp > 0.001 and count < 10:
        right = jmb @ dmb
        for i in range(num_vertices):
            for j in range(3):
                res_right[3 * i + j] = deltaT[0] * deltaT[0] * right[3 * i + j, 0] + mass * ym[3 * i + j]

        solver = ti.linalg.SparseSolver(solver_type="LDLT")
        solver.analyze_pattern(mhlb)
        solver.factorize(mhlb)
        x = solver.solve(res_right)
        assemble_d2(dmr, x)
        dmb = dmr.build()
        tmp = calc_converge(x, prev_x)
        prev_x = x
        count = count + 1

    update_pos(x)
    update_verts()


@ti.kernel
def time_integrate(t: ti.i32):
    for i in range(num_vertices):
        x_[t, i] = pos[int(i/clothResX), i % clothResX]
        v_[t, i] = vel[int(i/clothResX), i % clothResX]


@ti.kernel
def compute_loss(t: ti.i32):
    tmp = 0.0
    for i in range(num_vertices):
        dist = x_[t, i] - x_simulate[0, i]
        dn = get_length3(dist)
        tmp += dn
    tmp /= num_vertices
    loss[None] = tmp


@ti.kernel
def print_vert():
    for i, j in ti.ndrange(clothResX, clothResX):
        print(vertices[i*clothResX+j])


@ti.kernel
def print_pos():
    for i, j in ti.ndrange(3, 3):
        print(pos[i, j])


def visualize(output, t):

    scene.mesh(vertices, indices=indices, color=(0.5, 0.5, 0.6), two_sided=False, show_wireframe=False)
    scene.particles(ball_centers, radius=1, color=(0.5, 0.5, 0.8))

    scene.point_light(pos=(20.0, 20.0, 0.0), color=(1.0, 1.0, 1.0))
    camera.track_user_inputs(gui, movement_speed=0.03, hold_key=ti.ui.LMB)
    scene.set_camera(camera)

    canvas.scene(scene)
    gui.show()
    if output:
        gui.save_image('mass_spring_simple/{}/{:04d}.png'.format(output, t))


def forward(output=None):
    interval = vis_interval
    mid = int(clothResX / 2)
    count = 0

    if output:
        interval = output_vis_interval
        os.makedirs('mass_spring_simple/{}/'.format(output), exist_ok=True)

    for t in range(1, steps):  # steps
        init_f()

        apply_pd()
        time_integrate(t)

        # if (t + 1) % interval == 0:
        visualize(output, t)
        print(t)
        if pos[mid, mid][1] < 2.0:
            count += 1
        if count >= 8:
            break

    compute_loss(t)
    store_t[None] = t


@ti.kernel
def clear_states():
    for t in range(0, max_steps):
        for i in range(0, num_vertices):
            x_.grad[t, i] = ti.Vector([0.0, 0.0, 0.0])
            v_.grad[t, i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def reset_xvf():
    for i, j in pos:
        pos[i, j] = ti.Vector(
            [clothWid * (i / clothResX) - clothWid / 2.0-8.0, 10.0, clothHgt * (j / clothResX) - clothHgt / 2.0])
        pos_pre[i, j] = pos[i, j]
        vel[i, j] = ti.Vector([0.0, 0.0, 0.0])
        F[i, j] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def clear_ab():
    a.grad[None] = 0.0
    b.grad[None] = 0.0


def clear_tensors():
    clear_states()
    clear_ab()


@ti.kernel
def init_xv0():
    for ix in range(num_vertices):
        i = int(ix / clothResX)
        j = ix % clothResX
        x_[0, ix] = ti.Vector(
            [clothWid * (i / clothResX) - clothWid / 2.0-8.0, 10.0, clothHgt * (j / clothResX) - clothHgt / 2.0])
        v_[0, ix] = [0.0, 0.0, 0.0]
    a[None] = 0.0
    b[None] = 0.0


@ti.func
def compute_f(coord):
    F[coord] = gravity * mass + vel[coord] * damping + a[None]*ti.Vector([1.0, 0.0, 0.0]) + b[None]*ti.Vector([0.0, 0.0, 1.0])
    offcet = pos[coord] - ball_centers[0]
    dist = ti.sqrt(offcet.x * offcet.x + offcet.y * offcet.y + offcet.z * offcet.z)

    if dist < ball_radius[0] + 0.07:
        delta0 = (ball_radius[0] - dist) + 0.07
        pos[coord] += (pos[coord] - ball_centers[0]).normalized() * delta0
        pos_pre[coord] = pos[coord]
        vel[coord] = [0.0, 0.0, 0.0]
        F[coord] = [0.0, 0.0, 0.0]
    if pos[coord].y < 1:
        offset = abs(pos[coord].y - 1)
        pos[coord].y += offset
        vel[coord] = [0.0, 0.0, 0.0]
        F[coord] = [0.0, 0.0, 0.0]


@ti.kernel
def init_f():
    for i, j in pos:
        coord = ti.Vector([i, j])
        compute_f(coord)


@ti.kernel
def dproj_dx(dproj_dnew: ti.types.sparse_matrix_builder()):
    for i in range(0, NE):
        idx1 = edgelist[i][0]
        idx2 = edgelist[i][1]
        coord0 = ti.Vector([int(idx1 / clothResX), idx1 % clothResX])
        coord1 = ti.Vector([int(idx2 / clothResX), idx2 % clothResX])
        p = pos[coord0] - pos[coord1]
        i_three = ti.Matrix.identity(ti.f32, 3)
        d_posd_dx1 = i_three
        d_posd_dx2 = -i_three
        ln = get_length3(p)  # norm
        dirn = 1.0 / ln * p  # normalize
        # dirn_t = dirn.transpose()
        tpm = tm.mat3(0)
        for tmi in range(3):
            for tmj in range(3):
                tpm[tmi, tmj] = dirn[tmi] * dirn[tmj]
        ddir_dposdiff = (i_three - tpm) / ln
        dp_dx1 = pre_length[i] * ddir_dposdiff @ d_posd_dx1
        dp_dx2 = pre_length[i] * ddir_dposdiff @ d_posd_dx2
        for bi in range(3):
            for bj in range(3):
                dproj_dnew[3 * i + bi, 3 * idx1 + bj] += dp_dx1[bi, bj]
                dproj_dnew[3 * i + bi, 3 * idx2 + bj] += dp_dx2[bi, bj]


@ti.kernel
def dl_dx(dlx: ti.types.ndarray()):
    for i in range(num_vertices):
        dist = x_[store_t[None], i] - x_simulate[0, i]
        dn = get_length3(dist)
        for j in range(3):
            dlx[3 * i + j] = 1 / (dn * num_vertices) * dist[j]


@ti.kernel
def calc_g_ab(zn: ti.types.ndarray()):
    tma = 0.0
    tmb = 0.0
    for i in range(num_vertices):
        tma += zn[3 * i]
        tmb += zn[3 * i + 2]
        print("z:")
        print(zn[3*i], zn[3*i+1], zn[3*i+2])
    tma /= (num_vertices * deltaT[0] * deltaT[0])
    tmb /= (num_vertices * deltaT[0] * deltaT[0])
    grad_a[None] = tma
    grad_b[None] = tmb


@ti.kernel
def adapt_cloth():
    for i, j in pos:
        pos[i, j] = ti.Vector(
            [clothWid * (i / clothResX) - clothWid / 2.0, 3.0, clothHgt * (j / clothResX) - clothHgt / 2.0])
        pos_pre[i, j] = pos[i, j]


@ti.kernel
def store_simulate():
    for i in range(num_vertices):
        x_simulate[0, i] = pos[int(i/clothResX), i % clothResX]


def simulate():  # before forward()
    adapt_cloth()
    mid = int(clothResX / 2)
    count = 0
    for t in range(1, 11):  # steps
        init_f()
        apply_pd()
        visualize(0, t)
        if pos[mid, mid][1] < 2.0:
            count += 1
        if count >= 3:
            break
    store_simulate()


@ti.kernel
def ary_add(zd: ti.types.ndarray(), rzd: ti.types.ndarray(), rhsd: ti.types.ndarray()):
    for i in range(3 * num_vertices):
        rhsd[i] = zd[i] + rzd[i]


def backward():
    z = ti.ndarray(ti.f32, 3 * num_vertices)
    rhs = ti.ndarray(ti.f32, 3 * num_vertices)
    dproj_dx(dproj_dxnew)
    dpx = dproj_dxnew.build()
    dl_dx(Lx)

    for i in range(10):
        rz = deltaT[0] * deltaT[0] * dpx.transpose() @ jmb.transpose() @ z
        ary_add(rz, Lx, rhs)
        # rhs = deltaT[0] * deltaT[0] * dpx.transpose() @ jmb.transpose() @ z + Lx
        solver = ti.linalg.SparseSolver(solver_type="LDLT")
        solver.analyze_pattern(mhlb)
        solver.factorize(mhlb)
        z = solver.solve(rhs)

    # i = int(clothResX / 2)
    # xid = i * clothResX + i
    # (1.0/(deltaT[0]*deltaT[0])) * z[3 * xid]
    calc_g_ab(z)


def main():
    allocate_fields()
    init_xv0()
    clear_tensors()
    reset_cloth()
    init_edges()
    assemble_mhl(mhlr)
    global mhlb
    mhlb = mhlr.build()
    assemble_jm(jmr)
    global jmb
    jmb = jmr.build()

    simulate()
    clear_tensors()
    reset_xvf()

    forward('initial')

    losses = []
    for iteration in range(25):
        clear_tensors()
        reset_xvf()

        # with ti.ad.Tape(loss):
        forward()
        backward()

        print('Iter=', iteration, 'Loss=', loss[None])
        losses.append(loss[None])

        a[None] -= 5 * grad_a[None]
        b[None] -= 5 * grad_b[None]
        print(a[None], b[None], grad_a[None], grad_b[None], iteration)

    print(a[None], b[None])

    fig = plt.gcf()
    fig.set_size_inches(4, 3)

    plt.plot(losses)
    plt.title("Spring Rest Length Optimization")
    plt.xlabel("Gradient descent iterations")
    plt.ylabel("Loss")
    plt.tight_layout()

    plt.show()
    clear_tensors()
    reset_xvf()
    forward('final')


if __name__ == '__main__':
    main()
