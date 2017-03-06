import random
import copy
from libc cimport math

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import numpy as np
cimport numpy as np

from scipy import ndimage
from scipy.misc import toimage, imresize

from skimage.measure import approximate_polygon
from skimage.draw import line
from skimage import morphology

cimport cython

ctypedef np.double_t DTYPE_t


Tw = 30 / 2
Th = 56 / 2
tolerance = 2
a1 = 8.0 / math.M_PI
a2 = 2.0 / min(Tw, Th)
a3 = 0.5


def get_points(rr):
    res1 = []
    res2 = []
    for i, row in enumerate(rr):
        for j, col in enumerate(row):
            if i % 2 == 0 and j % 2 == 0:
                res2.append((j, i))
            if col:
                res1.append((j, i))
    return res1, res2


def logspace(d1, d2, n):
    sp = [(10 ** (d1 + k * (d2 - d1) / (n - 1))) for k in xrange(0, n - 1)]
    sp.append(10 ** d2)
    return sp


def euclid_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def get_angle(p1, p2):
    return math.atan2((p2[1] - p1[1]), (p2[0] - p1[0]))


def get_angle_2(v1, v2):
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)


class SC(object):
    def __init__(self, nbins_r=5, nbins_theta=12, r_inner=0.5, r_outer=2.0):
        self.nbins_r = nbins_r
        self.nbins_theta = nbins_theta
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.nbins = nbins_theta * nbins_r

    def _dist2(self, x, c):
        cdef int i, j
        cdef int l1=len(x)
        cdef int l2=len(c)
        cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros((l1, l2), dtype=np.double)
        for i in range(l1):
            for j in range(l2):
                result[i, j] = euclid_distance(x[i], c[j])
        return result

    def _get_angles2(self, x, y):
        cdef int i, j
        cdef int l1=len(x)
        cdef int l2=len(y)
        cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros((l1, l2), dtype=np.double)
        for i in range(l1):
            for j in range(l2):
                result[i, j] = get_angle(x[i], y[j])
        return result

    # TODO this is still very slow, boolean type is not implemented in cython numpy
    @cython.boundscheck(False)
    def compute(self, points, r=None):
        points1, points2 = points
        cdef np.ndarray[DTYPE_t, ndim=2] r_array = self._dist2(points2, points1)
        # mean_dist = r_array.mean()
        # r_array_n = r_array / mean_dist
        cdef np.ndarray[DTYPE_t, ndim=2] r_array_n = r_array
        r_bin_edges = logspace(np.log10(self.r_inner), np.log10(self.r_outer), self.nbins_r)
        cdef np.ndarray[np.int_t, ndim=2] r_array_q = np.zeros((len(points2), len(points1)), dtype=np.int)
        cdef int m, i, j, tq,rq
        cdef int l1 = len(points1)
        cdef int l2 = len(points2)
        for m in range(self.nbins_r):
            r_array_q += (r_array_n < r_bin_edges[m])
        fz = r_array_q > 0
        cdef np.ndarray[DTYPE_t, ndim=2] theta_array = self._get_angles2(points2, points1)
        # 2Pi shifted
        cdef np.ndarray[DTYPE_t, ndim=2] theta_array_2 = theta_array + 2 * math.M_PI * (theta_array < 0)
        cdef np.ndarray[DTYPE_t, ndim=2] theta_array_q = 1 + np.floor(theta_array_2 / (2 * math.M_PI / self.nbins_theta))
        cdef np.ndarray[np.int_t, ndim=2] BH = np.zeros((len(points2), self.nbins), dtype=np.int)
        cdef np.ndarray[np.int_t, ndim=2] sn = np.zeros((self.nbins_r, self.nbins_theta), dtype=np.int)
        for i in range(l2):
            sn = np.zeros((self.nbins_r, self.nbins_theta), dtype=np.int)
            for j in range(l1):
                if fz[i, j]:
                    sn[r_array_q[i, j] - 1, theta_array_q[i, j] - 1] += 1
            BH[i] = sn.reshape(self.nbins)
        return BH, theta_array_2, r_array_n


def get_neighbors(spot, array, cur):
    neighbors = []
    extra_neighbors = []
    for row_offset, col_offset in [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
        pos = (spot[0] + row_offset, spot[1] + col_offset)
        if 0 <= pos[0] < array.shape[0] and 0 <= pos[1] < array.shape[1] and array[pos] == 1:
            neighbors.append(pos)

        if 0 <= pos[0] < array.shape[0] and 0 <= pos[1] < array.shape[1] and array[pos] == 3:
            extra_neighbors.append(pos)

    if not neighbors:
        neighbors = extra_neighbors
    return neighbors, len(neighbors)


def DFS(cur, total, pos, M, tol):
    cur_pos = pos
    if M[cur_pos] == 2 or M[cur_pos] == 3:
        return
    if len(cur) > 0:
        M[cur_pos] = 2
    else:
        M[cur_pos] = 3
    cur.append(cur_pos)
    neignbors, n = get_neighbors(cur_pos, M, cur)
    while n == 1:
        cur_pos = neignbors[0]
        if M[cur_pos] == 1:
            M[cur_pos] = 2
        cur.append(cur_pos)
        neignbors, n = get_neighbors(cur_pos, M, cur)

    if len(cur) > 1:
        simplified_points = approximate_polygon(np.array(cur), tolerance=tol)
        total.append(simplified_points)

    if n == 0:
        M[cur_pos] = 3
        return

    for i in range(n):
        M[neignbors[i]] = 4

    for i in range(n):
        cur = [cur_pos]
        DFS(cur, total, neignbors[i], M, tol)

# TODO improve circle line

def extract_segments(M):
    cur = []
    total = []
    blacks = np.where(M)
    for i, j in zip(*blacks):
        if M[i, j] == 1:
            cur = []
            DFS(cur, total, (i, j), M, tolerance)
    return total


def display(total):
    def p_equal(p1, p2):
        if p1[0,0] == p2[0,0] and p1[0,1] == p2[0,1] and p1[-1,0] == p2[-1,0] and p1[-1,1] == p2[-1,1] and len(p1) == len(p2):
            print p1
            print p2
            return True
        return False
    import matplotlib.pyplot as plt
    for t in total:
        plt.plot(t[:, 0], t[:, 1])
    plt.show()


def generate_polyimage(total, r):
    segments = []
    new_img = np.zeros(r.shape, dtype=bool)
    for t in total:
        cur = t[0]
        for tt in t[1:]:
            if tuple(cur) != tuple(tt):
                # if abs(cur[0] - tt[0]) > 1 or abs(cur[1] - tt[1]) > 1:
                rr, cc = line(cur[0], cur[1], tt[0], tt[1])
                segments.append([tuple(cur), tuple(tt)])
                new_img[rr, cc] = True
                cur = tt
    return new_img, segments


def load_img(ratio, name, Rw):
    whole = Image.open(name)
    w, h = whole.size
    alpha = Th * 1.0 / Tw
    Rh = int(math.ceil(h * 1.0 / (alpha * math.ceil(w * 1.0 / Rw))))
    width = Rw * Tw
    height = Rh * Th
    whole = whole.resize((width, height))
    img_data = np.asarray(whole)
    # img_data = imresize(img_data, (w,h))
    try:
        res = img_data[:, :, 0] < 255 * ratio
    except:
        res = img_data[:, :] < 255 * ratio
    r = morphology.skeletonize(res)
    # toimage(r).show()
    M = r.copy().astype(np.int64)
    total = extract_segments(M)
    display(total)
    return generate_polyimage(total, r), Rh


@cython.cdivision(True)
cdef line_intersection(float p0_x, float p0_y, float p1_x, float p1_y,
                           float p2_x, float p2_y, float p3_x, float p3_y):
    cdef float i_x, i_y, dx, dy
    cdef float s1_x, s1_y, s2_x, s2_y
    cdef float div
    s1_x = p0_x - p1_x
    s1_y = p0_y - p1_y
    s2_x = p2_x - p3_x
    s2_y = p2_y - p3_y
    # div = s1_x*s2_y - s1_y*s2_x
    # if div == 0:
    #     return -1, -1
    dx = p0_x*p1_y - p1_x*p0_y
    dy = p2_x*p3_y - p2_y*p3_x
    x = (dx*s2_x - dy*s1_x) / (s1_x*s2_y - s1_y*s2_x)
    y = (dx*s2_y - dy*s1_y) / (s1_x*s2_y - s1_y*s2_x)
    tx1 = min(p0_x, p1_x)
    tx2 = max(p0_x, p1_x)
    ty1 = min(p0_y, p1_y)
    ty2 = max(p0_y, p1_y)
    tx3 = min(p2_x, p3_x)
    tx4 = max(p2_x, p3_x)
    ty3 = min(p2_y, p3_y)
    ty4 = max(p2_y, p3_y)
    if tx1 <= x <= tx2 and ty1 <= y <= ty2 and tx3 <= x <= tx4 and ty3 <= y <= ty4:
        return x, y
    else:
        return -1, -1


def nearest_point(s, x3, y3):  # x3,y3 is the point
    x1, y1 = s[0]
    x2, y2 = s[1]
    px = x2 - x1
    py = y2 - y1
    something = px * px + py * py
    u = ((x3 - x1) * px + (y3 - y1) * py) / float(something)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px
    y = y1 + u * py
    return math.floor(x), int(y)


cdef nearest_points(segments, point, degree):
    points = []
    res = set()
    cdef int i
    cdef int j
    cdef float min_x
    for i in range(0, 360, degree):
        x = int(math.sin(i / 180.0 * math.M_PI) * 150) + point[0]
        y = int(math.cos(i / 180.0 * math.M_PI) * 150) + point[1]
        # line1 = (point, (x, y))
        min_x = 9999
        min_s = None
        for j in range(len(segments)):
            s = segments[j]
            x, y = line_intersection(point[0], point[1], x , y, s[0][0], s[0][1], s[1][0], s[1][1])
            if x < 0 and y < 0:
                continue
            if abs(x-point[0]) < min_x:
                min_x = abs(x-point[0])
                min_s = s
        if min_s:
            res.add(','.join(np.array(min_s, dtype=str).reshape(4)))
    res = [np.array(x.split(','), dtype=int).reshape(2, 2) for x in res]
    for r in res:
        n_point = nearest_point(r, point[0], point[1])
        if point[0] != n_point[0] or point[1] != n_point[1]:
            points.append(n_point)
    return points


def local_deform(A, B, C):
    V_theta = math.exp(math.fabs(a1 * get_angle_2((B[0] - A[0], B[1] - A[1]), (C[0] - A[0], C[1] - A[1]))))
    r1 = euclid_distance(A, B)
    r2 = euclid_distance(A, C)
    diff = math.exp(a2 * math.fabs(r1 - r2))
    prop = math.exp(a3 * max(r1, r2) / min(r1, r2))
    V_r = max(diff, prop)
    return max(V_theta, V_r)


def deform(segments, A, B, C):
    """
    A == C return -1
    """
    if A[0] == C[0] and A[1] == C[1]:
        return -1
    d_l = local_deform(A, B, C)
    Pa = (int(B[0] + A[0]) / 2, int(B[1] + A[1]) / 2)
    Pa_ = (int(C[0] + A[0]) / 2, int(C[1] + A[1]) / 2)
    l = []
    d_a = 0
    total_length = 0
    cdef int i
    points = nearest_points(segments, Pa, 10)
    for i, p in enumerate(points):
        if (Pa[0] == p[0] and Pa[1] == p[1]) or (Pa_[0] == p[0] and Pa_[1] == p[1]):
            return -1
        l.append(euclid_distance(Pa, p))
        total_length += l[-1]
    for i, p in enumerate(points):
        d_a += local_deform(p, Pa, Pa_) * l[i] / total_length
    return max(d_l, d_a)


def get_index_and_length(A, B):
    if B[1] < A[1] or (B[1] == A[1] and B[0] < A[0]):
        B, A = A, B
    d0 = B[0] - A[0]
    d1 = B[1] - A[1]
    def fy(x):
        return d1 * 1.0 / d0 * (x - B[0]) + B[1]
    def fx(y):
        return d0 * 1.0 / d1 * (y - B[1]) + B[0]
    dx = 1 if d0 >= 0 else -1
    dy = 1 if d1 >= 0 else -1
    start_i = math.floor(A[0] / Th)
    end_i = math.floor(B[0] / Th)
    start_j = math.floor(A[1] / Tw)
    end_j = math.floor(B[1] / Tw)
    if dy < 0:
        ly = start_j * Tw
    else:
        ly = start_j * Tw + Tw
    if dx < 0:
        lx = start_i * Th
    else:
        lx = start_i * Th + Th
    res = [(start_i, start_j)]
    cur = A
    l = []
    while abs(start_i) != abs(end_i) or abs(start_j) != abs(end_j):
        if d0 == 0:
            t0 = np.inf
        else:
            t0 = fy(lx)
        if d1 == 0:
            t1 = np.inf
        else:
            t1 = fx(ly)
        if t0 - ly >= 0 and dx * (t1 - lx) <= 0:
            start_j += dy
            l.append(euclid_distance((t1, ly), cur))
            cur = t1, ly
            ly += Tw * dy
            lasty = ly
        elif t0 - ly < 0 and dx * (t1 - lx) > 0:
            start_i += dx
            l.append(euclid_distance((lx, t0), cur))
            cur = lx, t0
            lx += Th * dx
        else:
            start_i += dx
            start_j += dy
            l.append(euclid_distance((lx, ly), cur))
            cur = lx, ly
            lx += Th * dx
            ly += Tw * dy
        res.append((start_i, start_j))
    l.append(euclid_distance(B, cur))
    # total_weight = sum(l)
    # l = [x / total_weight for x in l]
    for i,_ in enumerate(l):
        if _ <= 1:
            l.pop(i)
            res.pop(i)
    return res, l


def convert_segments_to_dictionary(segments):
    res = {}
    segments_hash = {}
    for index, seg in enumerate(segments):
        for i, t in enumerate(seg):
            if t in segments_hash:
                segments_hash[t].add((index, i))
            else:
                segments_hash[t] = set([(index, i)])
            for segment in segments:
                s, e = segment[0], segment[1]
                if t == s:
                    if t not in res:
                        res[t] = set()
                    res[t].add(e)
                if t == e:
                    if t not in res:
                        res[t] = set()
                    res[t].add(s)
    return res, segments_hash


def prepare_characters(Tw=15, Th=28, font_size=24, ratio=0.9, more_char=True):
    font = ImageFont.truetype('Menlo.ttc', size=font_size) # 24
    letters = {}
    xx = list(range(32, 127))
    if more_char:
        xx.extend(
            [915, 956, 1084, 1085, 1096, 8213, 8242, 8712, 8736, 8743, 8746, 8747, 8765, 8978, 9472, 9474, 9484, 9488, 9496,
             9500, 9508, 9524, 9581, 9582, 9583, 9584, 9585, 9586, 9621, 9651])

    for i in xx:
        img = Image.new("RGB", (Tw, Th), "white")
        d = ImageDraw.Draw(img)
        if i < 128:
            d.text((0, 0), chr(i), font=font, fill='black')
        else:
            d.text((0, 0), unichr(i), font=font, fill='black')
        # img = ndimage.gaussian_filter(img, 1))
        img_data = np.asarray(img)
        ratio = 0.9
        res = img_data[:, :, 0] < 255 * ratio
        rr = morphology.skeletonize(res)
        a = SC(r_outer=8)
        points = get_points(rr)
        result = a.compute(points)[0:2]
        letters[i] = [result, len(points[0])]
    letters.pop(64)
    return letters


def compute_aiss(r, letters):
    a = SC(r_outer=8)
    points = get_points(r)
    m1 = len(points[0])
    if m1 < 2:
        return 0, 0
    to_test = a.compute(points)[0:2]
    min_cost = 9999
    let = -1
    for k, v in letters.iteritems():
        cost = np.linalg.norm(to_test[0] - v[0][0], axis=1).sum() * 1.0 / (m1 + v[1])**0.5
        if cost < min_cost:
            min_cost = cost
            let = k
    return min_cost, let


def test_once(aiss, final, Rw, Rh, whole, letters):
    for j in range(Rh):
        sen = ''
        for i in range(Rw):
            img_data = whole[j * Th:j * Th + Th, i * Tw:i * Tw + Tw]
            # img_data = ndimage.gaussian_filter(temp, 2) # gaussian blur
            if len(np.where(img_data == 1)) == 0:
                sen += ' '
                continue
            min_cost, let = compute_aiss(img_data, letters)
            aiss[j, i] = min_cost
            final[j, i] = let
            if let == 0:
                chara = ' '
            else:
                if let > 128:
                    chara = unichr(let)
                else:
                    chara = chr(let)
            sen += chara
        print sen


def image_to_ascii(file_name='/Users/liyang/Desktop/monk_1.bmp', ratio=0.2, Rw=18, perform_deformation=True, more_char=True):
    # -----------------------------------------------
    # calculate letters log polar
    letters = prepare_characters(more_char=more_char)

    # -----------------------------------------------
    # load image and extract segments
    (whole, segments), Rh = load_img(ratio, file_name, Rw)
    toimage(whole).show()

    cdef int c=0
    cdef int c0=0
    cdef int rad = min(Tw, Th)
    cdef int w, h
    cdef double initial_aiss, E, cur_E
    w, h = whole.shape

    # -----------------------------------------------
    # build segments_dict and segments_hash
    segments_dict, segments_hash = convert_segments_to_dictionary(segments)

    aiss = np.zeros((Rh, Rw), dtype=float)
    final = np.zeros((Rh, Rw), dtype=int)

    # -----------------------------------------------
    # test convert without deformation
    test_once(aiss, final, Rw, Rh, whole, letters)

    # -----------------------------------------------
    # TODO WARNING:Currently deformation does not work properly or with an ultra slow speed!
    # TODO save many E and select min one
    if perform_deformation:
        initial_aiss = np.sum(aiss)
        E = initial_aiss/np.count_nonzero(aiss)
        while c0 <= 5000:
            D_cell = np.zeros((Rh, Rw), dtype=float)
            # backup data
            back_up_whole = copy.deepcopy(whole)
            back_up_aiss = copy.deepcopy(aiss)
            back_up_final = copy.deepcopy(final)
            back_up_segments_dict = copy.deepcopy(segments_dict)
            back_up_segments_hash = copy.deepcopy(segments_hash)
            back_up_segments = copy.deepcopy(segments)
            # back_up_D_cell = copy.deepcopy(D_cell)
            # pick a random point
            point = random.choice(segments_dict.keys())
            near_points = segments_dict[point]
            t = np.random.uniform(0.0, 2.0 * np.pi)
            r = int(np.random.uniform(2, rad))
            x = int(r * np.cos(t) + point[0])
            y = int(r * np.sin(t) + point[1])
            # need regenerate point?
            x = w-1 if x > w-1 else x if x > 0 else 0
            y = h-1 if y > h-1 else y if y > 0 else 0
            new_point = (x, y)
            # update segments_dict
            if new_point in segments_dict:
                segments_dict[new_point] = segments_dict[new_point] | segments_dict.pop(point)
            else:
                segments_dict[new_point] = segments_dict.pop(point)
            # regenerate image
            for p in near_points:
                segments_dict[p].remove(point)
                segments_dict[p].add(new_point)
                rr, cc = line(point[0], point[1], p[0], p[1])
                whole[rr, cc] = False
                rr, cc = line(new_point[0], new_point[1], p[0], p[1])
                whole[rr, cc] = True
            cur_chosen_cell = (math.floor(point[0] / Th), math.floor(point[1] / Tw))
            cur_chosen_cell_D = []
            computed_cells = []
            wrong_point_chosen = False
            for p in near_points:
                cur_chosen_cell_l = 0
                cells, l = get_index_and_length(p, point)
                total_l = sum(l)
                new_cells, rubbish = get_index_and_length(p, new_point)
                D = deform(segments, p, point, new_point)
                if D < 0:
                    wrong_point_chosen = True
                    break
                Dtotal = 0
                # recompute new character cells
                for cell in new_cells:
                    # recompute aiss each cell
                    img_data = whole[int(cell[0] * Th):int(cell[0] * Th + Th), int(cell[1] * Tw):int(cell[1] * Tw + Tw)]
                    aiss[cell], final[cell] = compute_aiss(img_data, letters)
                # compute every cell's deform
                for index, cell in enumerate(cells):
                    # do not compute the repeated cell again
                    if cell not in computed_cells:
                        if cell != cur_chosen_cell or cur_chosen_cell_l <= 0:
                            for s in segments:
                                t_cells, t_l = get_index_and_length(s[0], s[1])
                                for t_index, t_cell in enumerate(t_cells):
                                    if cell == t_cell:
                                        D_cell[cell] += t_l[t_index]
                            # recompute aiss each old cell
                            img_data = whole[int(cell[0] * Th):int(cell[0] * Th + Th), int(cell[1] * Tw):int(cell[1] * Tw + Tw)]
                            aiss[cell], final[cell] = compute_aiss(img_data, letters)
                        # record the chosen cell's total length and partial Deform value
                        if cell == cur_chosen_cell:
                            cur_chosen_cell_l = D_cell[cell]
                            cur_chosen_cell_D.append((l[index], D))
                            continue
                        D_cell[cell] = (D_cell[cell] - l[index] + l[index] * D) / D_cell[cell]
                        computed_cells.append(cell)
            if wrong_point_chosen:
                whole = back_up_whole
                aiss = back_up_aiss
                final = back_up_final
                segments_dict = back_up_segments_dict
                continue

            # try to fix the chosen point cell's correct Deform
            temp_d = 0
            for cell in cur_chosen_cell_D:
                temp_d += cell[0] * (cell[1] - 1)
            if cur_chosen_cell_l > 0:
                D_cell[cur_chosen_cell] = (cur_chosen_cell_l + temp_d) / cur_chosen_cell_l
            D_cell_new = np.ones((Rh, Rw), dtype=float)
            not_empty_cells = np.where(D_cell > 0)
            for i, j in zip(*not_empty_cells):
                D_cell_new[i, j] = D_cell[i, j]
            cur_E = np.sum(np.multiply(D_cell_new, aiss))/np.count_nonzero(aiss)
            # break
            diff = abs(cur_E - E)
            # if energy become less, we continue
            c += 1
            if cur_E < E:
                c0 = 0
                E = cur_E
                # update segments
                for position in segments_hash[point]:
                    segments[position[0]][position[1]] = np.array(new_point)
                # update segments_hash
                if new_point in segments_hash:
                    segments_hash[new_point] = segments_hash[new_point] | segments_hash.pop(point)
                else:
                    segments_hash[new_point] = segments_hash.pop(point)
                # toimage(whole).show()
                continue
            c0 += 1
            Pr = math.exp(-diff / (0.2 * initial_aiss * c**0.997))
            # accept the current deform
            if Pr < np.random.uniform(0.0, 1.0):
                # update segments
                E = cur_E
                for position in segments_hash[point]:
                    segments[position[0]][position[1]] = np.array(new_point)
                # update segments_hash
                if new_point in segments_hash:
                    segments_hash[new_point] = segments_hash[new_point] | segments_hash.pop(point)
                else:
                    segments_hash[new_point] = segments_hash.pop(point)
                # toimage(whole).show()
                continue
            else:
                # revert all data changed
                whole = back_up_whole
                aiss = back_up_aiss
                final = back_up_final
                segments_dict = back_up_segments_dict
                segments_hash = back_up_segments_hash
                segments = back_up_segments

        for j in range(Rh):
            sen = ''
            for i in range(Rw):
                ind = final[j, i]
                if ind == 0:
                    let = ' '
                else:
                    if ind > 128:
                        let = unichr(ind)
                    else:
                        let = chr(ind)
                sen += let
            print sen
        toimage(whole).show()