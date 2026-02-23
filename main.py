import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

in_vertices = [
    (6, 8, 0),
    (5, 0, 2),
    (0, 5, 6),
    (46/7, 0, 0),
    (0, 29, 0),
    (0, 0, 19/3)
]

in_vert_a = (6, -5, 5)
in_vert_b = (-8, 8, 9)

"""

in_vertices = [
    (3, 3, 0),
    (5, 0, 3),
    (0, 6, 9),
    (41/8, 0, 0),
    (0, 13/2, 0),
    (0, 0, 36)
]

in_vert_a = (6, -9, 9)
in_vert_b = (-7, 8, 6)
"""

class vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return vec3(self.x*scalar, self.y*scalar, self.z*scalar)

    def __truediv__(self, scalar):
        return vec3(self.x/scalar, self.y/scalar, self.z/scalar)
    
    def __iter__(self):
        return iter([self.x, self.y, self.z])

def cross(v1, v2):
    result = vec3(
        v1.y*v2.z - v1.z*v2.y,
        v1.z*v2.x - v1.x*v2.z,
        v1.x*v2.y - v1.y*v2.x
    )
    return result

def length(v):
    return (v.x**2 + v.y**2 + v.z**2)**0.5

def normalize(v):
    return v / length(v)

def dot(v1, v2):
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z

def barycentric(a, b, c, normal, p):
    area_abc = abs(dot(normal, cross(b - a, c - a)))
    area_pbc = abs(dot(normal, cross(b - p, c - p)))
    area_pca = abs(dot(normal, cross(c - p, a - p)))
    area_pba = abs(dot(normal, cross(b - p, a - p)))

    k1 = area_pbc / area_abc
    k2 = area_pca / area_abc
    k3 = area_pba / area_abc
    return vec3(k1, k2, k3)

in_faces = [
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 4),
    (1, 2, 5)
]

kvd_signs = [
    [ 1,  1,  1],
    [-1,  1,  1],
    [ 1, -1,  1],
    [ 1,  1, -1],
    [-1, -1,  1],
    [-1,  1, -1],
    [ 1, -1, -1],
    [-1, -1, -1]
]

def tab_header(header):
    return "\\begin{tabular}{" + "|" + "|".join(["c"] * len(header)) + "|}" \
         + " \\hline" \
         + " \\rowcolor{blue!15}" \
         + " \\bfseries " + " & ".join(header) + " \\\\" \
         + " \\hline"

def str_table(vals, header, rows = None, trunc=0, split=1, row_colors=None):    
    if rows == None:
        rows = [str(i+1) for i in range(len(vals))]

    if split > 1:
        n = (len(vals) + split - 1) // split
        return '\n'.join([
            str_table(vals[n*i:n*(i+1)], header, rows[n*i:n*(i+1)], trunc, 1, row_colors[n*i:n*(i+1)] if row_colors else None)
            for i in range(split)
        ])
    
    copy_vals = vals[::1]
    if trunc > 0:
        for i in range(len(copy_vals)):
            copy_vals[i] = [round(x, trunc) for x in copy_vals[i]]

    lines = []
    lines.append("\\begingroup ")
    
    if len(header) >= 10: 
        lines.append("\\setlength{\\tabcolsep}{2pt}")
    else:
        lines.append("\\setlength{\\tabcolsep}{6pt}")
    
    lines.append(tab_header(header))
    for i in range(len(rows)):
        row_color = ""
        if row_colors and row_colors[i]:
            row_color = f"\\rowcolor{{{row_colors[i]}}}"
        
        row_color += " \\cellcolor{blue!5}"
        
        row = row_color + rows[i]
        row += " & " + " & ".join([str(copy_vals[i][j]) for j in range(len(header) - 1)]) + " \\\\"
        lines.append(row)

    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\endgroup")
    
    return "\n".join(lines)


vertices = []
for k in kvd_signs:
    for v in in_vertices:
        vertices.append((v[0]*k[0], v[1]*k[1], v[2]*k[2]))

unique_vertices = []

for v in vertices:
    if not v in unique_vertices:
        unique_vertices.append(v)

faces = []
for k in range(len(kvd_signs)):
    for f in in_faces:
        faces.append((f[0] + k*6, f[1] + k*6, f[2] + k*6))

class Plane:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

pl_normals = []
pl_points = []

planes = []
for f in faces:
    p1 = vec3(*vertices[f[0]])
    p2 = vec3(*vertices[f[1]])
    p3 = vec3(*vertices[f[2]])

    n = normalize(cross(p2 - p1, p3 - p1))
    if dot(n, p1) < 0:
        n = vec3(-n.x, -n.y, -n.z)

    pl_normals.append(n)
    pl_points.append((p1 + p2 + p3) / 3)
    
    planes.append(Plane(n.x, n.y, n.z, -dot(n, p1)))

plane_checks = []
for p in planes:
    row = []
    for v in unique_vertices:
        val = p.a*v[0] + p.b*v[1] + p.c*v[2] + p.d
        assert val <= 0
        row.append(val)
    plane_checks.append(row)

open("tex/plane_checks.gen.tex", "w").write(str_table(
    plane_checks,
    ["$g/v$"] + [str(i+1) for i in range(len(unique_vertices))],
    trunc=1,
))

open("tex/planes.gen.tex", "w").write(str_table(
    [[p.a, p.b, p.c, p.d] for p in planes],
    ["№", "$A$", "$B$", "$C$", "$D$"],
    trunc=4,
    split=2
))

open("tex/in_vertices.gen.tex", "w").write(str_table(
    in_vertices,
    ["№", "$x$", "$y$", "$z$"],
    trunc=4
))

open("tex/input_vertices.gen.tex", "w").write(str_table(
    in_vertices + [in_vert_a, in_vert_b],
    ["№", "$x$", "$y$", "$z$"],
    rows=["$v_1$", "$v_2$", "$v_3$", "$v_4$", "$v_5$", "$v_6$", "$a$", "$b$"],
    trunc=4
))

open("tex/input_faces.gen.tex", "w").write(str_table(
    [[f[0] + 1, f[1] + 1, f[2] + 1] for f in in_faces],
    ["№", "$v_1$", "$v_2$", "$v_3$"],
))

open("tex/all_faces.gen.tex", "w").write(str_table(
    [[f[0] + 1, f[1] + 1, f[2] + 1] for f in faces],
    ["№", "$v_1$", "$v_2$", "$v_3$"],
    rows=[f"$v_{{{idx+1}}}$" for idx in range(len(faces))],
    split=4
))

open("tex/unique_vertices.gen.tex", "w").write(str_table(
    unique_vertices,
    ["№", "$x$", "$y$", "$z$"],
    trunc=4,
    split=3
))

open("tex/all_vertices.gen.tex", "w").write(str_table(
    vertices,
    ["№", "$x$", "$y$", "$z$"],
    trunc=4,
    split=3
))

def image(callable, userdata, filename, elev, azim):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    callable(ax, userdata)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    filename2 = filename[::-1].replace(".", "_not_cropped."[::-1], 1)[::-1]
    plt.savefig(filename2, dpi=100)
    #plt.show()
    plt.close()

    os.system(f"convert {filename2} -trim +repage {filename}")
    

def draw_poly(ax, userdata):
    verts = np.array(userdata["verts"])
    faces = np.array(userdata["faces"])
    
    polys = [[verts[face[0]], verts[face[1]], verts[face[2]]] for face in faces]
    
    poly_collection = Poly3DCollection(polys, alpha=0.7, edgecolors='k', linewidths=0.5)
    poly_collection.set_facecolor((217/255, 217/255, 1))
    ax.add_collection3d(poly_collection)
    
def draw_poly_with_normals(ax, userdata):
    draw_poly(ax, userdata)

    for point, normal in zip(pl_points, pl_normals):
        end_point = point + normal * 1.0
        ax.plot(
            [point.x, end_point.x],
            [point.y, end_point.y],
            [point.z, end_point.z],
            color=(217/255, 0.2, 0.2), linewidth=2, zorder=10
        )

def draw_poly_barycentric(ax, userdata):
    draw_poly(ax, userdata)

    p = userdata["p"]
    v1 = userdata["v1"]

    ax.plot([0, v1.x], [0, v1.y], [0, v1.z], color="black", linewidth=1)
    ax.plot([v1.x, p.x], [v1.y, p.y], [v1.z, p.z], color="black", linewidth=1, zorder=10)

    ax.plot([p.x], [p.y], [p.z], color="black", marker="o", markersize=3, zorder=10)
    ax.plot([v1.x], [v1.y], [v1.z], color="red", marker="o", markersize=5, zorder=10)
    
image(draw_poly, {"verts": in_vertices, "faces": in_faces}, "tex/input_poly.png", 15, 30)
image(draw_poly, {"verts": vertices, "faces": faces}, "tex/poly.png", 15, 30)
image(draw_poly, {"verts": vertices, "faces": faces}, "tex/poly.png", 15, 30)

image(draw_poly_with_normals, {"verts": vertices, "faces": faces}, "tex/poly_normals.png", 0, 90)

def minkowski_norm_for_point(a, var_name):
    a = vec3(*a)

    bary_table = []
    projected_points = []
    
    for f_idx in range(len(planes)):
        n = pl_normals[f_idx]
        p1 = vec3(*pl_points[f_idx])
        
        dist_plane = dot(p1, n)
        dist_a = dot(a, n)

        if dist_a == 0:
            inf = float("inf")
            v = vec3(inf, inf, inf)
            projected_points.append(v)
            k = vec3(inf, inf, inf)
            bary_table.append([k.x, k.y, k.z, k.x + k.y + k.z, inf])
            
        else:
            v = a * (dist_plane / dist_a)
            projected_points.append(v)
            
            k = barycentric(
                vec3(*vertices[faces[f_idx][0]]),
                vec3(*vertices[faces[f_idx][1]]),
                vec3(*vertices[faces[f_idx][2]]),
                n, v
            )
            bary_table.append([k.x, k.y, k.z, k.x + k.y + k.z, dist_a / dist_plane])

    open(f"tex/bary_table_{var_name}.gen.tex", "w").write(str_table(
        bary_table,
        ["№", "$k_1$", "$k_2$", "$k_3$", "$\sum k$", "$\lambda$"],
        trunc=3,
        split=2,
        row_colors=["red!20" if str(row[3]) == "inf" else "green!20" if abs(row[3] - 1) < 1e-9 else None for row in bary_table]
    ))

    inside_idx = []
    for i in range(len(bary_table)):
        if abs(bary_table[i][3] - 1) < 1e-9:
            inside_idx.append(i)
    
    assert len(inside_idx) == 2

    idx1 = inside_idx[0]
    idx2 = inside_idx[1]
    l1 = bary_table[idx1][4]
    l2 = bary_table[idx2][4]
    norm = max(l1, l2)

    v1 = projected_points[idx1] if l1 > l2 else projected_points[idx2]

    open(f"tex/minkowski_norm_{var_name}.gen.tex", "w")\
        .write(f"$\\left\\lVert {var_name} \\right\\rVert = {round(norm, 4)}$")

    image(draw_poly_barycentric, {
        "verts": vertices,
        "faces": faces,
        "p": a,
        "v1": v1
    }, f"tex/minkowski_norm_{var_name}.png", 20, -80)

minkowski_norm_for_point(in_vert_a, "a")
minkowski_norm_for_point(in_vert_b, "b")