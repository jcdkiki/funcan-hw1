from mayavi import mlab
import numpy as np

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
    return vec3(
        v1.y*v2.z - v1.z*v2.y,
        v1.x*v2.z - v1.z*v2.x,
        v1.x*v2.y - v1.y*v2.x
    )

def length(v):
    return (v.x**2 + v.y**2 + v.z**2)**0.5

def normalize(v):
    return v / length(v)

def dot(v1, v2):
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z

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
    return "\\begin{tabular}{" + "|".join(["c"] * len(header)) + "}" \
         + " \\rowcolor{lightgray}" \
         + " & ".join(header) + " \\\\"

def str_table(vals, header, rows = None, trunc=0, paint_col=True, split=1):    
    if rows == None:
        rows = [str(i+1) for i in range(len(vals))]

    if split > 1:
        n = (len(vals) + split - 1) // split
        return '\n'.join([
            str_table(vals[n*i:n*(i+1)], header, rows[n*i:n*(i+1)], trunc, paint_col, title, 1)
            for i in range(split)
        ])
    
    copy_vals = vals[::1]
    if trunc > 0:
        for i in range(len(copy_vals)):
            copy_vals[i] = [round(x, trunc) for x in copy_vals[i]]

    lines = []
    lines.append(tab_header(header))
    for i in range(len(rows)):
        row = ("\\cellcolor{lightgray}" if paint_col else "") + rows[i]
        row += " & " + " & ".join([str(copy_vals[i][j]) for j in range(len(header) - 1)]) + " \\\\"
        lines.append(row)

    lines.append("\\end{tabular}")
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
    
    planes.append(Plane(n.x, n.y, n.z, dot(n, p1)))

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

open("tex/all_vertices.tex", "w").write(str_table(
    vertices,
    ["№", "$x$", "$y$", "$z$"],
    trunc=4,
    split=3
))

def draw_poly(verts_list, faces_list, filename, elev=30, azim=-60, normals=False, a=1.0):
    mlab.options.offscreen = True
    fig = mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))
    
    verts = np.array(verts_list)
    faces = np.array(faces_list)
    
    mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces,
                         color=(0.68, 0.85, 0.9), opacity=a,
                         representation='surface')
    
    if normals:
        for i in range(len(faces_list)):
            p = pl_points[i]
            n = pl_normals[i]
            mlab.quiver3d(p.x, p.y, p.z, n.x, n.y, n.z,
                         color=(1, 0, 0), mode='arrow', scale_factor=1.0)
    
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
    mlab.view(azimuth=azim, elevation=elev, distance='auto')
    mlab.savefig(filename)
    mlab.close()

draw_poly(in_vertices, in_faces, "tex/input_poly.png", 15, 30)
draw_poly(vertices, faces, "tex/poly.png", 15, 30)

draw_poly(vertices, faces, "tex/poly_normals_1.png", 0, 90, normals=True, a=0.5)
draw_poly(vertices, faces, "tex/poly_normals_2.png", -90, 0, normals=True, a=0.5)
draw_poly(vertices, faces, "tex/poly_normals_3.png", 0, 180, normals=True, a=0.5)
