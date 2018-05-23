from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from random import randint
import math
import cv2
import numpy as np
import os


LightCordinates = [45, 20, 15]
OFFSET = 1
SCREEN_SIZE = 600

Kd = 0.1
Ks = 1 - Kd
I = (1, 1, 1)


angle = 60 #initializing angle
RotateAxis = [0, 1, 0]

TWidth = 300
THeight = 300
Shading = 3


#Methods for LAB 4

def TextureFill(W, H):
    try:
        txt = cv2.imread("C:/Users/prave/PycharmProjects/graphics-hw3/data/blackwhite.png")
        txt = cv2.resize(txt, (W, H), interpolation=cv2.INTER_CUBIC)
        return txt
    except Exception, e:
        print "Texture fill failed", e

def MapToCylinder(Xc, Yc, Zc, W, Xo, Yo, Zo, H):
    #Mapping object to cylinder: Xc, Yc, Zc are the center of the circle
    #Xo, Yo, Zo are the coordinates of the point on Object
    try:
        # R is the radius
        R = W * 1.0 / (2*math.pi)
        # Distance between the point and the center of the circle
        distance = math.sqrt((Xo-Xc)*(Xo-Xc) + (Yo-Yc)*(Yo-Yc))
        #Cylinder's is equal to sphere's center
        y = R * 1.0 * (Yo-Yc) / distance + Yo
        x = (R-Xc) * 1.0 * (Xo-Xc) / distance
        if Xo-Xc == 0:
            z = (Zo - Zc) * 1.0 * (y - Yc) / (Yo - Yc) + Zc
        else:
            z = (Zo-Zc) * 1.0 * (x-Xc) / (Xo-Xc) + Zc

        #Checking for out of range
        if abs(z) > H*1.0/2:
            return 0, 0
        else:
            return x, z
    except Exception, e:
        print "EXCEPTION: Mapping to cylinder unsuccessful"


def TextureCylinder(x, z, H, W):
  # W: Width of the cylinder H: Height of the cylinder
    try:
        R = W * 1.0 / (2*math.pi)
        z = z + H * 1.0 / 2
        u = math.acos(x * 1.0 / R) / (2 * math.pi) * W
        v = z
        return u, v
    except Exception, e:
        print "Texture Exception", e, x, R


def Sphere(x, y, z, W, H):
    try:
        # Xc,Yc,Zc is the centroid of the sphere
        Xc, Yc, Zc = 0, 0, 0
        # a is the Centroid point
        a = np.mat([Xc, Yc, Zc])
        # b is the point on the object
        b = np.mat([x, y, z])  # point on the object
        pc = (a-b) / np.linalg.norm(a - b)
        p = pc.tolist()[0]
        X, Y, Z = p[0], p[1], p[2]
        u = 0.5 + math.atan2(Z, X) / (2*math.pi)
        v = 0.5 - math.asin(Y) / math.pi
        Xtexture = int(u*W)
        Ytexture = int(v*H)
        return Xtexture, Ytexture
    except Exception, e:
        print "Sphere exception", e


AmbientC = (0.3,0.3,0.3) #ambient color
ReflectionValue= 16 #N value


def Find_I(N, V, L, Od): #Function to find I (color)
    H = (np.mat(L) + np.mat(V)) / np.linalg.norm(np.mat(L) + np.mat(V))
    H = H.tolist()[0]
    color = []
    for i in range(3):
        Ir = AmbientC[i] + I[i] * (Kd * Od[i]*1.0/255 * DotMultiplication(N, L) + Ks * Od[i]*1.0/255 * math.pow(DotMultiplication(N, H), ReflectionValue))
        color.append(Ir)
    return color


def FindGouraud(lines): #function to find the normal for Gouraud shading
    vertex = {}
    for line in lines:
        N = line[-3:-2][0]
        for v in line[:-3]:
            if not vertex.get(v):
                vertex[v] = [N]
            else:
                vertex[v].append(N)
    gouraud_normal = {}
    for key, value in vertex.items():
        up = np.mat([0.0, 0.0, 0.0])
        for n in value:
            up += np.mat(n)
        GouraudNormal = up / np.linalg.norm(up)
        gouraud_normal[key] = GouraudNormal.tolist()[0]
    return gouraud_normal


#Reading the text file with coordinates
def ReadFile(path=None, rotate_angle=None, axis=None):
    matric = None
    if rotate_angle:
        matric = GetMatrix(rotate_angle, axis)
    if not path:
        path = "C:/Users/prave/PycharmProjects/graphics-hw3/data/sphere.d.txt"       #Change the path to the required file
    _sum = []
    vertexes = []
    lines = []
    _first_line = 1
    _vertex_line = 0
    with open(path) as f:
        for data in f.readlines():
            data = data.strip('\n')
            # nums for one line
            nums = data.split(" ")
            # read fist line
            if _first_line:
                for re in nums:
                    if re:
                        _sum.append(re)
                _first_line = 0
                continue
            _vertex_line = int(_sum[1])

    # reading vertices and lines
    with open(path) as f:
        i = 1
        for data in f.readlines():
            data = data.strip('\n')
            nums = data.split(" ")
            if i:
                i = 0
                continue
            if _vertex_line:
                _vertex = []
                #storing each x,y,z coordinates into vertex
                for re in nums:
                    if re:
                        _vertex.append(float(re))
                # storing all the vertexes coordinates into vertexes
                if matric is not None:
                    _vertex = np.dot(matric, np.mat(_vertex).T).T.tolist()[0]
                vertexes.append(_vertex)
                _vertex_line -= 1
                continue
            # lines
            _line = []
            for re in nums:
                if re:
                    _line.append(int(re)-1)    #Subtracting each coordinate by 1, since indexing starts from 0
            lines.append(_line[1:])
    return vertexes, lines


def GetMatrix(k, axis):
    x = axis[0]
    y = axis[1]
    z = axis[2]
    angle = np.pi / 180 * k
    c = np.cos(angle)
    s = np.sin(angle)
    m = 1 - c
    matrix= np.mat(list([[x * x * m + c, x * y * m + z * s, x * z * m - y * s],
                          [y * x * m - z * s, y * y * m + c, y * z * m + x * s],
                          [x * z * m + y * s, y * z * m - x * s, z * z * m + c]]))
    return matrix



#funtion to perform cross product
def VertexMultiplication(a, b):
    ax = a[0]
    ay = a[1]
    az = a[2]
    bx = b[0]
    by = b[1]
    bz = b[2]
    cx = ay*bz - az*by
    cy = az*bx - ax*bz
    cz = ax*by - ay*bx
    return [cx, cy, cz]


#function to perform dot product
def DotMultiplication(a, b):
    ax = a[0]
    ay = a[1]
    az = a[2]
    bx = b[0]
    by = b[1]
    bz = b[2]
    return ax*bx + ay*by + az*bz


#Function to find M perspective and M view
def FindPersView(c, p, v_prime, d=1.0, f=500.0, h=15.0):
    C = np.mat(c)
    P = np.mat(p)
    #To calculate this, N = 1/math.sqrt((P-C)*(P-C).T) * (P-C)
    N = (P-C) / np.linalg.norm(P - C)  # Z-axis
    N = N.tolist()[0]
    U = VertexMultiplication(N, v_prime) / np.linalg.norm(VertexMultiplication(N, v_prime))  # X-axis
    V = VertexMultiplication(U, N)  # Y-axis
    U = U.tolist()
    U_val = U + [0]
    V_val = V + [0]
    N_val = N + [0]
    R = np.mat([U_val, V_val, N_val, [0, 0, 0, 1]])
    T = np.mat([[1, 0, 0, -c[0]], [0, 1, 0, -c[1]], [0, 0, 1, -c[2]], [0, 0, 0, 1]])
    #Calculating M view
    M_view = R * T
    #Calculating M perspective
    M_pers = np.mat([[d/h, 0, 0, 0], [0, d/h, 0, 0], [0, 0, f/(f-d), -d*f/(f-d)], [0, 0, 1, 0]])
    return M_view, M_pers


#Function to perform the transformation
def transformation(c, p, v_prime, d=1.0, f=500.0, h=15.0, angle=None):
    M_view, M_pers = FindPersView(c, p, v_prime, d, f, h)
    vertexes, lines = ReadFile(rotate_angle=angle, axis=RotateAxis)
    # Mview * points
    view_vs = []
    for v in vertexes:
        v = v + [1]
        v = np.mat(v).T     #Self Transpose
        view_v = M_view * v
        view_v = view_v.T.tolist()[0]
        view_vs.append(view_v)
    #To find normal of the surface
    VisibleSurface = []
    for face in lines:
        ves = []
        for v in face:
            ves.append(view_vs[v])
        if not ves:
            continue
        if len(ves) < 3:
            VisibleSurface.append(face)
            continue
        dot1 = np.mat(ves[0])
        dot2 = np.mat(ves[1])
        dot3 = np.mat(ves[2])
        line1 = dot2 - dot1
        line2 = dot3 - dot2
        line_of_sight = (np.mat([0, 0, 0]) - np.mat(ves[0][:-1])).tolist()[0]

    #To find line of sight
        L = LightCordinates + [1]
        L = np.mat(L).T
        Lview = M_view * L
        Lview = Lview.T.tolist()[0][:-1]
        line_of_light = (np.mat(Lview) - np.mat(ves[0][:-1])).tolist()[0]

        vertex1 = line1.tolist()[0][0:-1]
        vertex2 = line2.tolist()[0][0:-1]

        #finding normal
        normal = VertexMultiplication(vertex1, vertex2)
        visible = DotMultiplication(normal, line_of_sight)
        if visible > 0:
            #Adding normal of the surface to the list
            face.append(normal)  # N
            face.append(line_of_sight)  # V
            face.append(line_of_light)  # L
            VisibleSurface.append(face)

    Visible_Surface = []
    for v in view_vs:
        v = v
        v = np.mat(v).T
        vs = M_pers * v  # Mp*Mv*V.T
        vs = vs.T.tolist()[0]   # lists
        # dividing x,y,z by W
        vs[0] = int((vs[0] / vs[-1] + OFFSET) * SCREEN_SIZE / (OFFSET+1))
        vs[1] = int((vs[1] / vs[-1] + OFFSET) * SCREEN_SIZE/ (OFFSET+1))
        vs[2] = abs((vs[2] / vs[-1]))
        Visible_Surface.append(vs)
    return Visible_Surface, VisibleSurface


#Function for Z Buffer
def zbuffer():
    global angle
    #Initializing zbuffer
    glClear(GL_COLOR_BUFFER_BIT)
    DepthBuffer, FrameBuffer = Find_Depth_Frame_Buffer()
    ScanLine(DepthBuffer, FrameBuffer, angle)    #Perform Scan line
    glBegin(GL_POINTS)
    #Coloring
    for x in range(len(FrameBuffer)):
        for y in range(len(FrameBuffer)):
            color = FrameBuffer[x][y]
            glColor3f(color[0], color[1], color[2])
            glVertex2i(x, y)
    glEnd()
    glFlush()
    angle += 10
    angle %= 360

#Function to implement Scan line
def ScanLine(DepthBuffer, FrameBuffer, angle = None):
    local_vertexs, local_line = ReadFile()
    # initializing variables
    d = 3.8
    f = 1
    h = 0.5
    V_prime = [0, 0.5, 0]  # Y-direction of camera
    vertexes, lines = transformation(C, P, V_prime, d, f, h, angle)
    gouraud_normal = FindGouraud(lines)
    #Initializing the table
    surfaces = []
    for l in lines:
        s = []
        for v in l[:-3]:
            vertex = vertexes[v][:-1]
            local_vertex=local_vertexs[v]
            vertex.append(local_vertex)
            if Shading == 3:
                vertex.append(gouraud_normal[v])
            s.append(vertex)
        s.append(l[-3:])
        surfaces.append(s)

    #For each side,
    for s in surfaces:
        # Finding normal vector
        others = s[-1]
        N = others[0]
        V = others[1]
        L = others[2]
        s = s[:-1]

        smax, smin = ScanlineScope(s)
        local_y_max, local_y_min = LocalVertexScope(s)
        # Finding edge table and active table
        EdgeTable = Edge_Table(smax, smin)
        AET = EdgeModel()
        # Taking 4 adjacent points to the current point, line connecting points 1 and 2 as edge are processed by this cycle
        v = len(s)
        # calculating singularity using points 0 and 3
        for i in range(v):
            x0 = s[(i-1+v) % v][0]
            x1 = s[i][0]
            z1 = s[i][2]
            x2 = s[(i+1) % v][0]
            z2 = s[(i+1) % v][2]
            x3 = s[(i+2) % v][0]
            y0 = s[(i-1 + v) % v][1]
            y1 = s[i][1]
            y2 = s[(i + 1) % v][1]
            y3 = s[(i + 2) % v][1]
        # Neglecting horizontal line
            if y1 == y2:
                continue

            #Finding Xmin, Xmax, Ymin, Ymax
            ymin = y1 if y1 < y2 else y2
            ymax = y1 if y1 > y2 else y2
            xmin = x1 if y1 < y2 else x2
            dx = (x1-x2) * 1.0 / (y1-y2)

            local_x1 = s[i][-2][0]
            local_y1 = s[i][-2][1]
            local_x2 = s[(i + 1) % v][-2][0]
            local_y2 = s[(i + 1) % v][-2][1]
            local_x_min = local_x1 if local_y1 < local_y2 else local_x2

            #if y2>y1>y0, y1 is a singular point and y1>y2>y3, y2 is a singular point
            if (y2 > y1 > y0) or (y1 > y2 > y3):
                ymin += 1
                xmin += dx

            #Creating edge
            edge = EdgeModel()
            edge.ymax = ymax
            edge.xmin = xmin
            edge.dx = dx
            edge.edge_next = EdgeTable[ymin]
            edge.edge_vertex1 = [x1, y1, z1]
            edge.edge_vertex2 = [x2, y2, z2]
            EdgeTable[ymin] = edge

            edge.local_vertex1 = s[i][-2]
            edge.local_vertex2 = s[(i + 1) % v][-2]
            edge.local_x_min = local_x_min

            if Shading == 3:
                edge.vertex_normal1 = s[i][-1]
                edge.vertex_normal2 = s[(i+1) % v][-1]
        Scanning(smin, smax, EdgeTable, AET, DepthBuffer, FrameBuffer, N, V, L, local_y_min, local_y_max)


def Scanning(ymin, ymax, EdgeTable, AET, DepthBuffer, FrameBuffer, N, V, L, local_y_min, local_y_max):
    # Scanning from bottom to top, y-cord increments one at a time
    if ymin == ymax:
        return
    d_local_y = 1.0 / (ymax - ymin) * (local_y_max - local_y_min)

    texture = TextureFill(TWidth, THeight)
    local_y = local_y_min
    local_i = 0

    for y in range(ymin, ymax + 1):
        if local_i > 0:
            local_y += d_local_y
        local_i += 1

        #Taking out all the edges intersecting with the scanline in line and inserting them into AET in ascending order of x
        while EdgeTable[y]:
            new_edge = EdgeTable[y]
            AET_edge = AET
            #Searching for a suitable position in AET for insertion
            while AET_edge.edge_next:
                if new_edge.xmin > AET_edge.edge_next.xmin:
                    AET_edge = AET_edge.edge_next
                    continue

                if new_edge.xmin == AET_edge.edge_next.xmin and new_edge.dx > AET_edge.edge_next.dx:
                    AET_edge = AET_edge.edge_next
                    continue
                break

            #inserting into current AET position
            EdgeTable[y] = new_edge.edge_next
            #Pointing to next edge
            new_edge.edge_next = AET_edge.edge_next
            AET_edge.edge_next = new_edge

        #Two sides of AET are paired and filled
        p = AET
        while p.edge_next and p.edge_next.edge_next:
            #Considering endpoints on the left
            v1 = p.edge_next.edge_vertex1
            v2 = p.edge_next.edge_vertex2

            #Considering end points on the right
            v3 = p.edge_next.edge_next.edge_vertex1
            v4 = p.edge_next.edge_next.edge_vertex2

            #Caluculating the mid-point of left side
            Za = v1[2] - (v1[2]-v2[2]) * (v1[1]-y)*1.0 / (v1[1] - v2[1])
            # Caluculating the mid-point of right side
            Zb = v3[2] - (v3[2]-v4[2]) * (v3[1]-y)*1.0 / (v3[1] - v4[1])

            local_vertex1 = p.edge_next.local_vertex1
            local_vertex2 = p.edge_next.local_vertex2
            local_vertex3 = p.edge_next.edge_next.local_vertex1
            local_vertex4 = p.edge_next.edge_next.local_vertex2
            local_z_a = local_vertex1[2] - (local_vertex1[2] - local_vertex2[2]) * (v1[1] - y) * 1.0 / (v1[1] - v2[1])
            local_z_b = local_vertex3[2] - (local_vertex3[2] - local_vertex4[2]) * (v3[1] - y) * 1.0 / (v3[1] - v4[1])

            if Shading == 3:
                N1 = np.mat(p.edge_next.vertex_normal1)
                N2 = np.mat(p.edge_next.vertex_normal2)
                N3 = np.mat(p.edge_next.edge_next.vertex_normal1)
                N4 = np.mat(p.edge_next.edge_next.vertex_normal2)
                Na = N1 * (y - v2[1]) * 1.0 / (v1[1] - v2[1]) + N2 * (v1[1] - y) * 1.0 / (v1[1] - v2[1])
                Nb = N3 * (y - v4[1]) * 1.0 / (v3[1] - v4[1]) + N4 * (v3[1] - y) * 1.0 / (v3[1] - v4[1])

            i = 0
            local_x = p.edge_next.local_x_min
            local_x_length = abs(p.edge_next.edge_next.local_x_min - p.edge_next.local_x_min)

            for x in range(int(p.edge_next.xmin), int(p.edge_next.edge_next.xmin)):
                xa = int(p.edge_next.xmin)
                xb = int(p.edge_next.edge_next.xmin)
                xp = x
                Zp = Zb - (Zb - Za) * (xb - xp)*1.0 / (xb - xa)

                if i > 0:
                    local_x += 1.0 / (xb - xa) * local_x_length
                local_z = local_z_b - (local_z_b - local_z_a) * (xb - xp) * 1.0 / (xb - xa)

                u, v = Sphere(local_x, local_y, local_z, TWidth, THeight)
                color = texture[v-1][u-1]
                if Shading == 3:
                    Np = Na * (xb - xp) * 1.0 / (xb - xa) + Nb * (xp - xa) * 1.0 / (xb - xa)
                    Np = Np.tolist()[0]
                    Ip = Find_I(Np, V, L, color)

                if x >= SCREEN_SIZE or y >= SCREEN_SIZE or x <= 0 or y <= 0:
                    continue

                if Zp < DepthBuffer[x][y]:
                    DepthBuffer[x][y] = Zp
                    if Shading == 1:
                        FrameBuffer[x][y] = Find_I(N, V, L)
                    if Shading in (2, 3):
                        FrameBuffer[x][y] = Ip
            p = p.edge_next.edge_next

        # Removing edges from AET for which y=y max
        p = AET
        while p.edge_next:
            if p.edge_next.ymax == y:
                p.edge_next = p.edge_next.edge_next
            else:
                p = p.edge_next

        # Updating x to x + increment and entering next cycle
        p = AET
        while p.edge_next:
            p.edge_next.xmin += p.edge_next.dx
            p = p.edge_next


#Finding Scanline scope
def ScanlineScope(vertexes):
    # Finding ymax and ymin for all points in the scan range
    ymax = 0
    ymin = SCREEN_SIZE
    for v in vertexes:
        if v[1] > ymax:
            ymax = v[1]
        if v[1] < ymin:
            ymin = v[1]
    return ymax, ymin


def LocalVertexScope(vertexes):
    ymax = -1000
    ymin = 1000
    for v in vertexes:
        local_v = v[-2]
        if local_v[1] > ymax:
            ymax = local_v[1]
        if local_v[1] < ymin:
            ymin = local_v[1]
    return ymax, ymin

#Creating edge table
def Edge_Table(ymax, ymin):
    edge_table = {}
    for i in range(ymin, ymax+1):
        edge_table[i] = None
    return edge_table

#Edge model class
class EdgeModel:
    def __init__(self):
        self.gouraud1 = None
        self.gouraud2 = None
        self.ymax = None
        self.xmin = None
        self.dx = None
        self.edge_next = None
        self.edge_vertex1 = None
        self.edge_vertex2 = None
        self.vertex_normal1 = None
        self.vertex_normal2 = None
        self.local_vertex1 = None
        self.local_vertex2 = None
        self.local_x_min = None

#Function to get frame buffer and depth buffer
def Find_Depth_Frame_Buffer():
    # init depth_buffer
    depth_buffer = [[1 for col in range(SCREEN_SIZE)] for row in range(SCREEN_SIZE)]
    frame_buffer = [[(0, 0, 0) for col in range(SCREEN_SIZE)] for row in range(SCREEN_SIZE)]
    return depth_buffer, frame_buffer


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    gluOrtho2D(0, SCREEN_SIZE, 0, SCREEN_SIZE)


#for the output window
def Output():
    global lines
    global vertexes
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE)
    glutInitWindowSize(SCREEN_SIZE, SCREEN_SIZE)
    glutCreateWindow("Output")
    glutDisplayFunc(zbuffer)
    glutIdleFunc(zbuffer)
    init()
    glutMainLoop()



if __name__ == '__main__':
    C = [4.0, 6.0, 10.0]  # Camera position
    P = [0.0, 0.0, 0.0]
    V_prime = [0, 1, 0]  # V' co-ordinates, Y-direction of Camera
    a = [2, 3, 5]
    b = [0, 1, 0]
    Output()
