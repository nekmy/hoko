import cv2
import numpy as np
import quaternion

def adopt_side(target, x1, x2):
    # x1 -> x2
    target = target - x1
    x_vector = x2 - x1
    theta = np.arccos(np.dot([1, 0], x_vector) / (np.linalg.norm([1, 0]) * np.linalg.norm(x_vector)))
    theta *= (-1) ** (x_vector[1] < 0)
    target = np.reshape(target, (-1, 1))
    lot = np.array([[np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])
    target = np.dot(lot, target)
    return target[1][0] <= 0

def get_depth(n_vector, v0, target, screan_depth):
    a, b, c = n_vector
    d = -(a * v0[0] + b * v0[1] + c * v0[2])
    e, f = target / screan_depth
    depth = -d / (a * e + b * f + c)
    return depth

class Pixel:
    def __init__(self, i, j, depth, n_vector):
        self.i = i
        self.j = j
        self.depth = depth
        self.n_vector = n_vector

class Vertex:
    def __init__(self, x):
        self.x = x
    
    def get_x(self, cam_pos, cam_vector):
        x = self.x - cam_pos
        # cam_posとcam_vectorの角度
        world_vector = (0, 0, 1)
        cam_vector1 = cam_vector.copy()
        cam_vector1[1] = 0
        theta = np.arccos(np.dot(world_vector, cam_vector1) / (np.linalg.norm(world_vector) * np.linalg.norm(cam_vector1)))
        n = np.cross(cam_vector1, world_vector) if theta else world_vector

        n /= np.linalg.norm(n)
        n *= np.sin(theta/2)
        q = np.quaternion(np.cos(theta/2), n[0], n[1], n[2])
        p = np.quaternion(0, x[0], x[1], x[2])
        p = q * p * q.conj()

        cam_vector2 = cam_vector.copy()
        cam_vector2[1] = 0
        theta = np.arccos(np.dot(cam_vector, cam_vector2) / (np.linalg.norm(cam_vector) * np.linalg.norm(cam_vector2)))
        n = np.cross(cam_vector, cam_vector2) if theta else world_vector
        n = np.array([-1.0, 0.0, 0.0])

        n /= np.linalg.norm(n)
        n *= np.sin(theta/2)
        q = np.quaternion(np.cos(theta/2), n[0], n[1], n[2])
        x = q * p * q.conj()

        x = np.array([x.x, x.y, x.z])

        return x

class Face:
    def __init__(self, v0, v1, v2):
        self.v = [v0, v1, v2]
    
    def get_pixels(self, img_shape, cam_pos, cam_vector, screan_depth, scale):
        debug_img = np.ones(img_shape)
        H, W = img_shape
        xs = [v.get_x(cam_pos, cam_vector) for v in self.v]
        xs_screan = [x.copy() for x in xs]
        for x in xs_screan:
            x[0:2] /= x[2]
            x[2] = 0.0
        xs_screan = [x * scale for x in xs_screan]
        n_vector = np.cross(xs[0] - xs[1], xs[2] - xs[1])
        n_vector /= np.linalg.norm(n_vector)
        center_i = int(H / 2)
        center_j = int(W / 2)
        pixels = []
        for i in range(H):
            y = center_i - i
            for j in range(W):
                x = j - center_j
                adopt0 = adopt_side([x, y], xs_screan[0][:2], xs_screan[1][:2])
                adopt1 = adopt_side([x, y], xs_screan[1][:2], xs_screan[2][:2])
                adopt2 = adopt_side([x, y], xs_screan[2][:2], xs_screan[0][:2])
                if (adopt0 * adopt1 * adopt2):
                    depth = get_depth(n_vector, xs[0], np.array([x, y]), screan_depth)
                    pixels.append(Pixel(i, j, depth, n_vector))
                    debug_img[i, j] = depth
        
        return pixels


        
class Drawer:
    def __init__(self):
        self.faces = []

    def append(self, faces):
        if type(faces) == Face:
            self.faces.append(faces)
        elif type(faces) in [list, tuple]:
            self.faces += faces

    def draw(self, cam_pos, cam_vector, wait):
        H, W = 50, 50
        img = np.ones((H, W, 3), dtype=np.float)
        scale = 100
        screan_depth = 1.0
        depthes = np.ones((H, W)) * float("inf")
        for face in self.faces:
            pixels = face.get_pixels(img.shape[:2], cam_pos, cam_vector, screan_depth, scale)
            for pixel in pixels:
                i = pixel.i
                j = pixel.j
                depth = pixel.depth
                if depthes[i, j] > depth:
                    depthes[i, j] = depth
                    img[i, j] = [0, 0, 0]


        cv2.imshow("image", img)
        cv2.waitKey(wait)
    
    def draw_face(self, img, face, scale, cam_pos, cam_vector):
        center_y, center_x, _ = [int(n / 2) for n in img.shape]

        for i in range(3):
            x0 = int(center_x + face.v[i].get_x(cam_pos, cam_vector)[0] * scale)
            y0 = int(center_y - face.v[i].get_x(cam_pos, cam_vector)[1] * scale)
            x1 = int(center_x + face.v[(i+1)%3].get_x(cam_pos, cam_vector)[0] * scale)
            y1 = int(center_y - face.v[(i+1)%3].get_x(cam_pos, cam_vector)[1] * scale)
            img = cv2.line(img, (x0, y0), (x1, y1), color=(0, 0, 0))

def main():
    drawer = Drawer()
    v = [Vertex([-1.0, 0.0, 0.0]),
         Vertex([0.0, 1.0, 0.0]),
         Vertex([0.0, 0.0, 1.0]),
         Vertex([0.0, -1.0, 0.0]),
         Vertex([0.0, 0.0, -1.0]),
         Vertex([1.0, 0.0, 0.0])]
    faces = [Face(v[0], v[2], v[1]),
             Face(v[0], v[3], v[2]),
             Face(v[0], v[4], v[3]),
             Face(v[0], v[1], v[4]),
             Face(v[1], v[2], v[4]),
             Face(v[5], v[4], v[2]),
             Face(v[5], v[2], v[3]),
             Face(v[5], v[3], v[4])]
    drawer.append(faces)
    cam_pos = np.array([0, 0, -50])
    cam_vector = np.array([0, 0, -1])
    wait = 10
    r = 5
    t = 0
    omega = 10 * np.pi / 180
    phi0 = np.pi / 2
    while True:
        #cam_pos[2] += 0.1
        #r -= 0.01
        cam_pos = np.array([-r*np.cos(omega*t+phi0), 2, r*np.sin(omega*t+phi0)])
        print("{:.2f}, {:.2f}, {:.2f}".format(cam_pos[0], cam_pos[1], cam_pos[2]))
        cam_vector = np.array([0, 0, 0]) - cam_pos
        cam_vector /= np.linalg.norm(cam_vector)
        drawer.draw(cam_pos, cam_vector, wait)
        t += 0.1

if __name__ == "__main__":
    main()
