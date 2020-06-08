import cv2
import numpy as np
import quaternion

class Pixel:
    def __init__(self, x, y, depth, n_vector):
        self.x = x
        self.y = y
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

        return np.array([x[0]/x[2], x[1]/x[2], 0])

class Face:
    def __init__(self, v0, v1, v2):
        self.v = [v0, v1, v2]
        
class Drawer:
    def __init__(self):
        self.faces = []

    def append(self, faces):
        if type(faces) == Face:
            self.faces.append(faces)
        elif type(faces) in [list, tuple]:
            self.faces += faces

    def draw(self, cam_pos, cam_vector, wait):
        img = np.ones((1080, 1080, 3), dtype=np.float)
        scale = 1000
        for face in self.faces:
            self.draw_face(img, face, scale, cam_pos, cam_vector)

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
    while True:
        #cam_pos[2] += 0.1
        #r -= 0.01
        cam_pos = np.array([-r*np.cos(omega*t), 2, r*np.sin(omega*t)])
        print("{:.2f}, {:.2f}, {:.2f}".format(cam_pos[0], cam_pos[1], cam_pos[2]))
        cam_vector = np.array([0, 0, 0]) - cam_pos
        cam_vector /= np.linalg.norm(cam_vector)
        drawer.draw(cam_pos, cam_vector, wait)
        t += 0.1

if __name__ == "__main__":
    main()
