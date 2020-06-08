import numpy as np
import cv2

from npcg import Player3DImage
from objloader import ObjReader

class DatasetCapture:

    def __init__(self, device):
        self.cap = cv2.VideoCapture(device)
        _, frame = self.cap.read()
        image_size = np.array(frame.shape[:2])
        h_scale = 10
        w_scale = h_scale * image_size[1] / image_size[0]
        image_scale = np.array([h_scale, w_scale])
        screan_z = 5
        self.player = Player3DImage(image_size, image_scale, screan_z)
        self.reset_states()
    
    def reset_states(self):
        self.cam_r = 20
        self.yaw = 0.0
        self.pitch = 0.0

    def add_object(self, objpath):
        objreader = ObjReader()
        objreader.read(objpath)
        self.player.append_vertex(objreader.vertexes)
        self.player.append_face(objreader.faces)
    
    def get_image(self):
        _, frame = self.cap.read()
        campos = np.array([0.0, 0.0, self.cam_r])
        rot_yow = np.array([[np.cos(self.yaw), 0, -np.sin(self.yaw)],
                            [0, 1, 0],
                            [np.sin(self.yaw), 0, np.cos(self.yaw)]])
        campos = np.dot(campos.reshape(1, -1), rot_yow)
        rot_pitch = np.array([[1, 0, 0],
                              [0, np.cos(self.pitch), np.sin(self.pitch)],
                              [0, -np.sin(self.pitch), np.cos(self.pitch)]])
        campos = np.dot(campos.reshape(1, -1), rot_pitch)
        camdir = np.array([0, 0, 0]) - campos
        image = self.player.get_image(frame, campos, camdir)
        return image

    def change_campos(self, cammov):
        self.yaw += cammov[0]
        self.pitch += cammov[1]
        
    def draw(self, window, fps=30):
        delay = 1000 / fps
        while True:
            image = self.get_image()
            cv2.imshow(window, image)
            cv2.waitKey(int(delay))

    
def main():
    device = 0
    capture = DatasetCapture(device)
    objpath = "obj1 v1.obj"
    capture.add_object(objpath)
    capture.draw("image")
    pass

if __name__ == "__main__":
    main()
        
