import numpy as np
from scipy.sparse import coo_matrix
import cv2

from objloader import ObjReader
from make_mov import MovWriter

def depth(nest):
    if isinstance(nest, list) or isinstance(nest, np.ndarray):
        return max([depth(n) for n in nest]) + 1
    else:
        return 0

class Player3DImage:
    def __init__(self, image_size, image_scale, screan_z):
        # 解像度
        self.image_size = image_size
        # img_scale: 実際の大きさ
        self.image_scale = image_scale
        self.screan_z = screan_z
        self.vertexes = []
        self.faces = []
        self.color = np.array([80., 255., 255.])
        
    
    def append_vertex(self, vertex):
        if depth(vertex) == 1:
            self.vertexes.append(vertex)
        elif depth(vertex) == 2:
            self.vertexes += vertex

    def append_face(self, face):
        if depth(face) == 1:
            self.faces.append(face)
        elif depth(face) == 2:
            self.faces += face
    
    def get_image(self, bg, cam_pos, cam_vector):
        vertexes = np.array(self.vertexes)
        faces = np.array(self.faces)
        vertexes = vertexes[faces]
        self.update_3d_pixels(vertexes, cam_pos, cam_vector)
        image = self.get_3d_image(bg)
        return image
    
    def draw2(self, bg, cam_pos, cam_vector, screan_z):
        vertexes = np.array(self.vertexes)
        faces = np.array(self.faces)
        vertexes = vertexes[faces]
        s_xy = self.update_3d_pixels(vertexes, cam_pos, cam_vector, wire_only=True)
        return s_xy
        
    def update_3d_pixels(self, vertexes, cam_pos, cam_vector, wire_only=False):
        # (face数, 頂点数, 次元数)
        # カメラに対する相対座標
        vertexes = vertexes - cam_pos
        # スクリーン上の座標
        # cam_vectorに対してカメラが向いている向き
        view_vector = np.array([0.0, 0.0, 1.0])
        view_vector = np.reshape(view_vector, (3, 1))
        cam_vector = np.reshape(cam_vector, (1, 3))
        # cam_vectorのzx方向
        cam_vector_zx = cam_vector * [1, 0, 1]
        cam_vector_zx = cam_vector_zx / np.linalg.norm(cam_vector_zx)
        # 絶対座標系とcam_vectorのy方向に関する差
        theta_y = np.arccos(np.matmul(cam_vector_zx, view_vector))
        theta_y = theta_y * (-1) ** (cam_vector[:, 0] > 0)
        theta_y = np.reshape(theta_y, ())
        lot_y = np.stack([[np.cos(theta_y), 0, -np.sin(theta_y)],
                        [0, 1, 0],
                        [np.sin(theta_y), 0, np.cos(theta_y)]], axis=0)
        vertexes = np.reshape(vertexes, (-1, 3))
        vertexes = np.matmul(vertexes, lot_y)
        cam_vector_zx = np.reshape(cam_vector_zx, (3, 1))
        mul = np.matmul(cam_vector, cam_vector_zx)
        eps = 0.001
        if 1 - mul[0, 0] > eps:
            theta_x = np.arccos(mul)
            theta_x = theta_x * (-1) ** (cam_vector[:, 1] > 0)
            theta_x = np.reshape(theta_x, ())
            rot_x = np.stack([[1, 0, 0],
                            [0, np.cos(theta_x), -np.sin(theta_x)],
                            [0, np.sin(theta_x), np.cos(theta_x)]])
            # 頂点にx軸の回転を適用
            vertexes = np.matmul(vertexes, rot_x)
        vertexes = np.reshape(vertexes, (-1, 3, 3))
        # vertexes: カメラ座標系の各頂点の座標
        screan_xy = vertexes[:, :, :2] * self.screan_z / np.expand_dims(vertexes[:, :, 2], axis=2)
        if wire_only:
            return screan_xy

        # i, j, x, y, d, r, g, b
        pixels = []
        for sxy, vs in zip(screan_xy, vertexes):
            pixels.append(self.bounding_box(sxy, vs))
        pixels = np.concatenate(pixels)
        if pixels.shape[0]:
            pixels[:, 0] = -(pixels[:, 0] + 1) + int(self.image_size[0] / 2)
            pixels[:, 1] = pixels[:, 1] + int(self.image_size[1] / 2)

            # screan_zの奥にあるものだけを取り出す.
            indices = np.where(pixels[:, 4]>self.screan_z)[0]
            pixels = pixels[indices]

            # screan_zが小さい順に並べる
            sort_id = np.argsort(pixels[:, 4])
            pixels = pixels[sort_id]

            pixel_idx = np.arange(pixels.shape[0])
            ij = (pixels[:, 0] * self.image_size[0] + pixels[:, 1]).astype(np.int)
            _, unique_id, unique_ij_id = np.unique(ij, return_index=True, return_inverse=True)
            pixel_idx_with_unique = np.stack([pixel_idx, unique_ij_id], axis=1)
            adopts = []
            for i in range(np.max(unique_ij_id)):
                adopts.append(np.where(pixel_idx_with_unique[:, 1]==i)[0])
                if len(np.where(pixel_idx_with_unique[:, 1]==i)) > 1:
                    pass
            if adopts:
                adopts = np.concatenate(adopts)
            self.pixels = pixels[adopts]
            self.bbox = np.concatenate([np.min(pixels[:, :2], axis=0), np.max(pixels[:, :2], axis=0)]).astype(np.int)
        self.indices = coo_matrix((np.arange(self.pixels.shape[0])+1, self.pixels[:, :2].T), self.image_size).todense() - 1
        
    def get_3d_image(self, bg):
        if np.max(self.indices) == -1:
            return bg
        
        image = self.pixels[:, 5:8][self.indices].astype(np.uint8)
        mask = self.indices < 0
        bg_indices = np.where(mask)
        image[bg_indices] = bg[bg_indices]
        
        image = cv2.rectangle(image, tuple(self.bbox[:2][::-1]), tuple(self.bbox[2:4][::-1]), (255, 255, 255))
        
        return image

    def bounding_box(self, screan_xy, vertexes):
        # screan_xy(v=3, xy=2)
        # バウンディングボックス整理
        # 最大値最小値からijを生成
        pixel_scale = np.array(self.image_scale) / np.array(self.image_size)
        # 最大値
        max_ji = np.ceil(np.max(screan_xy/pixel_scale, axis=0))
        max_window = (np.array(self.image_size[::-1]) / 2).astype(np.int64)
        max_ji = np.min([max_ji, max_window], axis=0)
        # 最小値
        min_ji = np.floor(np.min(screan_xy/pixel_scale, axis=0))
        min_window = (-np.array(self.image_size[::-1]) / 2).astype(np.int64)
        min_ji = np.max([min_ji, min_window], axis=0)

        rows = np.arange(min_ji[1], max_ji[1])
        columns = np.arange(min_ji[0], max_ji[0])
        # (ij, h, w)
        ij = np.meshgrid(rows, columns, indexing="ij")
        ij = np.transpose(ij, (1, 2, 0))
        # (h*w, ij)
        ij = np.reshape(ij, (-1, 2))
        ji = ij[:, ::-1]
        
        # ijから対応するxyを算出
        xy = (ji + 0.5) * pixel_scale.reshape(1, 2)

        # 面内かどうか
        boundary_vector = np.stack([screan_xy[1] - screan_xy[0],
                            screan_xy[2] - screan_xy[1],
                            screan_xy[0] - screan_xy[2]])
        boundary_vector = boundary_vector / np.linalg.norm(boundary_vector, axis=1, keepdims=True)
        horizon = np.array([1.0, 0.0]).reshape(-1, 1)
        theta = np.arccos(np.dot(boundary_vector, horizon))
        theta = theta.flatten() * (-1) ** (boundary_vector[:, 1] > 0)
        rot_y = np.array([np.sin(theta), np.cos(theta)])
        # (各ピクセル, xy, 各頂点)
        slided = np.expand_dims(xy, axis=2) -\
            np.expand_dims(np.transpose(screan_xy, (1, 0)), axis=0)
        
        # 内積して回転後のy方向のみ得る
        adopt = slided * rot_y
        adopt = np.sum(adopt, axis=1)
        adopt = adopt <= 0
        adopt = np.all(adopt, axis=1)
        indices = np.where(adopt)[0]
        bbox_pixels = np.concatenate([ij, xy], axis=1)
        bbox_pixels = bbox_pixels[indices]
        # 法線
        #normal = tf.cross(vs[0], vs[1])
        crossed = np.cross(vertexes[1] - vertexes[0], vertexes[2] - vertexes[1])
        normal = crossed / np.linalg.norm(crossed)
        # 分子
        numerator = np.dot(normal.reshape(1, -1), vertexes[0].reshape(-1, 1))
        screan_xy = bbox_pixels[:, 2:4]
        screan_xyz = np.concatenate([screan_xy, np.tile([[self.screan_z]], (bbox_pixels.shape[0], 1))], axis=1)
        denominator = np.dot(screan_xyz, normal.reshape(-1, 1))
        z = numerator / denominator * self.screan_z

        # カメラ方向を考慮して回転させた値を入れれば固定の光源となる
        light_vector = np.array([0, 0, 1])
        light_vector = light_vector / np.linalg.norm(light_vector)
        mul = np.dot(normal.reshape(1, -1), -light_vector.reshape(-1, 1))
        # 面で統一されているため一つの値
        mul = mul[0, 0]
        color = self.color * mul
        color = color.reshape(1, -1)
        color = np.tile(color, (bbox_pixels.shape[0], 1))
        bbox_pixels = np.concatenate([bbox_pixels, z, color], axis=1)
        return bbox_pixels

def main():
    mode = "pixel"
    device_num = 0
    cap = cv2.VideoCapture(device_num)
    _, frame = cap.read()
    image_size = np.array(frame.shape[:2])
    h_scale = 10
    w_scale = h_scale * image_size[1] / image_size[0]
    image_scale = np.array((h_scale, w_scale))
    screan_z = 5
    graphicker = Player3DImage(image_size, image_scale, screan_z)
    obj_path = "obj1 v1.obj"
    obj_reader = ObjReader()
    obj_reader.read(obj_path)
    graphicker.append_vertex(obj_reader.vertexes)
    graphicker.append_face(obj_reader.faces)
    omega = 10 * np.pi / 180
    t = 0
    phi0 = 0 * np.pi / 180
    r = 20
    fps = 30
    delta_t = 1 / fps
    # mov_writer = MovWriter("cg.mp4", image_size, fps)
    while t <= 20:
        ret, frame = cap.read()
        frame = frame[:image_size[0], :image_size[1]]
        cam_pos = np.array([-r*np.cos(omega*t+phi0), -0.000, r*np.sin(omega*t+phi0)])
        #cam_pos = np.array([-10, 5, -10])
        print("{:.2f}, {:.2f}, {:.2f}, {:}".format(cam_pos[0], cam_pos[1], cam_pos[2], ((omega*t+phi0)/np.pi*180)%360))
        cam_vector = np.array([0, 0, 0]) - cam_pos
        cam_vector = cam_vector / np.linalg.norm(cam_vector)
        if mode == "pixel":
            img = graphicker.get_image(frame, cam_pos, cam_vector)
        
        if mode == "wire":
            s_xy = graphicker.draw2(cam_pos, cam_vector, screan_z)
            img = np.zeros(image_size, dtype=np.uint8)
            for i in range(s_xy.shape[0]):
                for j in range(s_xy.shape[1]):
                    x0 = int((s_xy[i, j, 0] / image_scale[0] + 1 / 2) * image_size[1])
                    y0 = int((-s_xy[i, j, 1] / image_scale[1] + 1 / 2) * image_size[0])
                    x1 = int((s_xy[i, (j+1)%3, 0] / image_scale[0] + 1 / 2) * image_size[1])
                    y1 = int((-s_xy[i, (j+1)%3, 1] / image_scale[1] + 1 / 2) * image_size[0])
                    img = cv2.line(img, (x0, y0), (x1, y1), color=(255, 255, 255))
        cv2.imshow("img", img)
        cv2.waitKey(10)
        # mov_writer.write(img)
        t += delta_t
    mov_writer.release()

if __name__ == "__main__":
    main()