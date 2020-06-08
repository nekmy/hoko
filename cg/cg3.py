import tensorflow as tf
from tensorflow.python import debug as tfdbg
import numpy as np
import cv2

def depth(nest):
    if isinstance(nest, list):
        return max([depth(n) for n in nest]) + 1
    else:
        return 0

class Graphicker:
    def __init__(self, image_size, img_scale, tf_debug=False):
        self.image_size = image_size
        self.img_scale = img_scale
        self.faces = []
        self.vertexes, self.cam_pos, self.cam_vector, self.screan_z, self.img = self.drawer_graph()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        if tf_debug:
            self.sess = tfdbg.LocalCLIDebugWrapperSession(self.sess)
        self.sess.run(init)
    
    def append(self, face):
        if depth(face) == 2:
            self.faces.append(face)
        elif depth(face) == 3:
            self.faces += face
    
    def draw(self, cam_pos, cam_vector, screan_z):
        vertexes = np.array(self.faces)
        img = self.sess.run(self.img, feed_dict={self.vertexes: vertexes, self.cam_pos: cam_pos,
                    self.cam_vector: cam_vector, self.screan_z: screan_z})
        return img
        
        

    def drawer_graph(self):
        # (face数, 頂点数, 次元数)
        input_vertexes = tf.placeholder(tf.float64, (None, 3, 3), name="input_vertexes")
        input_cam_pos = tf.placeholder(tf.float64, (3,), name="input_cam_pos")
        input_cam_vector = tf.placeholder(tf.float64, (3,), name="input_cam_vector")
        vertexes = input_vertexes - input_cam_pos
        # スクリーン上の座標
        view_vector = tf.constant([0, 0, 1], dtype=tf.float64)
        view_vector = tf.reshape(view_vector, (3, 1))
        cam_vector = tf.reshape(input_cam_vector, (1, 3))
        cam_vector_y = cam_vector * [1, 0, 1]
        cam_vector_y = cam_vector_y / tf.norm(cam_vector_y)
        theta_y = tf.acos(tf.matmul(cam_vector_y, view_vector))
        theta_y = theta_y * (-1) ** tf.cast((cam_vector[:, 2] > 0), tf.float64)
        theta_y = tf.reshape(theta_y, ())
        lot_y = tf.stack([[tf.cos(theta_y), 0, -tf.sin(theta_y)],
                        [0, 1, 0],
                        [tf.sin(theta_y), 0, tf.cos(theta_y)]], axis=0)
        vertexes = tf.reshape(vertexes, (-1, 3))
        vertexes = tf.matmul(vertexes, lot_y)
        cam_vector_y = tf.reshape(cam_vector_y, (3, 1))
        theta_x = tf.acos(tf.matmul(cam_vector, cam_vector_y))
        theta_x = theta_x * (-1) ** tf.cast((cam_vector[:, 1] > 0), dtype=tf.float64)
        theta_x = tf.reshape(theta_x, ())
        rot_x = tf.stack([[1, 0, 0],
                        [0, tf.cos(theta_x), -tf.sin(theta_x)],
                        [0, tf.sin(theta_x), tf.cos(theta_x)]])
        vertexes = tf.matmul(vertexes, rot_x)
        vertexes = tf.reshape(vertexes, (-1, 3, 3), name="rotated_vertexes")
        input_screan_z = tf.placeholder(tf.float64, ())
        screan_xy = vertexes[:, :, :2] / tf.expand_dims(vertexes[:, :, 2], axis=2)
        screan_xy = tf.multiply(screan_xy, input_screan_z, name="screan_xy")
        outside_indices = int(self.image_size[0] / 2)

        def body(elems):
            screan_x, vertexes = elems
            vs = vertexes
            sx = screan_x
            maxes = tf.floor(tf.reduce_max(sx*self.img_scale, axis=0), name="maxes")
            mins = tf.floor(tf.reduce_min(sx*self.img_scale, axis=0), name="mins")
            rows = tf.range(mins[1], maxes[1], name="rows")
            columns = tf.range(mins[0], maxes[0], name="columns")
            # (ij, h, w)
            ij = tf.meshgrid(rows, columns, indexing="ij", name="meshgrid_ij")
            ij = tf.transpose(ij, (1, 2, 0))
            # (h*w, ij)
            ij = tf.reshape(ij, (-1, 2))
            scale = tf.cast(self.img_scale, dtype=tf.float64)
            ji = ij[:, ::-1]
            xy = tf.divide(ji + 0.5, scale, name="xy")

            x_vector = tf.stack([sx[1] - sx[0],
                                sx[2] - sx[1],
                                sx[0] - sx[2]], name="x_vector")
            x_vector = tf.divide(x_vector, tf.norm(x_vector, axis=1, keep_dims=True), name="x_vector_norm")
            horizon = tf.constant([[1], [0]], dtype=tf.float64)
            theta = tf.acos(tf.matmul(x_vector, horizon))
            theta = tf.reshape(theta, (-1,))
            theta = tf.multiply(theta, (-1) ** tf.cast((x_vector[:, 1] < 0), tf.float64), name="theta")
            rot_y = tf.stack([-tf.sin(theta), tf.cos(theta)], axis=0, name="rot_y")
            rot_y = tf.expand_dims(rot_y, axis=0)
            slided = tf.expand_dims(xy, axis=2) -\
                tf.expand_dims(tf.transpose(sx, (1, 0)), axis=0, name="slided")
            
            adopt = tf.multiply(slided, rot_y, name="rotated")
            adopt = tf.reduce_sum(adopt, axis=1)
            adopt = adopt <= 0
            adopt = tf.reduce_all(adopt, axis=1, name="adopt")
            indices = tf.where(adopt)[:, 0]
            bbox_pixels = tf.concat([ij, xy], axis=1)
            bbox_pixels = tf.gather(bbox_pixels, indices)
            # 法線
            normal = tf.cross(vs[0], vs[1])
            normal = normal / tf.norm(normal)
            # 分子
            numerator = tf.matmul(tf.reshape(normal, (1, -1)), tf.reshape(vs[0], (-1, 1))) * input_screan_z
            screan_xy = bbox_pixels[:, 2:4]
            screan_xyz = tf.concat([screan_xy, tf.tile([[input_screan_z]], (tf.shape(bbox_pixels)[0], 1))], axis=1)
            denominator = tf.matmul(screan_xyz, tf.reshape(normal, (-1, 1)))
            z = numerator / denominator


            color = tf.constant([[255., 255., 255.]], tf.float64)
            color = tf.tile(color, (tf.shape(bbox_pixels)[0], 1))
            bbox_pixels = tf.concat([bbox_pixels, z, color],axis=1, name="bbox_pixels")
            gap = self.image_size[0] * self.image_size[1] - tf.shape(bbox_pixels)[0]
            padding = [(0, gap), (0, 0)]
            pixels = tf.pad(bbox_pixels, padding, mode="CONSTANT", constant_values=outside_indices)
            return pixels

        # i, j, x, y, d, r, g, b
        pixels = tf.zeros((0, 8), dtype=tf.float64)
        pixels = tf.map_fn(body, elems=(screan_xy, vertexes), dtype=tf.float64)
        pixels = tf.reshape(pixels, (-1, 8))
        adopt = tf.logical_not(tf.equal(pixels[:, 0], outside_indices))
        adopt_indices = tf.where(adopt)[:, 0]
        pixels = tf.gather(pixels, adopt_indices)
        # 奥にあるピクセルを除外
        # 同じピクセルかどうか
        is_same_pixel = tf.equal(tf.reshape(pixels[:, :2], (-1, 1, 2)), tf.reshape(pixels[:, :2], (1, -1, 2)))
        is_same_pixel = tf.reduce_all(is_same_pixel, axis=2)

        # 相手より奥か
        is_deeper = tf.reshape(pixels[:, 4], (-1, 1)) >= tf.reshape(pixels[:, 4], (1, -1))

        # 見えないピクセル
        is_covered = tf.logical_and(is_same_pixel, is_deeper)
        invisible = tf.reduce_all(tf.logical_and(is_same_pixel, is_covered), axis=1)
        visible = tf.logical_not(invisible)
        visible_indices = tf.where(visible)[:, 0]
        pixels = tf.gather(pixels, visible_indices)

        pixel_indices = tf.cast(tf.range(tf.shape(pixels)[0]), dtype=tf.int64)
        sparse_indices = tf.cast(pixels[:, :2], dtype=tf.int64)
        sparse_indices = sparse_indices * tf.constant([[-1, 1]], dtype=tf.int64)
        sparse_indices = tf.add(sparse_indices, tf.constant(np.array(self.image_size)/2, dtype=tf.int64), name="sparse_img")
        all_indices = sparse_indices[:, 0] * self.image_size[1] + sparse_indices[:, 1]
        indices = tf.nn.top_k(all_indices, k=tf.shape(sparse_indices)[0]).indices[::-1]
        sparse_indices = tf.gather(sparse_indices, indices, name="sparse_indices")
        
        sparse_pixels = tf.SparseTensor(indices=sparse_indices, values=pixel_indices, dense_shape=self.image_size)
        background_indices = tf.cast(tf.shape(pixels)[0], dtype=tf.int64)
        img_pixels = tf.sparse_tensor_to_dense(sparse_pixels, default_value=background_indices)
        img_pixels = tf.reshape(img_pixels, (-1,))
        pixels_color = tf.concat([pixels[:, 5:], [[0, 0, 0]]], axis=0)
        img = tf.gather(pixels_color, img_pixels)
        img = tf.reshape(img, self.image_size+(3,), "img")
        
        return input_vertexes, input_cam_pos, input_cam_vector, input_screan_z, img

def main():
    image_size = (100, 100)
    graphicker = Graphicker(image_size, img_scale=500)
    v = [[-1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, -1.0, 0.0],
         [0.0, 0.0, -1.0],
         [1.0, 0.0, 0.0]]
    faces = [[v[0], v[2], v[1]],
             [v[0], v[3], v[2]],
             [v[0], v[4], v[3]],
             [v[0], v[1], v[4]],
             [v[1], v[2], v[4]],
             [v[5], v[4], v[2]],
             [v[5], v[2], v[3]],
             [v[5], v[3], v[4]]]
    graphicker.append(faces)
    omega = 0
    t = 0
    phi0 = 270
    r = 10
    while True:
        cam_pos = np.array([-r*np.cos(omega*t+phi0), 0, r*np.sin(omega*t+phi0)])
        cam_pos = np.array([-10, 0, 10])
        print("{:.2f}, {:.2f}, {:.2f}".format(cam_pos[0], cam_pos[1], cam_pos[2]))
        cam_vector = np.array([0, 0, 0]) - cam_pos
        cam_vector = cam_vector / np.linalg.norm(cam_vector)
        screan_z = 1.0
        img = graphicker.draw(cam_pos, cam_vector, screan_z)
        cv2.imshow("img", img)
        cv2.waitKey(10)
        t += 0.1

if __name__ == "__main__":
    main()