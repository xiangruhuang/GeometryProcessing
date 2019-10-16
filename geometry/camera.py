import open3d as o3d
import numpy as np

class PinholeCamera:
    def __init__(self, extrinsic=None):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window('ply', 320, 240, 50, 50, False)
        

    """ Inverse projection, from depth image to 3D point cloud
    Input:
        depth: np.ndarray of shape [W, H] (0 indicates invalid depth).
    Output:
        points: np.ndarray of shape [N, 3].
    """
    def depth2pointcloud(self, depth, ext=None, intc=None):
        ctr = self.vis.get_view_control()
        pinhole_params = ctr.convert_to_pinhole_camera_parameters()
        if intc is not None:
          intrinsic = intc
        else:
          intrinsic = pinhole_params.intrinsic.intrinsic_matrix
        if ext is not None:
          extrinsic = ext
        else:
          extrinsic = pinhole_params.extrinsic
        R = extrinsic[:3, :3]
        trans = extrinsic[:3, 3]
        x, y = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing='ij')
        valid_idx = np.where(depth > 1e-7)
        z = depth[valid_idx]
        x = x[valid_idx]*z
        y = y[valid_idx]*z
        flip = np.array([[0,1,0],
                         [1,0,0],
                         [0,0,1]])
        points = flip.dot(np.stack([x, y, z], axis=1).T).T

        points = R.T.dot(np.linalg.pinv(intrinsic).dot(points.T)-trans[:, np.newaxis]).T
        return points

    """ Project Triangle Mesh to depth image.
    Input:
        mesh: o3d.geometry.TriangleMesh object
        intersecting_triangles: if True, we are also collecting correspondences.
    Output:
        depth: np.ndarray of shape [W, H].
    Additional Output (if intersecting_triangles=True):
        points3d: np.ndarray of shape [M, 3].
                  point cloud in 3D, M is number of valid pixels/points.
        correspondences: np.ndarray of shape [M] and dtype np.int32
                         each entry lies in range [-1, N],
                         -1 indicates invalid.
        valid_pixel_indices: np.ndarray of shape [M, 2],
                             each row contains a valid pixel coordinates.
    """
    def project(self, mesh, intersecting_triangles=False):
        self.vis.add_geometry(mesh)
        depth = self.vis.capture_depth_float_buffer(True)
        depth = np.array(depth)
        ctr = self.vis.get_view_control()
        pinhole_params = ctr.convert_to_pinhole_camera_parameters()
        intrinsic = pinhole_params.intrinsic.intrinsic_matrix
        extrinsic = pinhole_params.extrinsic
        if intersecting_triangles:
            """ retrieve camera extrinsic """
            R = extrinsic[:3, :3]
            trans = extrinsic[:3, 3]
            """ Hash depth pixels """
            valid_idx = np.where(depth > 1e-7)
            x, y = np.meshgrid(np.arange(depth.shape[0]),
                               np.arange(depth.shape[1]), indexing='ij')
            z = depth[valid_idx]
            x = x[valid_idx]
            y = y[valid_idx]
            points3d = np.stack([y*z, x*z, z], axis=1)
            points3d = R.T.dot(np.linalg.pinv(intrinsic).dot(points3d.T)-
                               trans[:, np.newaxis]).T
            #hash_table = {}
            #for i in range(x.shape[0]):
            #    xi = x[i]
            #    yi = y[i]
            #    point = points[i, :]
            #    arr = hash_table.get((xi, yi), None)
            #    if arr is None:
            #        hash_table[(xi, yi)] = []
            #    hash_table[(xi, yi)].append((point, i))

            """ valid indices """
            vertices = np.array(mesh.vertices)
            from sklearn.neighbors import NearestNeighbors as NN
            tree = NN(n_neighbors=1, algorithm='kd_tree', n_jobs=10).fit(vertices)
            dists, indices = tree.kneighbors(points3d)
            print(dists.mean())
            ############################## DEBUG ############################
            #p1 = o3d.geometry.PointCloud()
            #p1.points = o3d.utility.Vector3dVector(vertices)
            #p1.paint_uniform_color([0,1,0])
            #p2 = o3d.geometry.PointCloud()
            #p2.points = o3d.utility.Vector3dVector(points)
            #p2.paint_uniform_color([0,0,1])
            #o3d.draw_geometries([p1, p2])
            ##########################################################

            correspondences = indices[:, 0]

            #import ipdb; ipdb.set_trace()
            #vertices3d = intrinsic.dot(R.dot(vertices.T)+trans.reshape((3, 1))).T
            #vertices2d = np.array(vertices3d)
            #vertices2d[:, 0] = vertices2d[:, 0] / vertices2d[:, 2]
            #vertices2d[:, 1] = vertices2d[:, 1] / vertices2d[:, 2]
            #triangles = np.array(mesh.triangles)
            #corresponding_triangles = np.zeros(points.shape[0])
            #for f in range(triangles.shape[0]):
            #    i1 = triangles[f, 0]
            #    i2 = triangles[f, 1]
            #    i3 = triangles[f, 2]
            #    v1 = vertices2d[i1, :2]
            #    v2 = vertices2d[i2, :2]
            #    v3 = vertices2d[i3, :2]
            #    vs = np.stack([v1,v2,v3], axis=1) # [2, 3]
            #    import ipdb; ipdb.set_trace()
            #    min_x, min_y = np.floor(vs.min(axis=1))
            #    max_x, max_y = np.ceil(vs.max(axis=1))
            #    for xx in range(min_x, max_x+1):
            #        for yy in range(min_y, max_y+1):
            #            points_in_cell = hash_table.get((xx, yy), None)
            #            if points_in_cell is None:
            #                continue
            #            for point, pid in points_in_cell:
            #                p1 = vertices3d[i1, :]
            #                p2 = vertices3d[i2, :]
            #                p3 = vertices3d[i3, :]
            #                check_if_on_triangle(point, p1, p2, p3)
            #                corresponding_triangles[pid] = f
            #            
            #import matplotlib.pyplot as plt
            #plt.scatter(vertices2d[:, 0], vertices2d[:, 1], 0.1)
            #plt.show()
            self.vis.remove_geometry(mesh)
            valid_pixel_indices = np.stack([valid_idx[0], valid_idx[1]], axis=1)
            return depth, extrinsic, intrinsic, points3d, correspondences, valid_pixel_indices
        else:
            self.vis.remove_geometry(mesh)
            return depth, extrinsic, intrinsic

if __name__ == '__main__':
    camera = PinholeCamera()
    mesh_male = o3d.io.read_triangle_mesh('example_data/mesh_male.ply')
    camera.set_extrinsic(np.diag([1,-1,1]), np.zeros(3))
    depth_image, index_mask = camera.project(mesh_male, intersecting_triangles=True)
    import matplotlib.pyplot as plt
    plt.imshow(depth_image)
    plt.show()
    mesh_male.transform(np.diag([1,-1,1, 1]))
    depth_image, index_mask = camera.project(mesh_male, intersecting_triangles=True)
    import matplotlib.pyplot as plt
    plt.imshow(depth_image)
    plt.show()
    points = camera.depth2pointcloud(depth_image)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    o3d.draw_geometries([pcd])
