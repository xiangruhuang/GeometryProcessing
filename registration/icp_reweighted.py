import numpy as np
import open3d as o3d
import sys
sys.addpath('../')
from sklearn.neighbors import NearestNeighbors as NN
import geometry.util as gutil

def icp_reweighted(source, target, sigma=0.01, stopping_threshold=1e-4):
    tree = NN(n_neighbors=1, algorithm='kd_tree', n_jobs=10)
    tree = tree.fit(np.array(target.points))
    n = np.array(source.points).shape[0]
    normals = np.array(target.normals)
    points = np.array(target.points)
    weights = np.zeros(n)
    errors = []
    transform = np.eye(4)

    for itr in range(100):
        p = np.array(source.points)
        R, trans = gutil.unpack(transform)
        p = (R.dot(p.T) + trans.reshape((3, 1))).T
        _, indices = tree.kneighbors(p)

        """ (r X pi + pi + t - qi)^T ni """
        """( <r, (pi X ni)> + <t, ni> + <pi-qi, ni> )^2"""
        """ (<(r; t), hi> + di)^2 """
        nor = normals[indices[:, 0], :]
        q = points[indices[:, 0], :]
        d = np.sum(np.multiply(p-q, nor), axis=1) #[n]
        h = np.zeros((n, 6))
        h[:, :3] = np.cross(p, nor)
        h[:, 3:] = nor
        weight = (sigma**2)/(np.square(d)+sigma**2)
        H = np.multiply(h.T, weight).dot(h)
        g = -h.T.dot(np.multiply(d, weight))
        delta = np.linalg.solve(H, g)
        errors = np.abs(d)
        print('iter=%d, delta=%f, mean error=%f, median error=%f' % (
                itr, np.linalg.norm(delta, 2),
                np.mean(errors), np.median(errors)))
        if np.linalg.norm(delta, 2) < stopping_threshold:
            break
        trans = delta[3:]
        R = gutil.rodrigues(delta[:3])
        T = gutil.pack(R, trans)
        transform = T.dot(transform)

    return transform

def main():
    import argparse
    parser = argparse.ArgumentParser(description='reweighted ICP algorithm')
    parser.add_argument('--source', type=str,
                        help='source point cloud or mesh in .ply format')
    parser.add_argument('--target', type=str,
                        help='target point cloud or mesh in .ply format')
    args = parser.parse()

    source = o3d.read_point_cloud(args.source)
    mesh = o3d.read_triangle_mesh(args.target)

    v = np.array(mesh.vertices)
    tri = np.array(mesh.triangles)
    v1 = v[tri[:, 0], :]
    v2 = v[tri[:, 1], :]
    v3 = v[tri[:, 2], :]
    normals = np.cross(v1-v3, v2-v3)
    normals = (normals.T / np.linalg.norm(normals, 2, axis=1)).T
    centers = (v1+v2+v3)/3.0

    target = o3d.PointCloud()
    target.points = o3d.Vector3dVector(centers)
    target.normals = o3d.Vector3dVector(normals)
    #o3d.draw_geometries([target, source])

    errors = icp_reweighted(source, target)
    threshold = 10.0
    indices = np.where(errors < threshold)[0]
    while indices.shape[0] < min(8000, errors.shape[0]):
        threshold += 0.001
        indices = np.where(errors < threshold)[0]
    print('threshold=%f' % threshold)
    p = np.array(source.points)[indices, :]
    source.points = o3d.Vector3dVector(p)

    o3d.write_point_cloud(sys.argv[3], source)

if __name__ == '__main__':
    main()
