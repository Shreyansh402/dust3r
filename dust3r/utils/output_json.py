import json
import os
import numpy as np
from plyfile import PlyData, PlyElement
import trimesh

from dust3r.utils.device import to_numpy                            

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
    
def get_pc(imgs, pts3d, mask):
        imgs = to_numpy(imgs)
        pts3d = to_numpy(pts3d)
        mask = to_numpy(mask)

        pts = np.concatenate([p for p in pts3d])
        col = np.concatenate([p for p in imgs])
        
        # pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        # col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        
        pts = pts.reshape(-1, 3)[::3]
        col = col.reshape(-1, 3)[::3]
        
        #mock normals:
        normals = np.tile([0, 1, 0], (pts.shape[0], 1))
        
        pct = trimesh.PointCloud(pts, colors=col)
        pct.vertices_normal = normals  # Manually add normals to the point cloud
        
        return pct

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def storePly(path, xyz, rgb):
    #xyz[:,1]*=-1
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    #rgb is 4 dimensional, so we need to remove the alpha channel
    rgb = rgb[:, :3]
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def outputJSON(saveoutdir, poses, focal_lengths, principal_points, imgs):
    output={}
    intrinsics={}
    frames={}
    extrinsics={}
    depth_frame=None
    for i in range(poses.shape[0]):
        rotation_matrix = poses[i, :3, :3]
        qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
        tx, ty, tz = poses[i, :3, 3]
        output[i]={}
        intrinsics['height']=imgs[i]['true_shape'][0][0]
        intrinsics['width']=imgs[i]['true_shape'][0][1]
        try:
            intrinsics['fx']=focal_lengths[i][0]
            intrinsics['fy']=focal_lengths[i][0]
        except IndexError:
            intrinsics['fx']=focal_lengths[0]
            intrinsics['fy']=focal_lengths[0]
        intrinsics['cx']=principal_points[i][0]
        intrinsics['cy']=principal_points[i][1]
        intrinsics['xys']=None
        intrinsics['points3D_ID']=None
        #frames['image_name']=imgs[i]['name']
        frames['image_name']="frame_{:04d}.png".format(i)
        extrinsics['qw']=qw
        extrinsics['qx']=qx
        extrinsics['qy']=qy
        extrinsics['qz']=qz
        extrinsics['tx']=tx
        extrinsics['ty']=ty
        extrinsics['tz']=tz
        frames['extrinsics']=dict(extrinsics)
        # depth_frame['image_name']=imgs[i]['name']
        # depth_frame['depth']=depths_unnorm[i]
        # depth_frame["depth_norm"]=depths[i]
        output[i]['intrinsics']=dict(intrinsics)
        output[i]['frames']=dict(frames)
        #output[i]['depth_frame']=dict(depth_frame)
        output[i]['depth_frame']=depth_frame
    with open(os.path.join(saveoutdir, 'output.json'), 'w') as f:
        f.write(json.dumps(output,cls=NumpyEncoder, indent=4))
    return output