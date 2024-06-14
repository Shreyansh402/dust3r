import json
import os
import numpy as np
from plyfile import PlyData, PlyElement

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def mergePoint3D(pts):
    pts = np.array(pts)
    pts = pts.reshape(-1, 3)
    
    

def storePly(path, xyz, rgb):
    xyz[:,1]*=-1
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def outputJSON(saveoutdir, poses, focal_lengths, principal_points, imgs, depths_unnorm, depths,xyss,points3D_IDs):
    intrinsics= {}
    extrinsics={}
    depth_frame={}
    frames={}
    output={}
    length=len(imgs)
    for i, pose in enumerate(poses):
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
        intrinsics['xys']=xyss[i]
        intrinsics['points3D_ID']=points3D_IDs[i]
        frames['image_name']=imgs[i]['name']
        extrinsics['qw']=pose[0]
        extrinsics['qx']=pose[1]
        extrinsics['qy']=pose[2]
        extrinsics['qz']=pose[3]
        if (length > 2):
            extrinsics['tx']=pose[4]
            extrinsics['ty']=pose[5]
            extrinsics['tz']=pose[6]
        else:
            extrinsics['tx']=0
            extrinsics['ty']=0
            extrinsics['tz']=0
        frames['extrinsics']=extrinsics
        depth_frame['image_name']=imgs[i]['name']
        depth_frame['depth']=depths_unnorm[i]
        depth_frame["depth_norm"]=depths[i]
        output[i]['intrinsics']=intrinsics
        output[i]['frames']=frames
        output[i]['depth_frame']=depth_frame
    with open(os.path.join(saveoutdir, 'output.json'), 'w') as f:
        f.write(json.dumps(output,cls=NumpyEncoder, indent=4))
    return output