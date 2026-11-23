"""GLB binary preservation and fixed-rotation FPS bbox registration."""
import copy
import json
import struct

import numpy as np
from scipy.spatial.transform import Rotation


def load_glb(path):
    data = path.read_bytes()
    magic, version, size = struct.unpack_from('<III', data)
    assert (magic, version, size) == (0x46546C67, 2, len(data))
    length, kind = struct.unpack_from('<II', data, 12)
    assert kind == 0x4E4F534A
    doc = json.loads(data[20:20+length])
    offset = 20+length
    length, kind = struct.unpack_from('<II', data, offset)
    assert kind == 0x004E4942
    return doc, data[offset+8:offset+8+length]


def accessor(doc, binary, index):
    """GLTF accessor -> numeric ndarray [N,C] with the declared component dtype."""
    a = doc['accessors'][index]
    v = doc['bufferViews'][a['bufferView']]
    dtype = np.dtype({5126: '<f4', 5125: '<u4', 5123: '<u2'}[a['componentType']])
    channels = {'VEC3': 3, 'VEC2': 2, 'SCALAR': 1}[a['type']]
    return np.ndarray((a['count'], channels), dtype=dtype, buffer=binary,
        offset=v.get('byteOffset', 0)+a.get('byteOffset', 0),
        strides=(v.get('byteStride', channels*dtype.itemsize), dtype.itemsize)).copy()


def write_glb(path, doc, binary, matrix):
    """matrix float64 [4,4] maps original mesh-local vertices to world; binary textures stay byte-identical."""
    doc = copy.deepcopy(doc)
    assert len(doc['nodes']) == 1
    for key in ('translation', 'rotation', 'scale'):
        doc['nodes'][0].pop(key, None)
    doc['nodes'][0]['matrix'] = matrix.flatten(order='F').tolist()
    payload = json.dumps(doc, separators=(',', ':')).encode()
    payload += b' '*((-len(payload)) % 4)
    binary += b'\0'*((-len(binary)) % 4)
    path.write_bytes(struct.pack('<III', 0x46546C67, 2, 28+len(payload)+len(binary))
        + struct.pack('<II', len(payload), 0x4E4F534A)+payload
        + struct.pack('<II', len(binary), 0x004E4942)+binary)


def node_matrix(node):
    """glTF node -> local-to-condition float64 [4,4], with the glTF identity defaults."""
    if 'matrix' in node:
        assert not {'translation', 'rotation', 'scale'} & node.keys()
        return np.asarray(node['matrix'], np.float64).reshape(4, 4, order='F')
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(node.get('rotation', [0., 0., 0., 1.])).as_matrix()
    matrix[:3, :3] *= np.asarray(node.get('scale', [1., 1., 1.]))[None]
    matrix[:3, 3] = node.get('translation', [0., 0., 0.])
    return matrix


def register_inverse_fps(raw_points, target_world, inverse, world_to_bbox):
    """raw_points float64 [N,3], target_world float64 [4096,3], inverse/world_to_bbox float64 [4,4] -> raw-to-world float64 [4,4], metadata."""
    assert raw_points.ndim == 2 and raw_points.shape[1] == 3
    assert target_world.shape == (4096, 3)
    assert inverse.shape == world_to_bbox.shape == (4, 4)
    raw_to_bbox = world_to_bbox @ inverse
    source = raw_points @ raw_to_bbox[:3, :3].T + raw_to_bbox[:3, 3]
    target = target_world @ world_to_bbox[:3, :3].T + world_to_bbox[:3, 3]
    source_bounds = np.percentile(source, [0.25, 99.75], axis=0)
    target_bounds = np.percentile(target, [0.25, 99.75], axis=0)
    assert np.ptp(source_bounds, axis=0).max() > 0
    scale = np.ptp(target_bounds, axis=0).max() / np.ptp(source_bounds, axis=0).max()
    offset = target_bounds.mean(0) - scale * source_bounds.mean(0)
    correction = np.eye(4)
    correction[:3, :3] *= scale
    correction[:3, 3] = offset
    world_correction = np.linalg.solve(world_to_bbox, correction @ world_to_bbox)
    assert np.allclose(world_correction[:3, :3], np.eye(3) * scale, atol=1e-10)
    return world_correction @ inverse, dict(
        method='saved condition inverse, then fixed-rotation robust bbox to unnormalized filtered-Gaussian FPS',
        targetFile='pointcloud-fps-world.json', targetPoints=len(target_world), meshSamples=len(raw_points),
        fixedRotation=True, rotationCorrectionDegrees=0, quantilesPercent=[0.25, 99.75],
        uniformScaleAfterInverse=float(scale), translationCanonical=offset.tolist(),
        sourceBoundsCanonical=source_bounds.tolist(), targetBoundsCanonical=target_bounds.tolist(),
        correctionCanonical=correction.flatten(order='F').tolist(),
        correctionWorld=world_correction.flatten(order='F').tolist())
