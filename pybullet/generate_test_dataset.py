import os
import math
import random
import yaml
import pybullet as p
import pybullet_data
import numpy as np
import cv2
from glob import glob
from scipy import ndimage
from tqdm import tqdm
from PIL import Image

# helper to extract bounding boxes from a segmentation mask
def mask_find_multi_bboxs(mask):
    return ndimage.find_objects(mask)

class World:
    def __init__(self, soil_directory, mesh_directory, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.soil_dir = soil_directory
        self.mesh_dir = mesh_directory

        # mesh scales
        self.scales = {
            'big_plant':    0.6,
            'cirsium':      1.3,
            'polygonum_v2': 0.004,
            'small_plant':  1.3
        }

        # connect and setup
        self.server_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setAdditionalSearchPath(self.mesh_dir)
        p.setGravity(0, 0, -10)

        # load plane
        plane_urdf = os.path.join(pybullet_data.getDataPath(), 'plane.urdf')
        self.plane_id = p.loadURDF(plane_urdf)

        # load soil textures
        textures = glob(os.path.join(self.soil_dir, '*.jpg'))
        self.texture_ids = [p.loadTexture(t) for t in textures]

        # placeholder for objects
        self.object_ids = []

    def spawn_multiple(self, plant_name, count=3):
        # remove old
        for oid in self.object_ids:
            p.removeBody(oid)
        self.object_ids.clear()

        # spawn N copies with same mesh
        mesh_file = f"{plant_name}.stl"
        scale = [self.scales[plant_name]] * 3
        for _ in range(count):
            col = p.createCollisionShape(p.GEOM_MESH,
                                        fileName=mesh_file,
                                        meshScale=scale)
            vis = p.createVisualShape(p.GEOM_MESH,
                                      fileName=mesh_file,
                                      meshScale=scale)
            oid = p.createMultiBody(baseMass=0.0,
                                     baseCollisionShapeIndex=col,
                                     baseVisualShapeIndex=vis,
                                     basePosition=[0,0,0.05])
            self.object_ids.append(oid)

    def change_plane(self):
        tex = random.choice(self.texture_ids)
        p.changeVisualShape(self.plane_id, -1, textureUniqueId=tex)

    def reset_objects(self):
        anchor = [random.random(), random.random(), 0.05]
        shifts = [(-random.uniform(0.2,0.3), -random.uniform(0.2,0.3),0),
                  (0,0,0),
                  (random.uniform(0.2,0.3), random.uniform(0.2,0.3),0)]
        for oid, shift in zip(self.object_ids, shifts):
            pos = [anchor[i] + shift[i] for i in range(3)]
            p.resetBasePositionAndOrientation(oid, pos, [0,0,0,1])
            color = [0, random.uniform(0.4,0.8),0, random.uniform(0.4,0.8)]
            p.changeVisualShape(oid, -1, rgbaColor=color)
        return anchor

    def render(self, width=640, height=640):
        self.change_plane()
        anchor = self.reset_objects()

        # camera
        r = 0.8 + 0.4 * random.random()
        theta = 2*math.pi*random.random()
        h = 0.8 + 0.4 * random.random()
        cam = [anchor[0] + r*math.sin(theta), anchor[1] + r*math.cos(theta), anchor[2] + h]
        tgt = [anchor[0]-0.2*random.random()+0.1,
               anchor[1]-0.2*random.random()+0.1,
               anchor[2]-0.1*random.random()+0.05]
        view = p.computeViewMatrix(cam, tgt, [0,0,1])
        proj = p.computeProjectionMatrixFOV(fov=49.1, aspect=width/height, nearVal=0.1, farVal=100)

        _, _, rgb, _, seg = p.getCameraImage(width=width, height=height,
                                             viewMatrix=view,
                                             projectionMatrix=proj,
                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.array(rgb)[:,:, :3][:,:, ::-1]
        raw_mask = np.array(seg, dtype=np.uint8)

        # create binary plant mask (0 background, 1 plant)
        bin_mask = (raw_mask > 0).astype(np.uint8)

        # compute 3 bboxes
        bboxes = mask_find_multi_bboxs(raw_mask)
        gt = np.zeros((3,4))
        for i, sl in enumerate(bboxes):
            sy, sx = sl
            gt[i] = [sx.start/width, sy.start/height,
                     sx.stop/width, sy.stop/height]
        return rgb, bin_mask, gt

    def close(self):
        p.disconnect(self.server_id)

if __name__ == '__main__':
    # load config
    with open(os.path.join('..','config.yml'),'r') as f:
        cfg = yaml.safe_load(f)
    root = os.path.expanduser(cfg['pybullet_dataset'])
    soil_dir = os.path.join(root,'input','soil_resized')
    mesh_dir = os.path.join(root,'input','weedbot_simulation','STLs')
    output = os.path.join(root,'output2')

    plants = ['big_plant','small_plant','polygonum_v2','cirsium']
    os.makedirs(os.path.join(output,'images'), exist_ok=True)
    os.makedirs(os.path.join(output,'masks'), exist_ok=True)
    os.makedirs(os.path.join(output,'labels'), exist_ok=True)

    world = World(soil_directory=soil_dir, mesh_directory=mesh_dir)
    plant = 'polygonum_v2'
    world.spawn_multiple(plant, count=3)

    for i in tqdm(range(400)):
        rgb, mask, gt = world.render()
        # save rgb
        img_p = os.path.join(output,'images', f'{plant}_{i:04d}.jpg')
        Image.fromarray(rgb).save(img_p)
        # save binary mask
        mask_p = os.path.join(output,'masks', f'{plant}_{i:04d}.png')
        Image.fromarray(mask*255, mode='L').save(mask_p)
        # save labels
        lbl_p = os.path.join(output,'labels', f'{plant}_{i:04d}.txt')
        h, w = mask.shape
        with open(lbl_p,'w') as lf:
            for j in range(3):
                cx = (gt[j,0]+gt[j,2])/2
                cy = (gt[j,1]+gt[j,3])/2
                bw = gt[j,2]-gt[j,0]
                bh = gt[j,3]-gt[j,1]
                lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    world.close()

