import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

trans_t = lambda t : torch.Tensor([
    [1,0,0,t],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])) @ c2w
    return c2w

def pose_around(theta1, theta2 , c2w):
    #c2w = rot_theta(theta / 180.0 * np.pi).cpu() @ c2w
    c2w = trans_t(theta1).cpu() @ rot_theta(theta2 / 180.0 * np.pi).cpu() @ c2w
    return c2w


def load_audface_data(basedir, testskip=1, test_file=None, aud_file=None, train_length=None, need_lip=False, need_torso=False, bc_type='torso_bc_imgs'):
    if test_file is not None:
        print(basedir)
        print(test_file)
        with open(os.path.join(basedir, test_file)) as fp:
            meta = json.load(fp)
        poses = []
        auds = []
        imgs = []
        lip_rects = []
        torso_bcs = []
        aud_features = np.load(os.path.join(basedir, aud_file))

        aud_start = 0
        for frame in meta['frames'][::testskip]:
            poses.append(np.array(frame['transform_matrix']))
            if aud_file == 'aud.npy':
                auds.append(aud_features[min(frame['aud_id'], aud_features.shape[0]-1)])
            else:
                auds.append(aud_features[min(aud_start, aud_features.shape[0]-1)])
                aud_start = aud_start+1
            fname = os.path.join(basedir, 'com_imgs', str(frame['img_id']) + '.jpg')
            imgs.append(fname)
            lip_rects.append(np.array(frame['lip_rect'], dtype=np.int32))
            torso_bcs.append(os.path.join(basedir, bc_type, str(frame['img_id']) + '.jpg'))
        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        lip_rects = np.array(lip_rects).astype(np.int32)
        #target = torch.as_tensor(imageio.imread(image_path)).to(device).float() / 255.0
        bc_img = imageio.imread(os.path.join(basedir, 'bc.jpg'))
        H, W = bc_img.shape[0], bc_img.shape[1]
        focal, cx, cy = float(meta['focal_len']), float(meta['cx']), float(meta['cy'])
        avg_c2w = poses.mean(0)
        #render_poses = torch.stack([pose_around(angle, -10, avg_c2w) for angle in [0.009, 0.011, 0.013, 0.015, 0.017]], 0, ).cuda()
        render_poses = torch.stack([pose_around(angle, 5, avg_c2w) for angle in [-0.017, -0.009, -0.005, 0.005, 0.009, 0.017]], 0, ).cuda()
        #render_poses = torch.stack([pose_around(angle, -10, avg_c2w) for angle in [0.017, 0.02]], 0, ).cuda()
        if need_lip:
            if need_torso:
                return imgs, poses, auds, bc_img, [H, W, focal, cx, cy], lip_rects, torso_bcs, render_poses
            else:
                return imgs, poses, auds, bc_img, [H, W, focal, cx, cy], lip_rects, None, render_poses
        else:
            if need_torso:
                return imgs, poses, auds, bc_img, [H, W, focal, cx, cy], torso_bcs
            else:
                return imgs, poses, auds, bc_img, [H, W, focal, cx, cy], None

    #every id has a dir
    id_list = sorted(os.listdir(os.path.join(basedir)))

    id_num = len(id_list)
    metas = {}
    all_imgs = {}
    all_poses = {}
    all_auds = {}
    all_sample_rects = {}
    all_lip_rects = {}
    counts= {}

    i_split = {}
    bc_img = {}
    all_torso_bcs = {}

    splits = ['train', 'val']
    for i in id_list:#range id
        metas[i] = {}
        for s in splits:
            if s in ['train']:
                with open(os.path.join(basedir, i, 'transforms_{}.json'.format(s)), 'r') as fp:
                    metas[i][s] = json.load(fp)
            else:
                with open(os.path.join(basedir, i, 'transforms_{}.json'.format(s)), 'r') as fp:
                    metas[i][s] = json.load(fp)
        all_imgs[i] = []
        all_poses[i] = []
        all_auds[i] = []
        all_sample_rects[i] = []
        all_lip_rects[i] = []
        all_torso_bcs[i] = []

        aud_features = np.load(os.path.join(basedir, i, 'aud.npy'))
        counts[i] = [0]
        for s in splits:
            meta = metas[i][s]
            imgs = []
            poses = []
            auds = []
            sample_rects = []
            lip_rects = []
            #mouth_rects = []
            #exps = []
            torso_bcs = []

            if s == 'train' or testskip == 0:
                skip = 1
            else:
                skip = testskip

            for frame in meta['frames'][::skip]:
                fname = os.path.join(basedir, i , 'com_imgs',
                                     str(frame['img_id']) + '.jpg')
                imgs.append(fname)
                poses.append(np.array(frame['transform_matrix']))
                auds.append(
                    aud_features[min(frame['aud_id'], aud_features.shape[0]-1)])
                sample_rects.append(np.array(frame['face_rect'], dtype=np.int32))
                lip_rects.append(np.array(frame['lip_rect'], dtype=np.int32))
                #add bc
                torso_bcs.append(os.path.join(basedir, i , bc_type,
                                     str(frame['img_id']) + '.jpg'))

            imgs = np.array(imgs)
            poses = np.array(poses).astype(np.float32)
            auds = np.array(auds).astype(np.float32)
            torso_bcs = np.array(torso_bcs)
            counts[i].append(counts[i][-1] + imgs.shape[0])
            all_imgs[i].append(imgs)
            all_poses[i].append(poses)
            all_auds[i].append(auds)
            all_sample_rects[i].append(sample_rects)
            all_lip_rects[i].append(lip_rects)
            all_torso_bcs[i].append(torso_bcs)

        i_split[i] = [np.arange(counts[i][j], counts[i][j+1]) for j in range(len(splits))]
        all_imgs[i] = np.concatenate(all_imgs[i], 0)
        all_torso_bcs[i] = np.concatenate(all_torso_bcs[i], 0)
        all_poses[i] = np.concatenate(all_poses[i], 0)
        all_auds[i] = np.concatenate(all_auds[i], 0)
        all_sample_rects[i] = np.concatenate(all_sample_rects[i], 0)
        all_lip_rects[i] = np.concatenate(all_lip_rects[i], 0)

        bc_img[i] = imageio.imread(os.path.join(basedir, i, 'bc.jpg'))

    H, W = bc_img[i].shape[:2]
    focal, cx, cy = float(meta['focal_len']), float(meta['cx']), float(meta['cy'])
    #if need the lip region, can return the all_lip_rects
    if need_lip:
        return all_imgs, all_poses, all_auds, bc_img, [H, W, focal, cx, cy], all_sample_rects, i_split, id_num, all_lip_rects, all_torso_bcs
    else:
        return all_imgs, all_poses, all_auds, bc_img, [H, W, focal, cx, cy], all_sample_rects, i_split, id_num, all_torso_bcs
