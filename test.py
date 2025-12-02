import argparse
import os
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import *
from tqdm import tqdm
import torch
import numpy as np
from models import LiveHPS
import sys
try:
    import cv2  # for video export using OpenCV
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_VIDEO_DEPS = True
except ImportError:
    HAS_VIDEO_DEPS = False
sys.path.append("./smpl")
from smpl import SMPL, SMPL_MODEL_DIR
from dataset.Livehps_dataset import Dataset
from scipy.spatial.transform import Rotation as R

def gen_smpl(smpl,rot,shape,device):
    num = int(rot.shape[0]/shape.shape[0])
    rot = matrix_to_axis_angle(rotation_6d_to_matrix(rot).view(-1, 3,3)).reshape(-1, 72)
    pose_b = rot[:,3:].float()
    g_r = rot[:,:3].float()
    shape = shape.reshape(-1,1,10).repeat([1,num,1]).reshape(-1,10).float()
    zeros = np.zeros((g_r.shape[0], 3))
    transl_blob = torch.from_numpy(zeros).float().to(device)
    mesh = smpl(betas=shape.to(device),body_pose=pose_b.to(device),global_orient = g_r.to(device),transl=transl_blob)
    joints = mesh.joints[:, :24, :]  # smplx SMPL returns 45 joints; keep 24 like rest of code
    v = mesh.vertices - joints[:,0,:].unsqueeze(1)
    j = joints - joints[:,0,:].unsqueeze(1)
    return v,j

def local2global(pose):
    kin_chains = [
        [20, 18, 16, 13, 9, 6, 3, 0],   # left arm
        [21, 19, 17, 14, 9, 6, 3, 0],   # right arm
        [7, 4, 1, 0],                   # left leg
        [8, 5, 2, 0],                   # right leg
        [12, 9, 6, 3, 0],               # head
        [0],                            # root, hip
    ]
    T = pose.shape[0]
    Rb2l = []
    cache = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    for chain in kin_chains:
        leaf_rotmat = torch.eye(3).unsqueeze(0).repeat(T,1,1)
        for joint in chain:
            joint_rotvec = pose[:, joint*3:joint*3+3]
            joint_rotmat = torch.from_numpy(R.from_rotvec(joint_rotvec.cpu()).as_matrix().astype(np.float32)).to("cpu")
            leaf_rotmat = torch.einsum("bmn,bnl->bml", joint_rotmat, leaf_rotmat)
            cache[joint] = leaf_rotmat
        Rb2l.append(leaf_rotmat)
    return cache

def cal_ang(gt_pose,pose):

    globalR = torch.from_numpy(pose[:, :3]).float()
    gt_matrix = local2global(gt_pose.reshape(-1,72))
    pose_matrix = local2global(torch.from_numpy(pose).reshape(-1,72))
    #print(gt_matrix)
    gt_matrix = torch.from_numpy(np.array([item.cpu().detach().numpy() for item in gt_matrix if item!=None])).reshape(-1,3,3)
    pose_matrix = torch.from_numpy(np.array([item.cpu().detach().numpy() for item in pose_matrix if item!=None])).reshape(-1,3,3)
    gt_axis = quaternion_to_axis_angle(matrix_to_quaternion(gt_matrix))
    pose_axis = quaternion_to_axis_angle(matrix_to_quaternion(pose_matrix))
    #print(gt_axis.shape)
    gt_norm = np.rad2deg(np.linalg.norm(gt_axis.numpy(),axis=1)).reshape((globalR.shape[0],-1,1))
    pose_norm = np.rad2deg(np.linalg.norm(pose_axis.numpy(),axis=1)).reshape((globalR.shape[0],-1,1))
    anger = np.abs((gt_norm-pose_norm)).mean(axis=1).mean()
    
    return anger

def jitter(pred):
    delta_t = 0.1
    pred_velo = (pred[:,1:,:,:]-pred[:,:-1,:,:]) / delta_t
    pred_acc = (pred_velo[:,1:,:,:]-pred_velo[:,:-1,:,:]) / delta_t
    jitter = torch.norm((pred_acc[:,1:,:,:]-pred_acc[:,:-1,:,:]) / delta_t, dim=-1)
    return torch.mean(jitter/100)

def test_one_epoch(model,device,test_loader,smpl,args):
    model.eval()
    test_loss = {
        'mpjpe':[],
        'mpvpe':[],
        'mpjpe-s':[],
        'mpvpe-s':[],
        'mpjpe-ts':[],
        'mpvpe-ts':[],
        'ang':[],
        'CD':[],
        'jitter':[]
    }
    save_outputs = args.save_npz or (args.save_video and HAS_VIDEO_DEPS)
    pred_vertices_all = []
    pred_joints_all = []
    gt_vertices_all = []
    frame_ids = []
    kintree_edges = []
    if args.save_video:
        if hasattr(smpl, 'kintree_table'):
            kt = smpl.kintree_table
            kintree_edges = [(int(kt[0, i]), int(kt[1, i])) for i in range(1, kt.shape[1])]
        elif hasattr(smpl, 'parents'):
            parents = smpl.parents.cpu().numpy()
            kintree_edges = [(int(parents[i]), int(i)) for i in range(1, len(parents))]
        else:
            kintree_edges = []

    for i,data in enumerate(tqdm(test_loader)):
        with torch.no_grad():  
            for j in range(3):
                seq_pc1 = data['data'+str(j+1)].to(device).float()
                if seq_pc1.numel() == 0:
                    continue
                B,T = seq_pc1.shape[0],seq_pc1.shape[1]
                # Reshape gt pose to (B*T*24,3) -> rotation matrices -> 6D representation
                seq_pose_tensor = data['gt_smpl'+str(j+1)].float()
                seq_pose_flat = seq_pose_tensor.view(-1,72)
                seq_pose_np = seq_pose_tensor.cpu().numpy().reshape(-1,3)
                gt_pose = torch.from_numpy(R.from_rotvec(seq_pose_np).as_matrix()).to(device).view(B,T,24,3,3)
                gt_pose = matrix_to_rotation_6d(gt_pose).reshape(B*T,24,6)
                gt_shape = data['shape'].to(device)
                seq_trans = data['T'+str(j+1)].to(device).float().reshape(-1,T,3)

                _,rot,shape,pre_trans = model(seq_pc1.float())

                final_pc = seq_pc1.reshape(B,T,256,3)-pre_trans.reshape(B,T,1,3)
                gt_v,gt_j = gen_smpl(smpl,gt_pose,gt_shape,device)
                pre_v,pre_j = gen_smpl(smpl,rot.reshape(B*T,-1,6),shape,device)
                pre_v_noshape,pre_j_noshape = gen_smpl(smpl,rot.reshape(B*T,-1,6),gt_shape,device)
                dif_tran = seq_trans.reshape(B,T,1,3)-pre_trans.reshape(B,T,1,3)

                loss1 = np.linalg.norm(pre_v[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3)-gt_v[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3),axis=2).mean(axis=1).mean()#(loss_fn(out.float(),seq_gt.float()))
                loss2 = np.linalg.norm(pre_j[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3)-gt_j[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3),axis=2).mean(axis=1).mean()#(loss_fn(out.float(),seq_gt.float()))
                loss11 = np.linalg.norm(pre_v_noshape[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3)-gt_v[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3),axis=2)#.mean(axis=1).mean()#(loss_fn(out.float(),seq_gt.float()))
                loss22 = np.linalg.norm(pre_j_noshape[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3)-gt_j[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3),axis=2).mean(axis=1).mean()#(loss_fn(out.float(),seq_gt.float()))
                
                pose = R.from_matrix(rotation_6d_to_matrix(rot.view(B,T,24,6)).view(-1, 3,3).cpu().detach().numpy()).as_rotvec().reshape(-1, 72)
                loss111 = np.linalg.norm((pre_v.reshape(B,T,-1,3)+dif_tran).reshape(B,T,-1)[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3)-gt_v[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3),axis=2).mean(axis=1)#.mean()#(loss_fn(out.float(),seq_gt.float()))
                loss222 = np.linalg.norm((pre_j.reshape(B,T,-1,3)+dif_tran).reshape(B,T,-1)[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3)-gt_j[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3),axis=2).mean(axis=1)#.mean()#(loss_fn(out.float(),seq_gt.float()))
                
                an = cal_ang(seq_pose_flat,pose)
                loss3,_ = chamfer_distance(final_pc.reshape(B*T,256,3),pre_v.reshape(B*T,-1,3))
                loss_jitter = jitter(pre_j.reshape(B,T,24,3))

                test_loss['mpjpe-s'].append(loss2.item())
                test_loss['mpvpe-s'].append(loss1.item())
                test_loss['mpjpe'].append(loss22.item())
                test_loss['mpvpe'].append(loss11.mean(axis=1).mean().item())
                test_loss['mpjpe-ts'].append(loss222.mean().item())
                test_loss['mpvpe-ts'].append(loss111.mean().item())
                test_loss['ang'].append(an.mean().item())
                test_loss['CD'].append(loss3.mean().item())
                test_loss['jitter'].append(loss_jitter.item())
                if save_outputs and j == 0:
                    pred_vertices_all.append(pre_v.reshape(B*T, -1, 3).cpu().numpy())
                    pred_joints_all.append(pre_j.reshape(B*T, -1, 3).cpu().numpy())
                    gt_vertices_all.append(gt_v.reshape(B*T, -1, 3).cpu().numpy())
                    if 'frame_id' in data:
                        fids = data['frame_id']
                        if isinstance(fids, torch.Tensor):
                            fids = fids.cpu().numpy()
                        if isinstance(fids, np.ndarray):
                            frame_ids.extend([str(x) for x in fids.reshape(-1).tolist()])
                        elif isinstance(fids, list):
                            for el in fids:
                                if isinstance(el, np.ndarray):
                                    frame_ids.extend([str(x) for x in el.reshape(-1).tolist()])
                                else:
                                    frame_ids.append(str(el))
                    else:
                        frame_ids.extend([f"batch{i}_frame{t}" for t in range(B*T)])
    loss_list = []
    for k in test_loss.keys():
        if len(test_loss[k])!=0:
            loss_list.append(np.array(test_loss[k]).mean())
            print(k,np.array(test_loss[k]).mean())
    print(f'{loss_list[0]*1000:.2f}/{loss_list[1]*1000:.2f}&{loss_list[2]*1000:.2f}/{loss_list[3]*1000:.2f}&{loss_list[4]*1000:.2f}/{loss_list[5]*1000:.2f}&{loss_list[6]:.2f}&{loss_list[7]*1000:.2f}/{loss_list[8]:.2f}')

    if save_outputs:
        os.makedirs(args.output_dir, exist_ok=True)
        if len(pred_vertices_all):
            preds_v = np.concatenate(pred_vertices_all, axis=0)
            preds_j = np.concatenate(pred_joints_all, axis=0)
            gts_v = np.concatenate(gt_vertices_all, axis=0)
            np.savez(os.path.join(args.output_dir, 'smpl_outputs.npz'),
                     pred_vertices=preds_v,
                     pred_joints=preds_j,
                     gt_vertices=gts_v,
                     frame_ids=np.array(frame_ids))
            print(f"[Saved] SMPL outputs -> {os.path.join(args.output_dir, 'smpl_outputs.npz')}")
        if args.save_video and HAS_VIDEO_DEPS and len(pred_vertices_all):
            # Use vertices instead of joints for full body mesh rendering
            vertices = np.concatenate(pred_vertices_all, axis=0)  # (N, 6890, 3)
            if vertices.shape[0] > args.video_max_frames:
                vertices = vertices[:args.video_max_frames]
            
            # Create video using OpenCV with mesh rendering
            print(f"Creating video with {vertices.shape[0]} frames using OpenCV...")
            video_path = os.path.join(args.output_dir, 'smpl_pred.mp4')
            width, height = 800, 800
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
            
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Calculate bounds from all vertices
            xs, ys, zs = vertices[:,:,0], vertices[:,:,1], vertices[:,:,2]
            xmin, xmax = xs.min() - 0.1, xs.max() + 0.1
            ymin, ymax = ys.min() - 0.1, ys.max() + 0.1
            zmin, zmax = zs.min() - 0.1, zs.max() + 0.1
            
            # Get SMPL faces for mesh rendering
            smpl_faces = smpl.faces if hasattr(smpl, 'faces') else None
            
            for f in tqdm(range(vertices.shape[0]), desc="Rendering frames"):
                ax.clear()
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])
                ax.set_zlim([zmin, zmax])
                ax.view_init(elev=20, azim=-60 + f * 0.5)  # Slowly rotate camera
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'Frame {f+1}/{vertices.shape[0]}')
                
                verts = vertices[f]
                
                # Render as mesh using plot_trisurf if faces available
                if smpl_faces is not None:
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                                   triangles=smpl_faces, 
                                   color='lightblue', 
                                   edgecolor='none',
                                   alpha=0.9,
                                   shade=True,
                                   linewidth=0)
                else:
                    # Fallback: render as point cloud with smaller points for mesh-like appearance
                    ax.scatter(verts[::5, 0], verts[::5, 1], verts[::5, 2], 
                             c='lightblue', s=2, alpha=0.8)
                
                # Convert matplotlib figure to opencv image
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                img = np.asarray(buf)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                img = cv2.resize(img, (width, height))
                out.write(img)
            
            plt.close(fig)
            out.release()
            print(f"[Saved] Video -> {video_path}")
        elif args.save_video and not HAS_VIDEO_DEPS:
            print("[Warning] OpenCV/matplotlib not installed; skipping video export.")

    return np.array(test_loss['mpvpe']).mean(),np.array(test_loss['mpvpe-s']).mean(),np.array(test_loss['mpvpe-ts']).mean()

def options():
    parser = argparse.ArgumentParser(description='Baseline network')
    parser.add_argument('--save_path',type=str,default='')
    parser.add_argument('-n','--exp_name',type=str,default='')
    parser.add_argument('--root_dataset_path',type=str,default='')
    parser.add_argument('--train',type=bool,default=True)
    parser.add_argument("--local_rank", type=int)

    parser.add_argument('--frames',type=int,default=32)
    parser.add_argument('--num_points',type=int,default=256)

    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--start_epoch',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=200)

    parser.add_argument('--pretrained',default='')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--save_npz', action='store_true', default=True)
    parser.add_argument('--save_video', action='store_true', default=True)
    parser.add_argument('--video_max_frames', type=int, default=300)

    args = parser.parse_args()
    return args

def main():
    args = options()
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    smpl = SMPL(SMPL_MODEL_DIR, create_transl=False).to(device)

    test_dataset = Dataset(args,'e')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,drop_last=False,pin_memory=False)
    model = LiveHPS()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(torch.cuda.device_count())
    model.to(device)

    test_one_epoch(model,device,test_loader,smpl,args)

if __name__ == "__main__":
    main()
