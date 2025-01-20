import os
import torch
import pandas as pd

import imageio
import numpy as np

from cotracker.datasets.utils import CoTrackerData

from cotracker.datasets.kubric_movif_dataset import CoTrackerDataset

class SynusDataset(CoTrackerDataset):
    def __init__(
        self,
        data_root,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        use_augs=False,
        random_seq_len=False,
        random_frame_rate=False,
        random_number_traj=False,
        split="train",
    ):
        super(SynusDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            use_augs=use_augs,
        )
        self.random_seq_len = random_seq_len
        self.random_frame_rate = random_frame_rate
        self.random_number_traj = random_number_traj
        self.pad_bounds = [0, 25]
        self.resize_lim = [0.75, 1.25]  # sample resizes from here
        self.resize_delta = 0.05
        self.max_crop_offset = 15
        self.split = split

        self.info = pd.read_csv(os.path.join(data_root, "info.csv"))
        self.info = self.info[self.info["split"] == self.split]

        self.seq_paths = self.info["file"].values
        if self.split == "valid":
            assert use_augs == False

        print(f'Found {len(self.seq_paths)} sequences in {self.split} split')

    def getitem_helper(self, index):
        gotit = True
        seq_path = self.seq_paths[index]

        rgbs = np.load(os.path.join(self.data_root, seq_path, 'video.npy')) # (num_frames, H, W)
        # expand to 3 channels
        rgbs = np.stack([rgbs] * 3, axis=3) # (num_frames, H, W, 3)
        traj_2d = np.load(os.path.join(self.data_root, seq_path, 'points.npy')) # (num_frames, num_points, 2)
        visibility = np.load(os.path.join(self.data_root, seq_path, 'visibility.npy')) # (num_frames, num_points)
        mask = np.load(os.path.join(self.data_root, seq_path, 'mask.npy')) # (H, W)


        frame_rate = 1
        final_num_traj = self.traj_per_sample
        crop_size = self.crop_size

        # random crop
        min_num_traj = 1
        assert self.traj_per_sample >= min_num_traj
        if self.random_seq_len and self.random_number_traj:
            final_num_traj = np.random.randint(min_num_traj, self.traj_per_sample)
            alpha = final_num_traj / float(self.traj_per_sample)
            seq_len = int(alpha * 10 + (1 - alpha) * self.seq_len)
            seq_len = np.random.randint(seq_len - 2, seq_len + 2)
            if self.random_frame_rate:
                frame_rate = np.random.randint(1, int((40 / seq_len)) + 1)
        elif self.random_number_traj:
            final_num_traj = np.random.randint(min_num_traj, self.traj_per_sample)
            alpha = final_num_traj / float(self.traj_per_sample)
            seq_len = 8 * int(alpha * 2 + (1 - alpha) * self.seq_len // 8)
            # seq_len = np.random.randint(seq_len , seq_len + 2)
            if self.random_frame_rate:
                frame_rate = np.random.randint(1, int((40 / seq_len)) + 1)
        elif self.random_seq_len:
            seq_len = np.random.randint(int(self.seq_len / 2), self.seq_len)
            if self.random_frame_rate:
                frame_rate = np.random.randint(1, int((40 / seq_len)) + 1)
        else:
            seq_len = self.seq_len
            if self.random_frame_rate:
                frame_rate = np.random.randint(1, int((40 / seq_len)) + 1)

        no_augs = False
        if seq_len < len(rgbs):
            if seq_len * frame_rate < len(rgbs):
                start_ind = np.random.choice(len(rgbs) - (seq_len * frame_rate), 1)[0]
            else:
                start_ind = 0
            rgbs = rgbs[start_ind : start_ind + seq_len * frame_rate : frame_rate]
            traj_2d = traj_2d[start_ind : start_ind + seq_len * frame_rate : frame_rate]
            visibility = visibility[
                start_ind : start_ind + seq_len * frame_rate : frame_rate
            ]

        assert seq_len <= len(rgbs)

        if not no_augs:
            if self.use_augs:
                rgbs, traj_2d, visibility = self.add_photometric_augs(
                    rgbs, traj_2d, visibility, replace=False
                )
                rgbs, traj_2d = self.add_spatial_augs(
                    rgbs, traj_2d, visibility, crop_size
                )
            else:
                rgbs, traj_2d = self.crop(rgbs, traj_2d, crop_size)

        visibility[traj_2d[:, :, 0] > crop_size[1] - 1] = False
        visibility[traj_2d[:, :, 0] < 0] = False
        visibility[traj_2d[:, :, 1] > crop_size[0] - 1] = False
        visibility[traj_2d[:, :, 1] < 0] = False

        visibility = torch.from_numpy(visibility)
        traj_2d = torch.from_numpy(traj_2d)

        crop_tensor = torch.tensor(crop_size).flip(0)[None, None] / 2.0
        close_pts_inds = torch.all(
            torch.linalg.vector_norm(traj_2d[..., :2] - crop_tensor, dim=-1) < 1000.0,
            dim=0,
        )
        traj_2d = traj_2d[:, close_pts_inds]
        visibility = visibility[:, close_pts_inds]

        inds_sampled = torch.randperm(traj_2d.shape[1])[: self.traj_per_sample]

        # visibile_pts_first_frame_inds = (visibility[0]).nonzero(as_tuple=False)[:, 0]

        # visibile_pts_mid_frame_inds = (visibility[seq_len // 2]).nonzero(
        #     as_tuple=False
        # )[:, 0]
        # visibile_pts_inds = torch.cat(
        #     (visibile_pts_first_frame_inds, visibile_pts_mid_frame_inds), dim=0
        # )
        # if self.sample_vis_last_frame:
        #     visibile_pts_last_frame_inds = (visibility[seq_len - 1]).nonzero(
        #         as_tuple=False
        #     )[:, 0]
        #     visibile_pts_inds = torch.cat(
        #         (visibile_pts_inds, visibile_pts_last_frame_inds), dim=0
        #     )
        # point_inds = torch.randperm(len(visibile_pts_inds))[: self.traj_per_sample]
        # if len(point_inds) < self.traj_per_sample:
        #     gotit = False

        # visible_inds_sampled = visibile_pts_inds[point_inds]

        trajs = traj_2d[:, inds_sampled].float()
        visibles = visibility[:, inds_sampled]
        valids = torch.ones_like(visibles)

        trajs = trajs[:, :final_num_traj]
        visibles = visibles[:, :final_num_traj]
        valids = valids[:, :final_num_traj]

        rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2).float()

        sample = CoTrackerData(
            video=rgbs,
            trajectory=trajs,
            visibility=visibles,
            valid=valids,
            seq_name=seq_path,
        )
        return sample, gotit

    def __len__(self):
        return len(self.seq_paths)