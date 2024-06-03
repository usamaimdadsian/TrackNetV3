import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from test import predict_location, get_ensemble_weight, generate_inpaint_mask
from dataset import Shuttlecock_Trajectory_Dataset
from utils.general import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VideoFrameSampler:
    def __init__(self, path, sample_seconds) -> None:
        self.path = path
        self.sample_seconds = sample_seconds

        self.cap = cv2.VideoCapture(path)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = 0

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def sampleFramesBySeconds(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
        required_frames = self.fps * self.sample_seconds
        extracted_frames = []

        while (
            self.cap.isOpened()
            and (len(extracted_frames) < required_frames)
            and (self.frame_count < self.total_frames)
        ):
            ret, frame = self.cap.read()
            if ret is not None:
                self.frame_count += 1
                extracted_frames.append(frame)
        return extracted_frames

    def __iter__(self):
        self.frame_count = 0
        return self

    def __next__(self):
        if self.frame_count >= self.total_frames:
            raise StopIteration
        else:
            return self.sampleFramesBySeconds()


def predict(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """Predict coordinates from heatmap or inpainted coordinates.

    Args:
        indices (torch.Tensor): indices of input sequence with shape (N, L, 2)
        y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
        c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
        img_scaler (Tuple): image scaler (w_scaler, h_scaler)

    Returns:
        pred_dict (Dict): dictionary of predicted coordinates
            Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """

    pred_dict = {"Frame": [], "X": [], "Y": [], "Visibility": []}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = (
        indices.detach().cpu().numpy() if torch.is_tensor(indices) else indices.numpy()
    )

    # Transform input for heatmap prediction
    if y_pred is not None:
        y_pred = y_pred > 0.5
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred)  # (N, L, H, W)

    # Transform input for coordinate prediction
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    # Predict from coordinate
                    c_p = c_pred[n][f]
                    cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(
                        c_p[1] * HEIGHT * img_scaler[1]
                    )
                elif y_pred is not None:
                    # Predict from heatmap
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    cx_pred, cy_pred = int(bbox_pred[0] + bbox_pred[2] / 2), int(
                        bbox_pred[1] + bbox_pred[3] / 2
                    )
                    cx_pred, cy_pred = int(cx_pred * img_scaler[0]), int(
                        cy_pred * img_scaler[1]
                    )
                else:
                    raise ValueError("Invalid input")
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                pred_dict["Frame"].append(int(f_i))
                pred_dict["X"].append(cx_pred)
                pred_dict["Y"].append(cy_pred)
                pred_dict["Visibility"].append(vis_pred)
                prev_f_i = f_i
            else:
                break

    return pred_dict


if __name__ == "__main__":
    video_file = "mini.mp4"
    tracknet_file = "ckpts/TrackNet_best.pt"
    inpaintnet_file = "ckpts/InpaintNet_best.pt"
    save_dir = "prediction"
    output_video = True
    video_sample_length = 1  # in seconds
    eval_mode = "weight"
    batch_size = 16

    num_workers = batch_size if batch_size <= 16 else 16
    num_workers = 4

    # DEFINE MODELS AND LOAD WEIGTHS
    tracknet_ckpt = torch.load(tracknet_file, map_location=DEVICE)
    tracknet_seq_len = tracknet_ckpt["param_dict"]["seq_len"]
    bg_mode = tracknet_ckpt["param_dict"]["bg_mode"]
    tracknet = get_model("TrackNet", tracknet_seq_len, bg_mode).to(DEVICE)
    tracknet.load_state_dict(tracknet_ckpt["model"])

    if inpaintnet_file:
        inpaintnet_ckpt = torch.load(inpaintnet_file, map_location=DEVICE)
        inpaintnet_seq_len = inpaintnet_ckpt["param_dict"]["seq_len"]
        inpaintnet = get_model("InpaintNet").to(DEVICE)
        inpaintnet.load_state_dict(inpaintnet_ckpt["model"])
    else:
        inpaintnet = None

    # SAMPLE FRAMES FROM THE VIDEO
    videosampler = VideoFrameSampler(video_file, video_sample_length)
    prediction_dicts = []

    for frame_list in videosampler:
        fps, w, h = videosampler.fps, videosampler.width, videosampler.height

        w_scaler, h_scaler = w / WIDTH, h / HEIGHT
        img_scaler = (w_scaler, h_scaler)

        print(f"Number of sampled frames: {len(frame_list)}")

        tracknet_pred_dict = {
            "Frame": [],
            "X": [],
            "Y": [],
            "Visibility": [],
            "Inpaint_Mask": [],
            "Img_scaler": (w_scaler, h_scaler),
            "Img_shape": (w, h),
        }

        # Test on TrackNet
        tracknet.eval()
        seq_len = tracknet_seq_len

        dataset = Shuttlecock_Trajectory_Dataset(
            seq_len=seq_len,
            sliding_step=1,
            data_mode="heatmap",
            bg_mode=bg_mode,
            frame_arr=np.array(frame_list)[:, :, :, ::-1],
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        weight = get_ensemble_weight(seq_len, eval_mode)

        # Init prediction buffer params
        num_sample, sample_count = len(dataset), 0
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len)  # [0, 1, 2, 3, 4, 5, 6, 7]
        frame_i = torch.arange(seq_len - 1, -1, -1)  # [7, 6, 5, 4, 3, 2, 1, 0]
        y_pred_buffer = torch.zeros(
            (buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32
        )

        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().to(DEVICE)
            b_size, seq_len = i.shape[0], i.shape[1]
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()

            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

            for b in range(b_size):
                if sample_count < buffer_size:
                    # Imcomplete buffer
                    y_pred = y_pred_buffer[batch_i + b, frame_i].sum(0)
                    y_pred /= sample_count + 1
                else:
                    # General case
                    y_pred = (
                        y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]
                    ).sum(0)

                ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                ensemble_y_pred = torch.cat(
                    (ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0
                )
                sample_count += 1

                if sample_count == num_sample:
                    # Last batch
                    y_zero_pad = torch.zeros(
                        (buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32
                    )
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                    for f in range(1, seq_len):
                        # Last input sequence
                        y_pred = y_pred_buffer[batch_i + b + f, frame_i].sum(0)
                        y_pred /= seq_len - f
                        ensemble_i = torch.cat(
                            (ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0
                        )
                        ensemble_y_pred = torch.cat(
                            (ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)),
                            dim=0,
                        )

            # Predict
            tmp_pred = predict(
                ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler
            )
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])

            # Update buffer, keep last predictions for ensemble in next iteration
            y_pred_buffer = y_pred_buffer[-buffer_size:]


        if inpaintnet is not None:
            inpaintnet.eval()
            seq_len = inpaintnet_seq_len
            tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=h*0.05)
            inpaint_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

            # Create dataset with overlap sampling for temporal ensemble
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='coordinate', pred_dict=tracknet_pred_dict)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
            weight = get_ensemble_weight(seq_len, eval_mode)

            # Init buffer params
            num_sample, sample_count = len(dataset), 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
            frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
            coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
            
            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                b_size = i.shape[0]
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.to(DEVICE), inpaint_mask.to(DEVICE)).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask)
                
                # Thresholding
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.

                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)
                
                for b in range(b_size):
                    if sample_count < buffer_size:
                        # Imcomplete buffer
                        coor_inpaint = coor_inpaint_buffer[batch_i+b, frame_i].sum(0)
                        coor_inpaint /= (sample_count+1)
                    else:
                        # General case
                        coor_inpaint = (coor_inpaint_buffer[batch_i+b, frame_i] * weight[:, None]).sum(0)
                    
                    ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                    ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                    sample_count += 1

                    if sample_count == num_sample:
                        # Last input sequence
                        coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                        coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                        
                        for f in range(1, seq_len):
                            coor_inpaint = coor_inpaint_buffer[batch_i+b+f, frame_i].sum(0)
                            coor_inpaint /= (seq_len-f)
                            ensemble_i = torch.cat((ensemble_i, i[-1][f].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)

                # Thresholding
                th_mask = ((ensemble_coor_inpaint[:, :, 0] < COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < COOR_TH))
                ensemble_coor_inpaint[th_mask] = 0.

                # Predict
                tmp_pred = predict(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
                # Update buffer, keep last predictions for ensemble in next iteration
                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]



        pred_dict = inpaint_pred_dict if inpaintnet is not None else tracknet_pred_dict
        prediction_dicts.append(pred_dict)

    print("Predictions Completed")
