import argparse
import os 
import sys
import numpy as np 
import torch
from torch.nn import functional as F
import torch.optim as optim
import time
import cv2
from utils import *
import torchshow as ts

from model import Model
from dataloader import data_generator
from datetime import datetime
from collections import defaultdict
import random
import itertools
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap
from scene_PGD import vis_trajectory

CUDA_VISIBLE_DEVICES='1'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()

def seed_torch(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch()

max_x = 1. 
max_y = 1. 
history_frames = 4 # 3 second * 2 frame/second
future_frames = 12 # 3 second * 2 frame/second

total_epoch = 100
base_lr = 0.01
lr_decay_epoch = 5
dev = 'cuda:0'
vis = False
use_map = True

input_type = ['vel', 'scene_norm', 'heading', 'map']	
#input_type = ['vel', 'heading', 'map']
work_dir = './trained_models/Expt100_nus_wmap_noda_traj_newheading'
if not os.path.exists(work_dir):
	os.makedirs(work_dir)
writer = SummaryWriter("losslog/Expt100_nus_wmap_noda_traj_newheading")
log_file_path = os.path.join(work_dir,'log.txt')
log_file = open(log_file_path, 'a+')
test_result_file = 'prediction_result.txt'

criterion = torch.nn.SmoothL1Loss()



def logging(epoch, total_epoch, iter, total_iter, ep, seq, frame, losses_str, log, train_flag):
	if train_flag == 1:
		print_log('{} | Epo: {:02d}/{:02d}, '
			'It: {:04d}/{:04d}, '
			'time: {:s}, seq {:s}, frame {:d}, {}'
        	.format('Training Nuscenes', epoch, total_epoch, iter, total_iter, convert_secs2time(ep), seq, frame, losses_str), log)
	else:
		print_log('{} | Epo: {:02d}/{:02d}, '
			'It: {:04d}/{:04d}, '
			'time: {:s}, seq {:s}, frame {:d}, {}'
        	.format('Evaluating Nuscenes', epoch, total_epoch, iter, total_iter, convert_secs2time(ep), seq, frame, losses_str), log)		

def my_save_model(pra_model, pra_epoch):
	path = '{}/model_epoch_{:04}.pt'.format(work_dir, pra_epoch)
	torch.save(
		{
			'xin_graph_seq2seq_model': pra_model.state_dict(),
		}, 
		path)
	print('Successfull saved to {}'.format(path))


def my_load_model(pra_model, pra_path):
	checkpoint = torch.load(pra_path)
	pra_model.load_state_dict(checkpoint['xin_graph_seq2seq_model'])
	print('Successfull loaded from {}'.format(pra_path))
	return pra_model

def load_pretrain_model(pra_model, pra_path, pra_path_gcn):
	model_dict = pra_model.state_dict()
	checkpoint = torch.load(pra_path)
	checkpoint2 = torch.load(pra_path_gcn)

	pretrained_dict = checkpoint['state_dict']
	pretrained_dict =  {k: v for k, v in model_dict.items() if k in pretrained_dict}

	pretrained_dict_gcn = checkpoint2['xin_graph_seq2seq_model']
	pretrained_dict_gcn =  {k: v for k, v in pretrained_dict_gcn.items() if k in model_dict} 

	model_dict.update(pretrained_dict)
	model_dict.update(pretrained_dict_gcn)
	pra_model.load_state_dict(model_dict)
	
	#freeze the gcn weights
	for name, param in pra_model.named_parameters():
		print(name, param.requires_grad)
		param.requires_grad = False
		if name == 'edge_importance.3':
			break

	print('Successfull loaded from {}'.format(pra_path + pra_path_gcn))
	return pra_model

def load_pretrain_gcn(pra_model, pra_path):
	model_dict = pra_model.state_dict()
	checkpoint = torch.load(pra_path)

	pretrained_dict = checkpoint['xin_graph_seq2seq_model']
	pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict} 

	model_dict.update(pretrained_dict)
	pra_model.load_state_dict(model_dict)

	#freeze the gcn weights
	for name, param in pra_model.named_parameters():
		print(name, param.requires_grad)
		param.requires_grad = False
		if name == 'edge_importance.3':
			break

	print('Successfull loaded from {}'.format(pra_path))
	return pra_model

def preprocess_data(in_data, train_flag):

	data = defaultdict(lambda: None)
	data['batch_size'] = len(in_data['pre_motion_3D'])
	data['agent_num'] = len(in_data['pre_motion_3D'])
	data['pre_motion'] = torch.stack(in_data['pre_motion_3D'], dim=0).to(dev).transpose(0, 1).contiguous()
	data['fut_motion'] = torch.stack(in_data['fut_motion_3D'], dim=0).to(dev).transpose(0, 1).contiguous()
	data['fut_motion_orig'] = torch.stack(in_data['fut_motion_3D'], dim=0).to(dev)   # future motion without transpose
	data['fut_mask'] = torch.stack(in_data['fut_motion_mask'], dim=0).to(dev)
	data['pre_mask'] = torch.stack(in_data['pre_motion_mask'], dim=0).to(dev)
	scene_orig_all_past = False
	
	data['scene_orig'] = data['pre_motion'][-1].mean(dim=0)
	if in_data['pre_heading'] is not None:
		#data['heading'] = torch.tensor(in_data['heading']).float().to(dev)
		data['pre_heading'] = torch.stack(in_data['pre_heading'], dim=0).to(dev).transpose(0, 1).contiguous()
		data['fut_heading'] = torch.stack(in_data['fut_heading'], dim=0).to(dev).transpose(0, 1).contiguous()

	# rotate the scene
	if train_flag:
		theta = torch.rand(1).to(dev) * np.pi * 2
		for key in ['pre_motion', 'fut_motion', 'fut_motion_orig']:
			data[f'{key}'], data[f'{key}_scene_norm'] = rotation_2d_torch(data[key], theta, data['scene_orig'])
		if in_data['pre_heading'] is not None:
			#data['heading'] += theta
			data['pre_heading'] += theta
			data['fut_heading'] += theta
	else:
		theta = torch.zeros(1).to(dev)
		for key in ['pre_motion', 'fut_motion', 'fut_motion_orig']:
			data[f'{key}_scene_norm'] = data[key] - data['scene_orig']   # normalize per scene	
		
	data['pre_vel'] = data['pre_motion'][1:] - data['pre_motion'][:-1, :]
	data['fut_vel'] = data['fut_motion'] - torch.cat([data['pre_motion'][[-1]], data['fut_motion'][:-1, :]])
	data['cur_motion'] = data['pre_motion'][[-1]]
	data['pre_motion_norm'] = data['pre_motion'][:-1] - data['cur_motion']   # normalize pos per agent
	data['fut_motion_norm'] = data['fut_motion'] - data['cur_motion']
	if in_data['pre_heading'] is not None:
		#data['heading_vec'] = torch.stack([torch.cos(data['heading']), torch.sin(data['heading'])], dim=-1)
		data['pre_heading_vec'] = torch.cat((torch.cos(data['pre_heading']), torch.sin(data['pre_heading'])), dim = -1)
		data['fut_heading_vec'] = torch.cat((torch.cos(data['fut_heading']), torch.sin(data['fut_heading'])), dim = -1)
    
	# agent maps
	# if use_map:
	# 	scene_map = in_data['scene_map']
	# 	tmp = torch.from_numpy(scene_map.data).to(dev)
	# 	scene_points = np.stack(in_data['pre_motion_3D'])[:, -1]
	# 	patch_size = [50, 10, 50, 90]
	# 	rot = -np.array(in_data['heading'])  * (180 / np.pi)
	# 	data['agent_maps'] = scene_map.get_cropped_maps(scene_points, patch_size, rot).to(dev)
	
	return data
	

def compute_RMSE_loss(data, pra_pred):
	diff = data['fut_motion_orig'][:, :pra_pred.shape[1]].cuda() - pra_pred
	mask = data['fut_mask'][:, :pra_pred.shape[1]].cuda()
	diff *= mask.unsqueeze(2)
	total_loss = diff.pow(2).sum()
	#loss = total_loss / (diff.shape[0] * diff.shape[1])
	return total_loss, diff.shape[0], diff.shape[1]

def compute_ADEFDE(data, pra_pred):
	diff = torch.abs(data['fut_motion_orig'][:, :pra_pred.shape[1]].cuda() - pra_pred)
	mask = data['fut_mask'][:, :pra_pred.shape[1]].cuda()
	diff *= mask.unsqueeze(2)
	diff = torch.pow(diff, exponent=2)
	diff = torch.sum(diff, dim=2)
	diff = torch.pow(diff, exponent=0.5) #Shape: (N, K, T)
	
	sum_FDE = torch.sum(diff[:, -1])
	sum_ADE = torch.sum(diff) / diff.shape[1] #Shape: (N, K)

	return sum_ADE, sum_FDE, diff.shape[0]

def PGD(pra_model, generator, epoch):
	# pra_model.to(dev)
	pra_model.train()
	for p in pra_model.parameters():
		p.requires_grad = False

	# train model using training data
	iter_n = 100
	gamma = 0.8
	eps = 8

	save_dir = "./output/PGD"
	pgd_log_dir = './logs/PGD'
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	if not os.path.exists(pgd_log_dir):
		os.mkdir(pgd_log_dir)
	log_save_path =  os.path.join(pgd_log_dir, 'adv_log_{:}_{:}_{:}_t.txt'.format(iter_n,gamma,eps) )
	map_save_dir = log_save_path
	with open(log_save_path, "w") as fl_out:
        # print(exp_name, file = fl_out)
		pass

	num_frame = 0
	while not generator.is_epoch_end():
		min_loss = 100
		since_train = time.time()
		in_data = generator()
		if in_data is None: continue
		seq, frame = in_data['seq'], in_data['frame']
		num_frame += 1
		# print("num: ", num_frame, seq,frame)
		if seq == 'scene-0017':
			break
		# if num_frame < 1525:
		# 	continue 
		data = preprocess_data(in_data, train_flag = 0)

		scene_map = in_data['scene_map']
		s_map_d = scene_map.data.astype(np.float)
		map_t = torch.from_numpy(s_map_d).to(dev)
		
		agent_maps = map_t.clone().detach()
		
		# m = 1 - map_t / 255
		# m[1] = m[0]
		# m[2] = m[0]

		# agent_maps.requires_grad = True
		for num_i in range(iter_n):
			map_t.requires_grad = True
			scene_points = np.stack(in_data['pre_motion_3D'])[:, -1]
			patch_size = [50, 10, 50, 90]
			rot = -np.array(in_data['heading'])  * (180 / np.pi)
			(data['agent_maps'], _) = scene_map.get_cropped_maps(scene_points, patch_size,map_t, rot)
			data['agent_maps'] = data['agent_maps'].to(dev)
			data['map_enc'] = pra_model.map_encoder(data['agent_maps'])

			predicted = pra_model(data=data, t_num = history_frames, input_type = input_type) # (N, C, T, V)=(N, 2, 6, 120)

			predicted = predicted.squeeze(0).permute(2, 1, 0)
			predicted = torch.cumsum(predicted, dim=2)

			predicted += data['pre_motion'][[-1]].cuda().permute(1, 0, 2)		
			sum_ADE, sum_FDE, num_agent = compute_ADEFDE(data, predicted)

			adv_loss =  - sum_FDE - sum_ADE
			adv_loss.backward()

			perturb_grad = map_t.grad
			adv_los =  adv_loss.detach().cpu().numpy()
			# print("iter num ", num_i, " adv loss: ",adv_los)

			if adv_los < min_loss:
				min_adv_loss = adv_los
				min_ADE = sum_ADE.detach().cpu().numpy()/num_agent
				min_FDE = sum_FDE.detach().cpu().numpy()/num_agent
				map_output = map_t * 1.0
				min_loss = adv_los
				min_n = num_i
				
            
			with torch.no_grad():
				image_perturb = perturb_grad.sign() * gamma
				# image_perturb = image_perturb*m        ######## only attack non drivable area

				map_t -= image_perturb
				# map_t[2] -= image_perturb[2]

				image_perturb = torch.clamp(map_t - agent_maps, -eps, eps)
				map_t = agent_maps + image_perturb
				# data['agent_maps'] = torch.max(
                #     torch.min(data['agent_maps'], agent_maps + eps), agent_maps - eps)
				
				map_t = torch.clamp(map_t, min=0, max=255.0)
		
		ep = time.time() - since_train
		print('seq: {seq:} |frame id: {b_id:} |Tot: {total:} |min_adv_loss: {loss:} |min_iter: {min_n:}|min_ADE: {min_a:}|min_FDE: {min_f:} \n'.format(
                    seq = seq, b_id=frame, total=ep, loss=min_adv_loss, min_n=min_n, min_a=min_ADE, min_f=min_FDE))
		
		with open(log_save_path, "a") as fl_out:
			print('seq: {seq:} |frame id: {b_id:} |Tot: {total:} |min_adv_loss: {loss:} |min_iter: {min_n:}|min_ADE: {min_a:}|min_FDE: {min_f:} \n'.format(
                    seq = seq, b_id=frame, total=ep, loss=min_adv_loss, min_n=min_n, min_a=min_ADE, min_f=min_FDE), file = fl_out)
		map_save_dir = save_map(seq,frame, map_output, iter_n, gamma, eps,save_dir)
	
	return map_save_dir,log_save_path

def val_model(pra_model, generator, epoch, map_dir,log_save_path):
	# pra_model.to(dev)
	pra_model.eval()

	since_train = time.time()
	lossfunc = {
		'ADE': compute_ADEFDE,
		'FDE': compute_ADEFDE
	}
	loss_meter = {x: AverageMeter() for x in lossfunc.keys()}
	
	# with open(log_save_path, "a") as fl_out:
	# 	print("starting val......", file = fl_out)

	# temp_folder = os.path.join(map_dir, "vis_o1")
	# DATAROOT = '/data/yaozhen/Nus_Transgrip/data/nus_official/full_nuscene/'
	# nuscenes = NuScenes('v1.0-trainval', dataroot=DATAROOT)

	# train model using training data
	while not generator.is_epoch_end():
		in_data = generator()
		if in_data is None: continue
		seq, frame = in_data['seq'], in_data['frame']

		# if frame > 4:
		# 	break
		if seq == 'scene-0017':
			break
		data = preprocess_data(in_data, train_flag = 0)

		############  for original val  ############
		# scene_map = in_data['scene_map']
		# s_map_d = scene_map.data.astype(np.float)
		# map_t = torch.from_numpy(s_map_d).to(dev)

		###########  for after attacked val ############
		input_path = os.path.join(map_dir, '{}_{:06d}.png'.format(seq, frame))
		
		# trajectron_seq =int(seq.split('-')[1])
		# input_path = os.path.join(map_dir, '{}_{:06d}.png'.format(trajectron_seq, frame))

		if not os.path.exists(input_path):
			print("no exist map path: ", input_path)
			continue
		map = cv2.imread(input_path).astype('float32')
		kernel = np.ones((3,3), np.uint8)
		map = cv2.morphologyEx(map, cv2.MORPH_OPEN, kernel)
		map = cv2.morphologyEx(map, cv2.MORPH_CLOSE, kernel)
		# map = np.load(input_path)
		map = map.transpose(2, 0, 1)
		map_t = torch.from_numpy(map).to(dev)
		scene_map = in_data['scene_map']

		scene_points = np.stack(in_data['pre_motion_3D'])[:, -1]
		patch_size = [50, 10, 50, 90]
		rot = -np.array(in_data['heading'])  * (180 / np.pi)
		
		(data['agent_maps'], _) = scene_map.get_cropped_maps(scene_points, patch_size,map_t, rot)
		data['agent_maps'] = data['agent_maps'].to(dev)
		data['map_enc'] = pra_model.map_encoder(data['agent_maps'])

		# data['agent_maps'] = torch.round(data['agent_maps'])

		predicted = pra_model(data=data, t_num = history_frames, input_type = input_type) # (N, C, T, V)=(N, 2, 6, 120)

		predicted = predicted.squeeze(0).permute(2, 1, 0)
		predicted = torch.cumsum(predicted, dim=2)

		predicted += data['pre_motion'][[-1]].cuda().permute(1, 0, 2)
						
		########################################################
		# Compute details for training
		########################################################
		sum_ADE, sum_FDE, num_agent = compute_ADEFDE(data, predicted)
		print("seq:", seq, "frame:", frame, "ADE:", sum_ADE.detach().cpu().numpy()/num_agent, "FDE:", sum_FDE.detach().cpu().numpy()/num_agent)
		# with open(log_save_path, "a") as fl_out:
		# 	print("seq:", seq, "frame:", frame, "ADE:", sum_ADE.detach().cpu().numpy()/num_agent, "FDE:", sum_FDE.detach().cpu().numpy()/num_agent, file = fl_out)
		loss_meter['ADE'].update(sum_ADE.item(), num_agent)
		loss_meter['FDE'].update(sum_FDE.item(), num_agent)

		# vis_trajectory(seq, frame, data, temp_folder, in_data, predicted,DATAROOT,nuscenes)

	ep = time.time() - since_train
	losses_str = ' '.join([f'{y.count}{x}: {y.avg:.3f} ' for x, y in loss_meter.items()])
	print('Tot:',ep,' results: ', losses_str)

	# with open(log_save_path, "a") as fl_out:
	# 	print('Tot:',ep,' results: ', losses_str, file = fl_out)

def run_PGD(pra_model):
	generator = data_generator(log_file, split='val', phase='testing')
	map_dir,log_save_path = PGD(pra_model, generator, epoch = 0)
	# generator = data_generator(log_file, split='val', phase='testing')
	# val_model(pra_model, generator, 0, map_dir,log_save_path)

def run_val(pra_model):
	generator = data_generator(log_file, split='val', phase='testing')
	print('#######################################Test')
	# map_dir = "./output/norm-PGD-t/adv_img_100_0.5_255_255"
	# map_dir = "./output/PGD/adv_img_100_0.8_8"
	map_dir = "./output/PSO-p1/adv_img_30_50_1_4"
	# map_dir = "../Trajectron-plus-plus/experiments/nuScenes/output/PGD_mostlikely/adv_img_100_0.8_8"
	log_save_path = "./logs/PGD/adv_log_100_0.8_8.txt"
	val_model(pra_model, generator, 0, map_dir,log_save_path)

def save_map(seq,frame, map_output, iter_n, gamma, eps,save_dir):
	img_dir =  os.path.join(save_dir, 'adv_img_{:}_{:}_{:}'.format(iter_n,gamma,eps) )
	# print("saving exp: ", img_dir)
	if not os.path.exists(img_dir):
		os.mkdir(img_dir)
	# num_maps = map_output.shape[0]
	# for i in range(num_maps):
	img = map_output.detach().cpu().numpy().transpose(1, 2, 0)
	# out_path = os.path.join(img_dir, '{:06d}.npy'.format(frame))
	# np.save(out_path, img)
	out_path = os.path.join(img_dir, '{}_{:06d}.png'.format(seq, frame))
	cv2.imwrite(out_path, img)
	
	return img_dir

if __name__ == '__main__':
	# apollo: 120; kitti_10hz_6: 30; kitti_10hz_30: 40; nuscene: 60; kitti_2hz: 35; 
	model = Model()

	pretrained_model_path = '/data/yaozhen/Nus_Transgrip/Grip_trans/trained_models/Expt123_nus/model_epoch_0047.pt'
	model = my_load_model(model, pretrained_model_path)
	model.to(dev)
	# run_PGD(model)
	run_val(model)

	
