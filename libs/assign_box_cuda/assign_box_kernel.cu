// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>
#include <vector>
#include <iostream>
#include <algorithm>



__global__ void assign_box_kernel(const long *label_cls, const float *label_box, 
									const float *locs, float *output, 
									const int im_h, const int im_w, const int ph, const int pw, 
									const int tlbr_max_min, const int tlbr_max_max, const int r,
									const int n_max)
{
	int i;
	int b_i = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int phXpw_i = by*1024 + tx;
	if (phXpw_i >= (ph*pw)) return;
	int ph_i = phXpw_i / pw;
	int pw_i = phXpw_i % pw;
	int hs = (im_h-1) / (ph-1);
	int ws = (im_w-1) / (pw-1);
	int out_base = b_i*ph*pw*5 + ph_i*pw*5 + pw_i*5;

	// to cyxyx
	__shared__ float cyxyx[200][5];
	__shared__ float shared_locs[4];
	int b_i_X_4 = b_i*4;
	int b_i_X_n_max = b_i*n_max;
	int b_i_X_n_max_X_4 = b_i_X_n_max*4;
	if(tx==0) {
		for (i=0; i<n_max; i++) {
			cyxyx[i][0] = (float)label_cls[b_i_X_n_max + i];
			cyxyx[i][1] = label_box[b_i_X_n_max_X_4 + i*4 + 0];//ymin
			cyxyx[i][2] = label_box[b_i_X_n_max_X_4 + i*4 + 1];//xmin
			cyxyx[i][3] = label_box[b_i_X_n_max_X_4 + i*4 + 2];//ymax
			cyxyx[i][4] = label_box[b_i_X_n_max_X_4 + i*4 + 3];//xmax
		}
		shared_locs[0] = locs[b_i_X_4 + 0];
		shared_locs[1] = locs[b_i_X_4 + 1];
		shared_locs[2] = locs[b_i_X_4 + 2];
		shared_locs[3] = locs[b_i_X_4 + 3];
	}
	__syncthreads();

	float center_y = ph_i * hs;
	float center_x = pw_i * ws;
	
	if ((center_y<shared_locs[0]) || (center_y>shared_locs[2]) ||
		(center_x<shared_locs[1]) || (center_x>shared_locs[3])) {
		output[out_base + 0] = -1;
		return;
	}
	
	float center_offset_max_2 = r*r;
	float ymin, xmin, ymax, xmax, top, left, bottom, right;
	float cy, cx, dist2, max_tlbr, bxa;
	float cls, pred_ymin, pred_xmin, pred_ymax, pred_xmax;
	float pred_c=-10, pred_area=99999999;

	for (i=0; i<n_max; i++) {
		
		cls  = cyxyx[i][0];
		ymin = cyxyx[i][1];
		xmin = cyxyx[i][2];
		ymax = cyxyx[i][3];
		xmax = cyxyx[i][4];

		top = center_y - ymin;
		bottom = ymax - center_y;
		left = center_x - xmin;
		right = xmax - center_x;
		
		cy = (ymin + ymax) / 2.0;
		cx = (xmin + xmax) / 2.0;
		bxa = (ymax - ymin)*(xmax - xmin);

		dist2 = (center_y - cy) * (center_y - cy) + (center_x - cx) * (center_x - cx);
		max_tlbr = max(top, max(left, max(bottom, right)));
		
		if ((cls>0) && (top>0) && (bottom>0) && (left>0) && (right>0) &&
				(dist2<center_offset_max_2) && 
				(max_tlbr>tlbr_max_min) && (max_tlbr<tlbr_max_max) &&
				(bxa<=pred_area)) {
			pred_area = bxa;
			pred_c = cls;
			pred_ymin = ymin;
			pred_xmin = xmin;
			pred_ymax = ymax;
			pred_xmax = xmax;
		}
	}
	
	if (pred_c > -1) {
		output[out_base + 0] = pred_c;
		output[out_base + 1] = pred_ymin;
		output[out_base + 2] = pred_xmin;
		output[out_base + 3] = pred_ymax;
		output[out_base + 4] = pred_xmax;
	}
}



at::Tensor assign_box_cuda(const at::Tensor &label_cls, const at::Tensor &label_box, 
						const at::Tensor &locs,
						const int im_h, const int im_w, const int ph, const int pw, 
						const int tlbr_max_min, const int tlbr_max_max, const int r)
{
	/*
	GPU >= 6.1

	Param:
	label_cls:  L(b, n_max)         0:bg  1~:fg, 0pad
	label_box:  F(b, n_max, 4)      ymin, xmin, ymax, xmax, 0:pad
	locs:       F(b, 4)             ymin, xmin, ymax, xmax

	im_h = 1025
	im_w = 1025
	ph = 129
	pw = 129

	tlbr_max_min = 5
	tlbr_max_max = 65
	r = 12

	Return:
	target_cls:  L(b, ph, pw)       -1:ign  0:bg  1~:fg
	target_box:  F(b, ph, pw, 4)    ymin, xmin, ymax, xmax

	-> F(b, ph, pw, 1 + 4)

	Note:
	n_max <= 200
	*/
	const int b = label_cls.size(0);
	const int n_max = label_cls.size(1);
	auto output = at::zeros({b, ph, pw, 1 + 4}, label_box.options());
	if (output.numel() == 0) {
		THCudaCheck(cudaGetLastError());
		return output;
	}
	dim3 grid(b, ph*pw/1024+1), block(1024);
	assign_box_kernel<<<grid, block>>>(
		label_cls.contiguous().data<long>(),
		label_box.contiguous().data<float>(),
		locs.contiguous().data<float>(),
		output.contiguous().data<float>(),
		im_h, im_w, ph, pw, 
		tlbr_max_min, tlbr_max_max, r, n_max);
	THCudaCheck(cudaGetLastError());
	return output;
}



__global__ void smooth_kernel(const float *target_cls, float *output, 
								const int ph, const int pw)
{
	int i, j;
	int b_i = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int phXpw_i = by*1024 + tx;
	if (phXpw_i >= (ph*pw)) return;
	int ph_i = phXpw_i / pw;
	int pw_i = phXpw_i % pw;
	int base = b_i*ph*pw + ph_i*pw + pw_i;
	int b_base = b_i*ph*pw;
	int ptr;
	float val = target_cls[base];
	if (val == 0) {
		for(i=ph_i-1; i<=ph_i+1; i++) {
			for(j=pw_i-1; j<=pw_i+1; j++) {
				ptr = b_base + i*pw + j;
				if ((i>=0) && (j>=0) && (i<ph) && (j<pw)) {
					if(target_cls[ptr] > 0) val = -1;
				}
			} 
		}
	}
	output[base] = val;
}



at::Tensor smooth_cuda(const at::Tensor &target_cls)
{
	// target_cls:  F(b, ph, pw)       -1:ign  0:bg  1~:fg
	const int b = target_cls.size(0);
	const int ph = target_cls.size(1);
	const int pw = target_cls.size(2);
	auto output = at::zeros({b, ph, pw}, target_cls.options());
	if (output.numel() == 0) {
		THCudaCheck(cudaGetLastError());
		return output;
	}
	dim3 grid(b, ph*pw/1024+1), block(1024);
	smooth_kernel<<<grid, block>>>(
		target_cls.contiguous().data<float>(),
		output.contiguous().data<float>(),
		ph, pw);
	THCudaCheck(cudaGetLastError());
	return output;
}
