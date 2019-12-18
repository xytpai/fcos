#include <torch/extension.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")



at::Tensor assign_box_cuda(const at::Tensor &label_cls, const at::Tensor &label_box, 
						const at::Tensor &locs,
						const int im_h, const int im_w, const int ph, const int pw, 
						const int tlbr_max_min, const int tlbr_max_max, const int r);
at::Tensor assign_box(const at::Tensor &label_cls, const at::Tensor &label_box, 
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
	CHECK_CUDA(label_cls);
	CHECK_CUDA(label_box);
	CHECK_CUDA(locs);
	return assign_box_cuda(label_cls, label_box, locs, im_h, im_w, ph, pw, 
							tlbr_max_min, tlbr_max_max, r);
}



at::Tensor smooth_cuda(const at::Tensor &target_cls);
at::Tensor smooth(const at::Tensor &target_cls)
{
	// target_cls:  F(b, ph, pw)       -1:ign  0:bg  1~:fg
	CHECK_CUDA(target_cls);
	return smooth_cuda(target_cls);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
	m.def("assign_box", &assign_box, "assign_box (CUDA)");
	m.def("smooth", &smooth, "smooth (CUDA)");
}
