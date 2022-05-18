import os
from pydoc import classname
os.environ['PYTHONHASHSEED'] = '2'
import sys
sys.path.append("..")
import constants as cnst


import tqdm
from model.stg2_generator import StyledGenerator
import numpy as np
from my_utils.visualize_flame_overlay import OverLayViz
from my_utils.flm_dynamic_fit_overlay import camera_ringnetpp
from my_utils.generic_utils import lyh_save_set_of_images, save_set_of_images
import constants
from dataset_loaders import fast_image_reshape
import torch
from my_utils import generic_utils
from my_utils.eye_centering import position_to_given_location
from copy import deepcopy
from my_utils.photometric_optimization.models import FLAME
from my_utils.photometric_optimization import util
import random
import csv



# how many data you wanna generate?
total_datasets_counts = 5

'''
three states:
change_normal: by switching normal texture channel
change_albedo: by scaling albedo texture
change_lit:    by changing lit paramter
'''
change_state = 'change_normal'

basic_folder = '/home/lyh/DataSet_0407'
GroundTruth_csv_path = '/home/lyh/DataSet_0407/GroundTruth.csv'








def ge_gen_in(flm_params, textured_rndr, norm_map, normal_map_cond, texture_cond):
    if normal_map_cond and texture_cond:
        return torch.cat((textured_rndr, norm_map), dim=1)
    elif normal_map_cond:
        return norm_map
    elif texture_cond:
        return textured_rndr
    else:
        return flm_params


def corrupt_flame_given_sigma(flm_params, corruption_type, sigma, jaw_sigma, pose_sigma):
    # import ipdb; ipdb.set_trace()
    # np.random.seed(2)
    corrupted_flame = deepcopy(flm_params)
    if corruption_type == 'shape' or corruption_type == 'all':
        corrupted_flame[:, :10] = flm_params[:, :10] 
        # corrupted_flame[:, :10] = flm_params[:, :10] + \
        #                            np.clip(np.random.normal(0, sigma, flm_params[:, :10].shape),
        #                                    -3 * sigma, 3 * sigma).astype('float32')
    if corruption_type == 'exp_jaw'or corruption_type == 'all':
        # Expression
        # corrupted_flame[:, 100:110] = flm_params[:, 100:110] + \
        #                               np.clip(np.random.normal(0, sigma, flm_params[:, 100:110].shape),
        #                                       -3 * sigma, 3 * sigma).astype('float32')
        corrupted_flame[:, 100:110] = flm_params[:, 100:110] 
        # Jaw pose
        # corrupted_flame[:, 153] = flm_params[:, 153] + \
        #                           np.random.normal(0, jaw_sigma, corrupted_flame.shape[0])
                                  
        corrupted_flame[:, 153] = flm_params[:, 153]

    if corruption_type == 'pose' or corruption_type == 'all':
        # pose_perturbation = np.random.normal(0, pose_sigma[i], (corrupted_flame.shape[0], 3))
        # corrupted_flame[:, 150:153] += np.clip(pose_perturbation, -3 * pose_sigma[i], 3 * pose_sigma[i])
        pose_perturbation = np.random.normal(0, pose_sigma, (corrupted_flame.shape[0],))
        corrupted_flame[:, 151] = flm_params[:, 151] 
                                   
        # pose_perturbation = np.random.normal(0, pose_sigma, (corrupted_flame.shape[0],))
        # corrupted_flame[:, 151] = flm_params[:, 151] + \
        #                            np.clip(pose_perturbation, -3 * pose_sigma, 3 * pose_sigma)

    return corrupted_flame


# # General settings
# save_images = True
# code_size = 236
# use_inst_norm = True
# core_tensor_res = 4
# resolution = 256
# alpha = 1
# step_max = int(np.log2(resolution) - 2)
# # 计算的数量，也就是最后出图数量，注意为batch_size的整数倍
# num_smpl_to_eval_on = 1
# use_styled_conv_stylegan2 = True

# flength = 5000
# cam_t = np.array([0., 0., 0])
# camera_params = camera_ringnetpp((512, 512), trans=cam_t, focal=flength)

# # Uncomment the appropriate run_id
# run_ids_1 = [29, ]  # with sqrt(2)
# # run_ids_1 = [7, 24, 8, 3]
# # run_ids_1 = [7, 8, 3]
# # run_ids_1 = [7]

# settings_for_runs = \
#     {24: {'name': 'vector_cond', 'model_idx': '216000_1', 'normal_maps_as_cond': False,
#           'rendered_flame_as_condition': False, 'apply_sqrt2_fac_in_eq_lin': False},
#      29: {'name': 'full_model', 'model_idx': '294000_1', 'normal_maps_as_cond': True,
#           'rendered_flame_as_condition': True, 'apply_sqrt2_fac_in_eq_lin': True},
#      7: {'name': 'flm_rndr_tex_interp', 'model_idx': '051000_1', 'normal_maps_as_cond': False,
#          'rendered_flame_as_condition': True, 'apply_sqrt2_fac_in_eq_lin': False},
#      3: {'name': 'norm_mp_tex_interp', 'model_idx': '203000_1', 'normal_maps_as_cond': True,
#          'rendered_flame_as_condition': False, 'apply_sqrt2_fac_in_eq_lin': False},
#      8: {'name': 'norm_map_rend_flm_no_tex_interp', 'model_idx': '009000_1', 'normal_maps_as_cond': True,
#          'rendered_flame_as_condition': True, 'apply_sqrt2_fac_in_eq_lin': False},}


# overlay_visualizer = OverLayViz()
# # overlay_visualizer.setup_renderer(mesh_file=None)

# flm_params = np.zeros((num_smpl_to_eval_on, code_size)).astype('float32')
# fl_param_dict = np.load(cnst.all_flame_params_file, allow_pickle=True).item()
# np.random.seed(3)

# '''
# #!这个地方是用来修改生成的参数的
# flame_param 根据key来读取对应的pkl文件，每一个pkl文件中包含了生成的参数
# shape_params 模型参数，由100个float来控制，全为0则为默认参数
# exp_params 表情
# pose_params 姿势，能看到嘴和角度的变化,第二个参数是head pose y-axis，第四个是jaw pose x-pose，猜测前三个是head的xyz，后三个是jaw的xyz
# lit 光照，保持一致的话全使用同一个文件就行，使用的是球谐函数来表示的[N, 9(shcoeff), 3(rgb)]
# texture 改变了模型的贴图（换了的话看起来就不是同一个人了）
# cam 相机，MVP变换的时候使用了，不过跑数据的时候感觉这个参数都没变

# 交换法线通道： util.py里修改
# 修改albedo： gif_helper.py里修改
# '''

# # for i, key in enumerate(fl_param_dict):
# key = 1
# flame_param = fl_param_dict['00000.pkl']
# # flame_param = fl_param_dict[key]
# # shape_params = np.concatenate((np.random.normal(0, 1, [10,]), np.zeros(90))).astype('float32')
# # exp_params = np.concatenate((np.random.normal(0, 1, [10,]), np.zeros(40))).astype('float32')
# shape_params = np.zeros(100).astype('float32')
# exp_params = np.ones(50).astype('float32')
# # +- pi/4 for bad samples +- pi/8 for good samples
# # pose = np.array([0, np.random.uniform(-np.pi/4, np.pi/4, 1), 0,
# #                  np.random.uniform(0, np.pi/12, 1), 0, 0]).astype('float32')
# # pose = np.array([0, np.random.uniform(-np.pi / 8, np.pi / 8, 1), 0,
# #                  np.random.uniform(0, np.pi / 12, 1), 0, 0]).astype('float32')
# pose = np.array([0, 0, 0,
#                     0, 0, 0]).astype('float32')
# # texture = np.zeros(50).astype('float32')
# # texture = np.random.normal(0, 1, [50]).astype('float32')
# # texture = flame_param['tex']
# texture = fl_param_dict['00001.pkl']['tex']

# # lit = fl_param_dict['00001.pkl']['lit']
# # lit = np.ones(27)

# lit = np.array([[3.742953, 3.577128, 3.869045],
#         [0.02612345, 0.051215664, 0.057771258],
#         [0.22151725, 0.20861958, 0.22362661],
#         [-0.297732, -0.6012796, -0.8950591],
#         [0.059372187, 0.08085689, 0.07886178],
#         [-0.10977558, -0.19952367, -0.25157148],
#         [-0.2340106, -0.18055701, -0.22514306],
#         [0.010122169, -1.0088086e-05, -0.05537848],
#         [0.51583904, 0.90824336, 1.2818625]])
    

# # lit = np.array([[2, 2, 2],
# #        [0.02612345, 0.051215664, 0.057771258],
# #        [0.22151725, 0.20861958, 0.22362661],
# #        [-0.297732, -0.6012796, -0.8950591],
# #        [0.059372187, 0.08085689, 0.07886178],
# #        [-0.10977558, -0.19952367, -0.25157148],
# #        [-0.2340106, -0.18055701, -0.22514306],
# #        [0.010122169, -1.0088086e-05, -0.05537848],
# #        [0.51583904, 0.90824336, 1.2818625]])
# # flame_param = np.hstack((shape_params, exp_params, pose, flame_param['cam'],
# #                          texture, flame_param['lit'].flatten()))
# flame_param = np.hstack((shape_params, exp_params, pose, flame_param['cam'],
#                             texture, lit.flatten()))
# # tz = camera_params['f'][0] / (camera_params['c'][0] * flame_param[:, 156:157])
# # flame_param[:, 156:159] = np.concatenate((flame_param[:, 157:], tz), axis=1)

# # import ipdb; ipdb.set_trace()
# flm_params[0, :] = flame_param.astype('float32')
# # if key == num_smpl_to_eval_on - 1:
# #     break


# def param_controller():
    
    
    
def single_pic_generate(flm_params, save_dir = '/home/lyh/results/GIF_DataSetTest', folder_name = 'p0', pic_number = 99, delta_albedo=1, normals_exchange = [0,1,2]):
    #! changed by lyh
    batch_size = 1
    # batch_size = 32

    num_sigmas = 1
    corruption_sigma = np.linspace(0, 1.5, num_sigmas)
    jaw_rot_range = (0, np.pi/8)
    jaw_rot_sigmas = np.linspace(0, (jaw_rot_range[1] - jaw_rot_range[0])/6, num_sigmas)
    pose_range = (-np.pi/3, np.pi/3)
    pose_sigmas = np.linspace(0, (pose_range[1] - pose_range[0])/6, num_sigmas)
    config_obj = util.dict2obj(cnst.flame_config)
    flame_decoder = FLAME.FLAME(config_obj).cuda().eval()

    for run_idx in run_ids_1:
        # import ipdb; ipdb.set_trace()
        generator_1 = torch.nn.DataParallel(
            StyledGenerator(embedding_vocab_size=69158,
                            rendered_flame_ascondition=settings_for_runs[run_idx]['rendered_flame_as_condition'],
                            normal_maps_as_cond=settings_for_runs[run_idx]['normal_maps_as_cond'],
                            apply_sqrt2_fac_in_eq_lin=settings_for_runs[run_idx]['apply_sqrt2_fac_in_eq_lin'],
                            core_tensor_res=core_tensor_res,
                            w_truncation_factor=1.0,
                            n_mlp=8)).cuda()
        model_idx = settings_for_runs[run_idx]['model_idx']
        ckpt1 = torch.load(f'{cnst.output_root}checkpoint/{run_idx}/{model_idx}.model')
        generator_1.load_state_dict(ckpt1['generator_running'])
        generator_1 = generator_1.eval()

        params_to_save = {'cam': [], 'shape': [], 'exp': [], 'pose': [], 'light_code': [], 'texture_code': [],
                        'identity_indices': []}

        for i, sigma in enumerate(corruption_sigma):
            images = np.zeros((num_smpl_to_eval_on, 3, resolution, resolution)).astype('float32')
            flame_mesh_imgs = np.zeros((num_smpl_to_eval_on, 3, resolution, resolution)).astype('float32')
            flame_normal_map_imgs = np.zeros((num_smpl_to_eval_on, 3, resolution, resolution)).astype('float32')
            flame_rend_imgs = np.zeros((num_smpl_to_eval_on, 3, resolution, resolution)).astype('float32')
            
            flame_albedo_imgs = np.zeros((num_smpl_to_eval_on, 3, resolution, resolution)).astype('float32')
            flame_pos_mask = np.zeros((num_smpl_to_eval_on, 3, resolution, resolution)).astype('float32')
            flame_shading_img = np.zeros((num_smpl_to_eval_on, 3, resolution, resolution)).astype('float32')
            
            pbar = tqdm.tqdm(range(0, num_smpl_to_eval_on, batch_size))
            pbar.set_description('Generating_images')
            # print(flm_params[1, :])
            for batch_idx in pbar:
                flm_batch = flm_params[batch_idx:batch_idx+batch_size, :]
                flm_batch = torch.from_numpy(flm_batch).cuda()
                # flm_batch = eye_cntr_reg.substitute_flame_batch_with_regressed_camera(flm_batch)
                flm_batch = position_to_given_location(flame_decoder, flm_batch)

                if settings_for_runs[run_idx]['normal_maps_as_cond'] or \
                        settings_for_runs[run_idx]['rendered_flame_as_condition']:

                    batch_size_true = flm_batch.shape[0]
                    cam = flm_batch[:, constants.DECA_IDX['cam'][0]:constants.DECA_IDX['cam'][1]:]
                    shape = flm_batch[:, constants.INDICES['SHAPE'][0]:constants.INDICES['SHAPE'][1]]
                    exp = flm_batch[:, constants.INDICES['EXP'][0]:constants.INDICES['EXP'][1]]
                    pose = flm_batch[:, constants.INDICES['POSE'][0]:constants.INDICES['POSE'][1]]
                    # import ipdb; ipdb.set_trace()
                    light_code = \
                        flm_batch[:, constants.DECA_IDX['lit'][0]:constants.DECA_IDX['lit'][1]:].view((batch_size_true, 9, 3))
                    # light_code = \
                    #         flm_batch[:, 209:209:27].view((batch_size_true,9,3))
                    texture_code = flm_batch[:, constants.DECA_IDX['tex'][0]:constants.DECA_IDX['tex'][1]:]

                    params_to_save['cam'].append(cam.cpu().detach().numpy())
                    params_to_save['shape'].append(shape.cpu().detach().numpy())
                    params_to_save['shape'].append(shape.cpu().detach().numpy())
                    params_to_save['exp'].append(exp.cpu().detach().numpy())
                    params_to_save['pose'].append(pose.cpu().detach().numpy())
                    params_to_save['light_code'].append(light_code.cpu().detach().numpy())
                    params_to_save['texture_code'].append(texture_code.cpu().detach().numpy())
                    
                    
                    '''
                    #! 
                        第一次渲染，按照参数正常进行
                        代码调用层级如下
                        lyh_dataset_generate.py         ---- overlay_visualizer.get_rendered_mesh
                        visualize_flame_overlay.py      ---- 类OverLayViz中get_rendered_mesh将传入参数解耦传递到下一级，
                                                            这里我修改了输出为了能够输出更多图片
                        gif_helper.py                   ---- 类render_utils中render_tex_and_normal
                                                            首先是分析albedo使用FLAME的贴图数据集还是纯色，然后分别调用渲染管线
                        renderer.py                     ---- 类Renderer中的forward进行前向渲染，这里真正的对渲染参数进行了使用
                                                            不过说实话我没看明白他那个类是怎么直接调的...不过肯定是调了
                                                            返回一个dic，可以根据需要自己调对应的图片，补充，是一个神经渲染器
                    '''
                    
                    
                    '''
                    #! 
                        1st render round, control by parameters
                        The code invocation hierarchy is as follows:
                        lyh_dataset_generate.py         ---- overlay_visualizer.get_rendered_mesh
                        visualize_flame_overlay.py      ---- get_rendered_mesh(in Class OverLayViz), Pass the incoming parameter 
                                                             decoupling to the next level, I change the output to generate more pics
                        gif_helper.py                   ---- render_tex_and_normal(in Class render_utils), judge if albedo use FLAME
                                                             Dataset or just solid color, then use rendering pipline
                        renderer.py                     ---- forward(in Class Renderer), It's a nerual rendering pipline, return a dic,
                                                             can custom output by using it.
                    '''
                    
                    norma_map_img, _, _, _, rend_flm, albedo_img, pos_mask, shading_img = \
                        overlay_visualizer.get_rendered_mesh(flame_params=(shape, exp, pose, light_code, texture_code),
                                                            camera_params=cam, normals_exchange = normals_exchange, delta_albedo=delta_albedo)
                    # import ipdb; ipdb.set_trace()

                    rend_flm = torch.clamp(rend_flm, 0, 1) * 2 - 1
                    norma_map_img = torch.clamp(norma_map_img, 0, 1) * 2 - 1
                    rend_flm = fast_image_reshape(rend_flm, height_out=256, width_out=256, mode='bilinear')
                    norma_map_img = fast_image_reshape(norma_map_img, height_out=256, width_out=256, mode='bilinear')

                    albedo_img = torch.clamp(albedo_img, 0, 1)*2 -1
                    albedo_img = fast_image_reshape(albedo_img, height_out=256, width_out=256, mode='bilinear')
                    
                    pos_mask = torch.clamp(pos_mask, 0, 1) * 2 -1
                    pos_mask = fast_image_reshape(pos_mask, height_out=256, width_out=256, mode='bilinear')
                    
                    shading_img = torch.clamp(shading_img, 0, 1) * 2 -1
                    shading_img = fast_image_reshape(shading_img, height_out=256, width_out=256, mode='bilinear')
                    
                    # Render the 2nd time to get backface culling and white texture
                    # norma_map_img_to_save, _, _, _, rend_flm_to_save = \
                    #     overlay_visualizer.get_rendered_mesh(flame_params=(shape, exp, pose, light_code, texture_code),
                    #                                          camera_params=cam, cull_backfaces=True, constant_albedo=0.6)
                    # Back face culling temporarily un-availabe

                    '''#! lyh:
                    这里再进行一次渲染是为了渲染出没有贴图的（纯模型）的结果，所以将albedo设置为了常数，
                    但是作者的渲染管线中并没有对constant_albedo进行归一化,因此原代码中0.6是全黑输出
                    设置0-255即可，我试了120还可以
                    '''
                    '''
                    2nd render round, used to render result without texture
                    so albedo set to ZERO,
                    author didn't normalize constant_albedo, so 0.6 in source code will get a None output
                    need to use int{0-255}, I tried 120, it's good
                    '''
                    norma_map_img_to_save, _, _, _, rend_flm_to_save, albedo_img_to_save, pos_mask_to_save, shading_img_to_save = \
                        overlay_visualizer.get_rendered_mesh(flame_params=(shape, exp, pose, light_code, texture_code),
                                                            camera_params=cam, normals_exchange = normals_exchange, cull_backfaces=False, constant_albedo=120, delta_albedo=1)
                    rend_flm_to_save = torch.clamp(rend_flm_to_save, 0, 1) * 2 - 1
                    # rend_flm_to_save = rend_flm
                    # norma_map_img_to_save = torch.clamp(norma_map_img, 0, 1) * 2 - 1
                    rend_flm_to_save = fast_image_reshape(rend_flm_to_save, height_out=256, width_out=256, mode='bilinear')
                    # norma_map_img_to_save = fast_image_reshape(norma_map_img, height_out=256, width_out=256, mode='bilinear')

                    
                else:
                    rend_flm = None
                    norma_map_img = None
                    

                gen_1_in = ge_gen_in(flm_batch, rend_flm, norma_map_img, settings_for_runs[run_idx]['normal_maps_as_cond'],
                                    settings_for_runs[run_idx]['rendered_flame_as_condition'])

                # torch.manual_seed(2)
                identity_embeddings = torch.randint(low=0, high=69158, size=(gen_1_in.shape[0], ), dtype=torch.long,
                                                    device='cuda')
                mdl_1_gen_images = generic_utils.get_images_from_flame_params(
                    flame_params=gen_1_in.cpu().numpy(), pose=None,
                    model=generator_1,
                    step=step_max, alpha=alpha,
                    input_indices=identity_embeddings.cpu().numpy())

                params_to_save['identity_indices'].append(identity_embeddings.cpu().detach().numpy())
                # import ipdb; ipdb.set_trace()
                images[batch_idx:batch_idx+batch_size_true] = torch.clamp(mdl_1_gen_images, -1, 1).cpu().numpy()
                # if flame_mesh_imgs is None:
                flame_mesh_imgs[batch_idx:batch_idx+batch_size_true] = torch.clamp(rend_flm_to_save, -1, 1).cpu().numpy()
                flame_normal_map_imgs[batch_idx:batch_idx+batch_size_true] = torch.clamp(norma_map_img, -1, 1).cpu().numpy()
                flame_rend_imgs[batch_idx:batch_idx+batch_size_true] = torch.clamp(rend_flm, -1, 1).cpu().numpy()
                
                flame_albedo_imgs[batch_idx:batch_idx+batch_size_true] = torch.clamp(albedo_img, -1, 1).cpu().numpy()
                flame_pos_mask[batch_idx:batch_idx+batch_size_true] = torch.clamp(pos_mask, -1, 1).cpu().numpy()
                flame_shading_img[batch_idx:batch_idx+batch_size_true] = torch.clamp(shading_img, -1,1).cpu().numpy()

            if save_images:
                mdl_name = settings_for_runs[run_idx]['name']
                for key in params_to_save.keys():
                    params_to_save[key] = np.concatenate(params_to_save[key], axis=0)
                # ! Changed by lyh
                # save_dir = os.path.join(cnst.output_root, 'sample', str(run_idx), f'random_samples_q_eval_{mdl_name}')
                
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, 'params.npy'), params_to_save)
                
                
                
                
                csv_path = save_dir + '/'+folder_name + '.csv'
                csv_data = []                
                csv_label = ['cam', 'shape', 'exp', 'pose', 'light_code', 'texture_code', 'identity_indices']
                for key in csv_label:
                    csv_data.append(params_to_save[key])
                if not os.path.exists(csv_path):
                    print("Create New CSV File!")
                    with open(csv_path, "w") as csvfile:
                        file = csv.writer(csvfile)
                        file.writerow(csv_label)
                        file.writerow(csv_data)
                else:
                    print("Open Exist CSV File!")
                    with open(csv_path, "a") as csvfile:
                        file = csv.writer(csvfile)

                        file.writerow(csv_data)
                
                
                stylegan_images_path = os.path.join(save_dir, 'stylegan_images')
                lyh_save_set_of_images(path = os.path.join(stylegan_images_path, folder_name), prefix='', images = (images+1)/2, pic_number = pic_number)

                mesh_path = os.path.join(save_dir, 'mesh')
                lyh_save_set_of_images(path = os.path.join(mesh_path, folder_name), prefix='', images = (flame_mesh_imgs+1)/2, pic_number = pic_number)
                
                norm_path = os.path.join(save_dir, 'norm')
                lyh_save_set_of_images(path = os.path.join(norm_path, folder_name), prefix='', images = (flame_normal_map_imgs+1)/2, pic_number = pic_number)
                
                images_path = os.path.join(save_dir, 'images')
                lyh_save_set_of_images(path = os.path.join(images_path, folder_name), prefix='', images = (flame_rend_imgs+1)/2, pic_number = pic_number)

                albedo_path = os.path.join(save_dir, 'albedo')
                lyh_save_set_of_images(path = os.path.join(albedo_path, folder_name), prefix='', images = (flame_albedo_imgs+1)/2, pic_number = pic_number)
                
                mask_path = os.path.join(save_dir, 'mask')
                lyh_save_set_of_images(path = os.path.join(mask_path, folder_name), prefix='', images = (flame_pos_mask+1)/2, pic_number = pic_number)
                

                
                shading_path = os.path.join(save_dir, 'shading')
                lyh_save_set_of_images(path = os.path.join(shading_path, folder_name), prefix='', images = (flame_shading_img+1)/2, pic_number = pic_number)
                    
    

# General settings
save_images = True
code_size = 236
use_inst_norm = True
core_tensor_res = 4
resolution = 256
alpha = 1
step_max = int(np.log2(resolution) - 2)
# 计算的数量，也就是最后出图数量，注意为batch_size的整数倍, 不需要修改了
num_smpl_to_eval_on = 1

use_styled_conv_stylegan2 = True

flength = 5000
cam_t = np.array([0., 0., 0])
camera_params = camera_ringnetpp((512, 512), trans=cam_t, focal=flength)

# Uncomment the appropriate run_id
run_ids_1 = [29, ]  # with sqrt(2)
# run_ids_1 = [7, 24, 8, 3]
# run_ids_1 = [7, 8, 3]
# run_ids_1 = [7]

settings_for_runs = \
    {24: {'name': 'vector_cond', 'model_idx': '216000_1', 'normal_maps_as_cond': False,
        'rendered_flame_as_condition': False, 'apply_sqrt2_fac_in_eq_lin': False},
    29: {'name': 'full_model', 'model_idx': '294000_1', 'normal_maps_as_cond': True,
        'rendered_flame_as_condition': True, 'apply_sqrt2_fac_in_eq_lin': True},
    7: {'name': 'flm_rndr_tex_interp', 'model_idx': '051000_1', 'normal_maps_as_cond': False,
        'rendered_flame_as_condition': True, 'apply_sqrt2_fac_in_eq_lin': False},
    3: {'name': 'norm_mp_tex_interp', 'model_idx': '203000_1', 'normal_maps_as_cond': True,
        'rendered_flame_as_condition': False, 'apply_sqrt2_fac_in_eq_lin': False},
    8: {'name': 'norm_map_rend_flm_no_tex_interp', 'model_idx': '009000_1', 'normal_maps_as_cond': True,
        'rendered_flame_as_condition': True, 'apply_sqrt2_fac_in_eq_lin': False},}


overlay_visualizer = OverLayViz()
# overlay_visualizer.setup_renderer(mesh_file=None)

flm_params = np.zeros((num_smpl_to_eval_on, code_size)).astype('float32')
fl_param_dict = np.load(cnst.all_flame_params_file, allow_pickle=True).item()
# print(fl_param_dict['69999.pkl'])

np.random.seed(3)




cam = fl_param_dict['00000.pkl']['cam']
for i in range(total_datasets_counts):
    class_name_list = ['o', 'p0', 'p1']
    # shape_params = np.zeros(100).astype('float32')
    # exp_params = np.ones(50).astype('float32')
    shape_params = fl_param_dict[random.choice(list(fl_param_dict))]['shape']
    exp_params = fl_param_dict[random.choice(list(fl_param_dict))]['exp']   
    
    
    texture = fl_param_dict[random.choice(list(fl_param_dict))]['tex']
    # pose = np.array([0, 0, 0, np.random.uniform(0, np.pi/12, 1), np.random.uniform(0, np.pi/12, 1), np.random.uniform(0, np.pi/12, 1)]).astype('float32')
    pose = np.array([0, 0, 0, 0,0,0]).astype('float32')
    lit = np.zeros((3,9,3))
    
    GroundTruth = np.random.randint(0,2)
    delta_albedo = np.ones(3)
    normals_exchange = [[0,1,2], [0,1,2], [0,1,2]]
    if change_state =='change_lit':
    
        lit[0] = fl_param_dict[random.choice(list(fl_param_dict))]['lit']
        delta_lit = lit[0]*0.4
        if GroundTruth==0:
            lit[1] = random.uniform(lit[0]-delta_lit, lit[0]+delta_lit)
            lit[2] = fl_param_dict[random.choice(list(fl_param_dict))]['lit']
            # print(lit[1])
        else:
            lit[1] = fl_param_dict[random.choice(list(fl_param_dict))]['lit']
            lit[2] = random.uniform(lit[0]-delta_lit, lit[0]+delta_lit)
            
    elif change_state =='change_albedo':
        if GroundTruth==0:
            delta_albedo[1] = 0.7+ 0.1*random.random()
            delta_albedo[2] = 0.5+ 0.1*random.random()
        else:
            delta_albedo[1] = 0.5+ 0.1*random.random()
            delta_albedo[2] = 0.7+ 0.1*random.random()
        lit_all = fl_param_dict[random.choice(list(fl_param_dict))]['lit']
        lit[:] = lit_all
        # pass
    
    elif change_state =='change_normal':
        GT_normals_status = [[0,2,1], [2,1,0], [1,0,2]]
        FK_normals_status = [[1,2,0], [2,0,1]]
        if GroundTruth==0:
            normals_exchange[1] = random.choice(GT_normals_status)
            normals_exchange[2] = random.choice(FK_normals_status)
        else:
            normals_exchange[1] = random.choice(FK_normals_status)
            normals_exchange[2] = random.choice(GT_normals_status)
        lit_all = fl_param_dict[random.choice(list(fl_param_dict))]['lit']
        lit[:] = lit_all
        
    for j in range(3):
        class_name = class_name_list[j]
        pose[0] = np.random.uniform(0, np.pi / 48, 1)
        pose[1] = np.random.uniform(0, np.pi / 48, 1)
        pose[2] = np.random.uniform(0, np.pi / 48, 1)
        
        # lit_random_code = np.random.randint(0,306)
        # lit_random_data = str(lit_random_code).zfill(5)+'.pkl'
        # lit_random_data = random.choice(list(fl_param_dict))
        # lit[j] = fl_param_dict[lit_random_data]['lit']
        
        
        
        
        flame_param = np.hstack((shape_params, exp_params, pose, cam, texture, lit[j].flatten()))
        flm_params[0, :] = flame_param.astype('float32')
        single_pic_generate(save_dir = basic_folder,folder_name=class_name, pic_number=i, flm_params=flm_params, delta_albedo=delta_albedo[j], normals_exchange=normals_exchange[j])
        
    # # 计算方法1， 使用L2距离
    # lit_dis0 = np.sqrt(np.sum((lit[0]-lit[1])**2))
    # lit_dis1 = np.sqrt(np.sum((lit[0]-lit[2])**2))
    
    # 计算方法2，每一个球谐函数三个通道计算L2距离，然后求均值, 目测准确率比第一个好
    # lit_dis0_mat = np.zeros(9)
    # lit_dis1_mat = np.zeros(9)
    # for i in range(9):
    #     lit_dis0_mat[i] = np.sqrt(np.sum((lit[0][i]-lit[1][i])**2))
    #     lit_dis1_mat[i] = np.sqrt(np.sum((lit[0][i]-lit[2][i])**2))
    # lit_dis0 = np.mean(lit_dis0_mat)
    # lit_dis1 = np.mean(lit_dis1_mat)
    
    # 计算方法3， 在2的基础上增加了球谐照明的权重
    lit_dis0_mat = np.zeros(9)
    lit_dis1_mat = np.zeros(9)
    for i in range(9):
        lit_dis0_mat[i] = np.sqrt(np.sum((lit[0][i]-lit[1][i])**2))
        lit_dis1_mat[i] = np.sqrt(np.sum((lit[0][i]-lit[2][i])**2))
    lit_wight = np.array((1, 1/4., 1/4., 1/4., 1/9., 1/9., 1/9., 1/9., 1/9.))
    lit_dis0 = np.mean((lit_dis0_mat*lit_wight))
    lit_dis1 = np.mean((lit_dis1_mat*lit_wight))
    
    # GroundTruth = 0 if lit_dis0<lit_dis1 else 1
    

    if not os.path.exists(GroundTruth_csv_path):
        print("Create New CSV File!")
        with open(GroundTruth_csv_path, "w") as csvfile:
            file = csv.writer(csvfile)
            file.writerow([GroundTruth])
    else:
        print("Open Exist CSV File!")
        with open(GroundTruth_csv_path, "a") as csvfile:
            file = csv.writer(csvfile)
            file.writerow([GroundTruth])
# return class_name, shape_params, exp_params, pose, cam,texture, lit



# single_pic_generate(flm_params, save_dir = '/home/lyh/results/GIF_DataSetTest1', class_name = 'p2', pic_number = 999)
# param_controller()