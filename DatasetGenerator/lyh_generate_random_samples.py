import os
os.environ['PYTHONHASHSEED'] = '2'
import sys
sys.path.append("..")
import constants as cnst


import tqdm
from model.stg2_generator import StyledGenerator
import numpy as np
from my_utils.visualize_flame_overlay import OverLayViz
from my_utils.flm_dynamic_fit_overlay import camera_ringnetpp
from my_utils.generic_utils import save_set_of_images
import constants
from dataset_loaders import fast_image_reshape
import torch
from my_utils import generic_utils
from my_utils.eye_centering import position_to_given_location
from copy import deepcopy
from my_utils.photometric_optimization.models import FLAME
from my_utils.photometric_optimization import util


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


# General settings
save_images = True
code_size = 236
use_inst_norm = True
core_tensor_res = 4
resolution = 256
alpha = 1
step_max = int(np.log2(resolution) - 2)
# ?????????????????????????????????????????????????????????batch_size????????????
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
np.random.seed(3)

'''
#!?????????????????????????????????????????????
flame_param ??????key??????????????????pkl??????????????????pkl?????????????????????????????????
shape_params ??????????????????100???float??????????????????0??????????????????
exp_params ??????
pose_params ???????????????????????????????????????,??????????????????head pose y-axis???????????????jaw pose x-pose?????????????????????head???xyz???????????????jaw???xyz
lit ????????????????????????????????????????????????????????????????????????????????????????????????[N, 9(shcoeff), 3(rgb)]
texture ???????????????????????????????????????????????????????????????????????????
cam ?????????MVP??????????????????????????????????????????????????????????????????????????????

????????????????????? util.py?????????
??????albedo??? gif_helper.py?????????
'''

for i, key in enumerate(fl_param_dict):
    # flame_param = fl_param_dict['00000.pkl']
    flame_param = fl_param_dict[key]
    # shape_params = np.concatenate((np.random.normal(0, 1, [10,]), np.zeros(90))).astype('float32')
    # exp_params = np.concatenate((np.random.normal(0, 1, [10,]), np.zeros(40))).astype('float32')
    shape_params = np.zeros(100).astype('float32')
    exp_params = np.ones(50).astype('float32')
    # +- pi/4 for bad samples +- pi/8 for good samples
    # pose = np.array([0, np.random.uniform(-np.pi/4, np.pi/4, 1), 0,
    #                  np.random.uniform(0, np.pi/12, 1), 0, 0]).astype('float32')
    # pose = np.array([0, np.random.uniform(-np.pi / 8, np.pi / 8, 1), 0,
    #                  np.random.uniform(0, np.pi / 12, 1), 0, 0]).astype('float32')
    pose = np.array([0, 0, 0,
                     0, 0, 0]).astype('float32')
    # texture = np.zeros(50).astype('float32')
    # texture = np.random.normal(0, 1, [50]).astype('float32')
    # texture = flame_param['tex']
    texture = fl_param_dict['00001.pkl']['tex']
    
    # lit = fl_param_dict['00001.pkl']['lit']
    # lit = np.ones(27)
    
    lit = np.array([[3.742953, 3.577128, 3.869045],
           [0.02612345, 0.051215664, 0.057771258],
           [0.22151725, 0.20861958, 0.22362661],
           [-0.297732, -0.6012796, -0.8950591],
           [0.059372187, 0.08085689, 0.07886178],
           [-0.10977558, -0.19952367, -0.25157148],
           [-0.2340106, -0.18055701, -0.22514306],
           [0.010122169, -1.0088086e-05, -0.05537848],
           [0.51583904, 0.90824336, 1.2818625]])
      
    
    # lit = np.array([[2, 2, 2],
    #        [0.02612345, 0.051215664, 0.057771258],
    #        [0.22151725, 0.20861958, 0.22362661],
    #        [-0.297732, -0.6012796, -0.8950591],
    #        [0.059372187, 0.08085689, 0.07886178],
    #        [-0.10977558, -0.19952367, -0.25157148],
    #        [-0.2340106, -0.18055701, -0.22514306],
    #        [0.010122169, -1.0088086e-05, -0.05537848],
    #        [0.51583904, 0.90824336, 1.2818625]])
    # flame_param = np.hstack((shape_params, exp_params, pose, flame_param['cam'],
    #                          texture, flame_param['lit'].flatten()))
    flame_param = np.hstack((shape_params, exp_params, pose, flame_param['cam'],
                             texture, lit.flatten()))
    # tz = camera_params['f'][0] / (camera_params['c'][0] * flame_param[:, 156:157])
    # flame_param[:, 156:159] = np.concatenate((flame_param[:, 157:], tz), axis=1)

    # import ipdb; ipdb.set_trace()
    flm_params[i, :] = flame_param.astype('float32')
    if i == num_smpl_to_eval_on - 1:
        break

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
                    ??????????????????????????????????????????
                    ????????????????????????
                    generate_random_samples_lyh.py ---- overlay_visualizer.get_rendered_mesh
                    visualize_flame_overlay.py     ---- ???OverLayViz???get_rendered_mesh??????????????????????????????????????????
                                                        ??????????????????????????????????????????????????????
                    gif_helper.py                  ---- ???render_utils???render_tex_and_normal
                                                        ???????????????albedo??????FLAME???????????????????????????????????????????????????????????????
                    renderer.py                    ---- ???Renderer??????forward??????????????????????????????????????????????????????????????????
                                                        ???????????????????????????????????????????????????????????????...?????????????????????
                                                        ????????????dic?????????????????????????????????????????????
                '''
                
                norma_map_img, _, _, _, rend_flm, albedo_img, pos_mask, shading_img = \
                    overlay_visualizer.get_rendered_mesh(flame_params=(shape, exp, pose, light_code, texture_code),
                                                         camera_params=cam)
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
                ????????????????????????????????????????????????????????????????????????????????????????????????albedo?????????????????????
                ??????????????????????????????????????????constant_albedo???????????????,??????????????????0.6???????????????
                ??????0-255??????????????????120?????????
                '''
                norma_map_img_to_save, _, _, _, rend_flm_to_save, albedo_img_to_save, pos_mask_to_save, shading_img_to_save = \
                    overlay_visualizer.get_rendered_mesh(flame_params=(shape, exp, pose, light_code, texture_code),
                                                         camera_params=cam, cull_backfaces=False, constant_albedo=120)
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
            save_dir = '/home/lyh/results/GIF_ShadingTest'
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, 'params.npy'), params_to_save)

            save_path_current_id = os.path.join(save_dir, 'images')
            save_set_of_images(path=save_path_current_id, prefix='', images=(images + 1) / 2, show_prog_bar=False)

            #save flam rndr
            save_path_current_id_flm_rndr = os.path.join(save_dir, 'conditions')
            save_set_of_images(path=save_path_current_id_flm_rndr, prefix='mesh', images=(flame_mesh_imgs + 1) / 2,
                               show_prog_bar=True)
            
            
            save_set_of_images(path=save_path_current_id_flm_rndr, prefix='norm_',
                               images=(flame_normal_map_imgs + 1) / 2, show_prog_bar=False)
            
            
            save_set_of_images(path=save_path_current_id_flm_rndr, prefix='rndr_',
                               images=(flame_rend_imgs + 1) / 2, show_prog_bar=False)


            save_set_of_images(path=save_path_current_id_flm_rndr, prefix='albedo_',
                               images=(flame_albedo_imgs + 1) / 2, show_prog_bar=False)
            
            
            save_set_of_images(path=save_path_current_id_flm_rndr, prefix='mask_',
                               images=(flame_pos_mask + 1) / 2, show_prog_bar=False)

            save_set_of_images(path=save_path_current_id_flm_rndr, prefix='shading_',
                               images=(flame_shading_img + 1) / 2, show_prog_bar=False)