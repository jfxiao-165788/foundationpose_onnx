# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import onnxruntime as ort
from omegaconf import OmegaConf

# ==============================================================================
# 新增：ONNX 模型包装器
# ==============================================================================
class ONNXModelWrapper:
    def __init__(self, onnx_path, model_type='score'):
        print(f"Loading ONNX model from: {onnx_path}")
        # 使用 CUDA 和 CPU 提供程序
        self.session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model_type = model_type
        
    def __call__(self, *args, **kwargs):
        # 模拟 PyTorch 的 forward 调用
        if self.model_type == 'score':
            # ScoreNet forward(A, B, L)
            # args[0]: A, args[1]: B, args[2]: L

            input_A = args[0]
            input_B = args[1]
            
            # 处理 L 参数
            if len(args) > 2:
                L_val = args[2]
            elif 'L' in kwargs:
                L_val = kwargs['L']
            else:
                raise ValueError("Missing argument 'L' for ScoreNet")
            
            input_feed = {
                'input_A': input_A.detach().cpu().numpy(),
                'input_B': input_B.detach().cpu().numpy(),
                'L': np.array(L_val, dtype=np.int64)
            }
            outputs = self.session.run(['score_logit'], input_feed)
            return {'score_logit': torch.from_numpy(outputs[0]).cuda()}
            
        elif self.model_type == 'refine':
            # RefineNet forward(A, B)
            input_feed = {
                'input_A': args[0].detach().cpu().numpy(),
                'input_B': args[1].detach().cpu().numpy()
            }
            outputs = self.session.run(['trans', 'rot'], input_feed)
            return {
                'trans': torch.from_numpy(outputs[0]).cuda(),
                'rot': torch.from_numpy(outputs[1]).cuda()
            }
            
    def eval(self): pass
    def cuda(self): return self
    def load_state_dict(self, *args): pass

class ScorePredictorONNX(ScorePredictor):
    def __init__(self, onnx_path):
        # 1. 复制父类 __init__ 的配置加载逻辑，但去掉加载权重的部分
        self.amp = True
        self.run_name = "2024-01-11-20-02-45"
        code_dir = os.path.dirname(os.path.realpath(__file__))
        # 加载配置
        self.cfg = OmegaConf.load(f'{code_dir}/weights/{self.run_name}/config.yml')
        self.cfg['enable_amp'] = True

        # 设置默认值 (保持与父类一致)
        if 'use_normal' not in self.cfg: self.cfg['use_normal'] = False
        if 'use_BN' not in self.cfg: self.cfg['use_BN'] = False
        if 'zfar' not in self.cfg: self.cfg['zfar'] = np.inf
        if 'c_in' not in self.cfg: self.cfg['c_in'] = 4
        if 'normalize_xyz' not in self.cfg: self.cfg['normalize_xyz'] = False
        if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None: self.cfg['crop_ratio'] = 1.2

        # 初始化数据集 (需要用到)
        self.dataset = ScoreMultiPairH5Dataset(cfg=self.cfg, mode='test', h5_file=None, max_num_key=1)
        
        # 2. 关键点：不创建 PyTorch 模型，直接使用 ONNX Wrapper
        self.model = ONNXModelWrapper(onnx_path, model_type='score')
        print(f"[ScorePredictorONNX] Initialized with ONNX model: {onnx_path}")

class PoseRefinePredictorONNX(PoseRefinePredictor):
    def __init__(self, onnx_path):
        # 1. 复制父类 __init__ 的配置加载逻辑
        self.amp = True
        self.run_name = "2023-10-28-18-33-37"
        code_dir = os.path.dirname(os.path.realpath(__file__))
        self.cfg = OmegaConf.load(f'{code_dir}/weights/{self.run_name}/config.yml')
        self.cfg['enable_amp'] = True

        # 设置默认值
        if 'use_normal' not in self.cfg: self.cfg['use_normal'] = False
        if 'use_mask' not in self.cfg: self.cfg['use_mask'] = False
        if 'use_BN' not in self.cfg: self.cfg['use_BN'] = False
        if 'c_in' not in self.cfg: self.cfg['c_in'] = 4
        if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None: self.cfg['crop_ratio'] = 1.2
        if 'n_view' not in self.cfg: self.cfg['n_view'] = 1
        if 'trans_rep' not in self.cfg: self.cfg['trans_rep'] = 'tracknet'
        if 'rot_rep' not in self.cfg: self.cfg['rot_rep'] = 'axis_angle'
        if 'zfar' not in self.cfg: self.cfg['zfar'] = 3
        if 'normalize_xyz' not in self.cfg: self.cfg['normalize_xyz'] = False
        if isinstance(self.cfg['zfar'], str) and 'inf' in self.cfg['zfar'].lower(): self.cfg['zfar'] = np.inf
        if 'normal_uint8' not in self.cfg: self.cfg['normal_uint8'] = False

        # 初始化数据集
        self.dataset = PoseRefinePairH5Dataset(cfg=self.cfg, h5_file='', mode='test')
        
        # 2. 关键点：不创建 PyTorch 模型，直接使用 ONNX Wrapper
        self.model = ONNXModelWrapper(onnx_path, model_type='refine')
        self.last_trans_update = None
        self.last_rot_update = None
        print(f"[PoseRefinePredictorONNX] Initialized with ONNX model: {onnx_path}")

def compute_pose_error(pose_gt, pose_est):
    """计算平移误差(cm)和旋转误差(deg)"""
    trans_err = np.linalg.norm(pose_gt[:3, 3] - pose_est[:3, 3]) * 100 # m to cm
    R_gt = pose_gt[:3, :3]
    R_est = pose_est[:3, :3]
    trace = np.trace(R_gt.T @ R_est)
    trace = np.clip((trace - 1) / 2, -1.0, 1.0)
    rot_err = np.arccos(trace) * 180 / np.pi # rad to deg
    return trans_err, rot_err

def print_gpu_memory_info(stage_name=""):
    """打印当前GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        print(f"\n{'='*60}")
        print(f"GPU Memory Info - {stage_name}")
        print(f"{'='*60}")
        print(f"Allocated:     {allocated:.3f} GB")
        print(f"Reserved:      {reserved:.3f} GB")
        print(f"Max Allocated: {max_allocated:.3f} GB")
        print(f"{'='*60}\n")
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated
        }
    else:
        print(f"{stage_name}: CUDA not available")
        return None

def reset_peak_memory_stats():
    """重置峰值内存统计"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

if __name__=='__main__':
  code_dir = os.path.dirname(os.path.realpath(__file__))
  mesh_file = f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj' # 默认mesh文件路径
  test_scene_dir = f'{code_dir}/demo_data/mustard0' # 测试场景目录
  est_refine_iter = 5 # 初始估计的优化迭代次数
  track_refine_iter = 2  # 跟踪的优化迭代次数
  debug = 1 # debug等级，0无，1显示，2保存图片，3保存更多中间结果
  # debug_dir = f'{code_dir}/debug' # debug文件夹路径

  debug_dir = f'{code_dir}/debug_onnx_compare'
  SCORE_ONNX_PATH = f'{code_dir}/weights/2024-01-11-20-02-45/score_model.onnx'
  REFINE_ONNX_PATH = f'{code_dir}/weights/2023-10-28-18-33-37/refine_model.onnx'

  set_logging_format()
  set_seed(0)

  # load mesh
  mesh = trimesh.load(mesh_file)
  # print_gpu_memory_info("After loading mesh")

  debug = debug
  debug_dir = debug_dir
  # 清空debug目录并创建可视化和姿态输出子目录
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  # 计算模型的有向边界框，用于可视化
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  # print_gpu_memory_info("After comkputing oriented bounds")
  
  # 初始化评分器、细化器、渲染上下文
  # scorer = ScorePredictor()
  # refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()

  # 初始化主估算器对象
  # est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  # logging.info("estimator initialization done")
  # print_gpu_memory_info("After initializing estimator")

#     # --- 1. 初始化 PyTorch 版本 (Baseline) ---
#   print("Initializing PyTorch Estimator...")
#   scorer_pt = ScorePredictor()
#   refiner_pt = PoseRefinePredictor()
#   est_pt = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer_pt, refiner=refiner_pt, debug_dir=debug_dir, debug=0, glctx=glctx)

  # --- 2. 初始化 ONNX 版本 ---
  print("Initializing ONNX Estimator...")
#   scorer_onnx = ScorePredictor()
#   scorer_onnx.model = ONNXModelWrapper(SCORE_ONNX_PATH, model_type='score') # 替换模型
  
#   refiner_onnx = PoseRefinePredictor()
#   refiner_onnx.model = ONNXModelWrapper(REFINE_ONNX_PATH, model_type='refine') # 替换模型

  scorer_onnx = ScorePredictorONNX(SCORE_ONNX_PATH)
  refiner_onnx = PoseRefinePredictorONNX(REFINE_ONNX_PATH)
  
  est_onnx = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer_onnx, refiner=refiner_onnx, debug_dir=debug_dir, debug=debug, glctx=glctx)
  
  logging.info("estimator initialization done")

  # 初始化数据读取器，读取测试场景数据
  reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)

  print(f"\n{'='*90}")
  print(f"{'Frame':<5} | {'Trans Err (cm)':<15} | {'Rot Err (deg)':<15} | {'Status':<10}")
  print(f"{'='*90}")

  # 遍历所有帧，进行姿态估算和跟踪
  # print_gpu_memory_info("Before processing frames")
  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      # 第一帧：使用mask进行姿态注册（初始估算）
      mask = reader.get_mask(0).astype(bool)
      # pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
    #   # PyTorch 推理
    #   pose_pt = est_pt.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
      # ONNX 推理
      pose_onnx = est_onnx.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)

      if debug>=3:
        # debug等级高时，保存mesh变换结果和点云
        m = mesh.copy()
        m.apply_transform(pose_onnx)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      # 后续帧：使用跟踪器进行姿态跟踪
      # pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=track_refine_iter)
    #   # PyTorch 推理
    #   pose_pt = est_pt.track_one(rgb=color, depth=depth, K=reader.K, iteration=track_refine_iter)
      # ONNX 推理
      pose_onnx = est_onnx.track_one(rgb=color, depth=depth, K=reader.K, iteration=track_refine_iter)

    # --- 计算并打印误差 ---
    # t_err, r_err = compute_pose_error(pose_pt, pose_onnx)
    # status = "OK" if t_err < 0.1 and r_err < 0.1 else "DIFF"
    # print(f"{i:<5} | {t_err:<15.6f} | {r_err:<15.6f} | {status:<10}")

    # 姿态结果保存为txt文件
    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose_onnx.reshape(4,4))

    if debug>=1:
      # debug>=1：绘制3D包围盒和坐标轴，窗口显示
      # center_pose = pose@np.linalg.inv(to_origin)
      center_pose = pose_onnx@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      # debug>=2：保存可视化图片
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
  # print_gpu_memory_info("At the end of processing")
