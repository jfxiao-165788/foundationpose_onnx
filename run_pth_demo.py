# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse

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
  debug_dir = f'{code_dir}/debug' # debug文件夹路径

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
  
  # 初始化评分器、细化器、渲染上下文
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  # 初始化主估算器对象
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  # 初始化数据读取器，读取测试场景数据
  reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)

  print(f"\n{'='*90}")
  print(f"{'Frame':<5} | {'Trans Err (cm)':<15} | {'Rot Err (deg)':<15} | {'Status':<10}")
  print(f"{'='*90}")
  
  results = [] # 用于存储每一帧的测试结果
  # 遍历所有帧，进行姿态估算和跟踪
  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    # 统计数据容器
    frame_stats = {'frame': i}
    if i==0:
      # 第一帧：使用mask进行姿态注册（初始估算）
      mask = reader.get_mask(0).astype(bool)
      torch.cuda.reset_peak_memory_stats() # 重置显存统计
      torch.cuda.synchronize()
      start_t = time.time()
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)

      torch.cuda.synchronize()
      end_t = time.time()
      frame_stats['pth_time_sec'] = end_t - start_t
      frame_stats['pth_mem_MB'] = torch.cuda.max_memory_allocated() / (1024**2)
      
      if debug>=3:
        # debug等级高时，保存mesh变换结果和点云
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      # 后续帧：使用跟踪器进行姿态跟踪
      torch.cuda.reset_peak_memory_stats()
      torch.cuda.synchronize()
      start_t = time.time()
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=track_refine_iter)
      torch.cuda.synchronize()
      end_t = time.time()
      frame_stats['pth_time_sec'] = end_t - start_t
      frame_stats['pth_mem_MB'] = torch.cuda.max_memory_allocated() / (1024**2)
    
    # 姿态结果保存为txt文件
    results.append(frame_stats)
    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      # debug>=1：绘制3D包围盒和坐标轴，窗口显示
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      # debug>=2：保存可视化图片
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

  # --- 循环结束后，保存 Excel ---
  print(f"\n正在保存对比结果...")
  df = pd.DataFrame(results)
  
  # 计算平均值并添加到底部
  mean_row = df.mean(numeric_only=True)
  mean_row['frame'] = 'Average'
  df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
  
  excel_path = f'{debug_dir}/compare_results.xlsx'
  df.to_excel(excel_path, index=False)
  print(f"✅ 成功保存 Excel 文件: {excel_path}")
  print(f"平均 PyTorch 时间: {mean_row['pth_time_sec']:.4f}s")
  # print(f"平均 ONNX 时间:    {mean_row['onnx_time_sec']:.4f}s")
#   print(f"平均平移误差:      {mean_row['trans_err_cm']:.6f} cm")
#   print(f"平均旋转误差:      {mean_row['rot_err_deg']:.6f} deg")
