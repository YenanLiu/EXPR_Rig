import pickle
import random
 

file_path = "/project/zhangwei/xusheng/rig2img/rig2img_bernice_20240919_p2.pkl"

with open(file_path, 'rb') as f:
    data = pickle.load(f)

num_items_to_select = int(len(data) * 0.3)
selected_items = dict(random.sample(data.items(), num_items_to_select))

remaining_items = {k: v for k, v in data.items() if k not in selected_items}

selected_file_path = "/project/zhangwei/xusheng/rig2img/rig2img_bernice_20240919_val_p2.pkl"
with open(selected_file_path, 'wb') as f:
    pickle.dump(selected_items, f)

remaining_file_path = "/project/zhangwei/xusheng/rig2img/rig2img_bernice_20240919_train_p2.pkl"
with open(remaining_file_path, 'wb') as f:
    pickle.dump(remaining_items, f)

print(f"30% of the key-value pairs have been saved to {selected_file_path} len {len(selected_items)}")
print(f"The remaining key-value pairs have been saved to {remaining_file_path} len {len(remaining_items)}")

# 65075
# 151844

# pkl_path = "/project/zhangwei/xusheng/rig2img/rig2img_bernice_20240919_p2.pkl"
# data_dir = "/project/zhangwei/xusheng/img2rig_data/"

# new_dict = {}
# not_exists_path={}
# save_path = "/project/zhangwei/xusheng/rig2img/rig2img_bernice_20240919_p2_clean.pkl"
# save_path2 = "/project/zhangwei/xusheng/rig2img/rig2img_lackingdata.pkl"

# with open(pkl_path, 'rb') as f:
#     data = pickle.load(f)
#     print("ok") # 21, 6919

#     for name, rigs in data.items():
#         img_path = os.path.join(data_dir, name)
#         if os.path.exists(save_path):
#             new_dict[name] = rigs
#         else:
#             not_exists_path[name] = rigs
#             print(img_path)

# print(len(not_exists_path))

# with open(save_path, 'wb') as f:  # Open the file in binary write mode
#     pickle.dump(new_dict, f)

# with open(save_path2, 'wb') as f:  # Open the file in binary write mode
#     pickle.dump(not_exists_path, f)