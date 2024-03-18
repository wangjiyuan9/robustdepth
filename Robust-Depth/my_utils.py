# 引入模块
from PIL import Image
from PIL import ImageChops

# from tensorboard.backend.event_processing import event_accumulator
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
# from layers import depth_to_disp, disp_to_depth
from tqdm import trange
import cv2
from gpu_chek.gpu_checker import get_gpu_info
from utils import *
import options as g

MIN_DEPTH = g.MIN_DEPTH
MAX_DEPTH = g.MAX_DEPTH
new_width = g.defalut_width
new_height = g.defalut_height
verbose = False  # Enter True to print the other information
save_tmp = False  # Enter True to save the middle result
dataset = 'kitti'  # or kitti cityscape


def ini():
    print("please check the super parameters in my_utils.py")
    print("MIN_DEPTH:", MIN_DEPTH, ";MAX_DEPTH:", MAX_DEPTH, ";new_width:", new_width, ";new_height:", new_height,
        ";verbose:", verbose, ";save_tmp:", save_tmp)
    print("dataset:", dataset)


def crop_image(re_img, nw=new_width, nh=new_height):
    if isinstance(re_img, Image.Image):
        width, height = re_img.size
    else:
        height, width = re_img.shape[:2]
    left = (width - nw) / 2
    top = (height - nh) / 2
    right = (width + nw) / 2
    bottom = (height + nh) / 2
    crop_im = re_img.crop((left, top, right, bottom)) if isinstance(re_img, Image.Image) else re_img[
                                                                                              int(top):int(bottom),
                                                                                              int(left):int(right)]
    return crop_im


def crop(basepath, outpath):
    basepath = input('please input the image path:') if basepath is None else basepath
    outpath = input('please input the output path:') if outpath is None else outpath
    os.makedirs(outpath, exist_ok=True)
    dir_list = os.listdir(basepath)
    print("length of dir_list:", len(dir_list))
    for i in trange(len(dir_list)):
        if os.path.exists(os.path.join(outpath, dir_list[i])):
            if verbose:
                print("file exist")
            continue
        path = os.path.join(basepath, dir_list[i])
        if verbose:
            print("path:", path)
        img = Image.open(path).convert('RGB')
        crop_im = crop_image(img)
        crop_im.save(os.path.join(outpath, dir_list[i]))


def val(input1=None, input2=None, post_process='resize'):
    path1 = input('please input the image1 path:') if input1 is None else input1
    path2 = input('please input the image2 path:') if input2 is None else input2

    # 转化为rgb，alpha通道默认为255
    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')
    if post_process == 'resize':
        img1 = img1.resize((new_width, new_height))
        img2 = img2.resize((new_width, new_height))
    elif post_process == 'crop':
        img1 = crop_image(img1)
        img2 = crop_image(img2)
    if save_tmp:
        img1.save('val/my.png')
        img2.save('val/off.png')
    # 两图相减观察区别
    diffence = ImageChops.difference(img1, img2)
    diffence.save('val/diffence.png')


# train main
# def VisulizeDepth(path=None, depth=None, out_name='depth', process='crop', writer=None, rcd=None, debug=0, nwidth=new_width, nheight=new_height):
#     '''用于可视化，在程序调用模板为：
#     if Visualization:
#         # out and check the result
#         val_depth(writer,None, pred_depth, 'val_pred_depth', process='None',epoch=epoch)
#         raise NotImplementedError
#     可给writer,之后第一个写none，第二个写pred_depth或者gt_depth，表示可视化的图像，第三个是输出的图像名字，不带任何路径和后缀，
#     自动保存在val文件夹下，第四个是处理方式，resize或者crop或者none,最后是给了writer后，还可以给rcd，表示记录的位置，比如epoch
#     '''
#     path = input('please input the depth path:') if path is None and depth is None else path
#     if depth is None:
#         depth = np.load(path) if path.endswith('npy') else np.array(Image.open(path))
#     if debug >= 2: print("output process:", process, ";depth shape:", depth.shape, ";name:", out_name, ";clip depth:", MIN_DEPTH, "~", MAX_DEPTH)
#     if depth.ndim == 3:
#         depth = depth.squeeze(0)
#     if dataset == 'cityscape':
#         height, width = depth.shape
#         p_depth = cv2.resize(depth, (int(width / 2), int(height / 2)))
#         p_depth = p_depth[:int(0.75 * height / 2), :]
#     elif process == 'resize':
#         p_depth = np.array(Image.fromarray(depth).resize((nwidth, nheight)))
#     elif process == 'crop':
#         p_depth = crop_image(depth, nwidth, nheight)
#     elif process == 'clip':
#         p_depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH, out=depth)
#     else:
#         p_depth = depth
#     org_disp = depth_to_disp(p_depth, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
#     disp = np.where(org_disp < 0.99, org_disp, 0)
#     if debug >= 2:
#         print("out shape:", disp.shape)
#         print("min depth:", np.min(p_depth[p_depth > 0]), ";max depth:", np.max(p_depth))
#     if writer is not None:
#         vmin = np.percentile(disp[disp < 0.99], 1)
#         vmax = np.percentile(disp[disp < 0.99], 99)
#         color_map = plt.get_cmap('inferno')
#         if debug >= 3: print("ok with plt!")
#         disp_color = color_map((disp - vmin) / (vmax - vmin))
#         disp_color = (disp_color * 255).astype(np.uint8)
#         disp_color = np.transpose(disp_color, (2, 0, 1))
#         if "gt" in out_name:
#             writer.add_image(out_name, disp_color, global_step=0)
#         else:
#             writer.add_image(out_name, disp_color, global_step=rcd)
#     else:
#         plt.imsave('./figures/predictions/{}'.format(out_name + ".png"), disp, cmap="inferno", vmin=np.percentile(disp[disp < 0.99], 1),
#             vmax=np.percentile(disp[disp < 0.99], 99))


def val_gt_depth(path, id, save=True, process='crop'):
    gt_path = input('please input the gt_depth path:') if path is None else path
    id = input('please input the id:') if id is None else id
    print("begin to load gt_depths...")
    gt_depths = np.load(gt_path, allow_pickle=True)
    print(gt_depths.shape)
    # raise NotImplementedError
    # gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    print("down!!!")
    if dataset == 'cityscape':
        gt_depth = gt_depths  # cityscape的gt_depths是一个list，只有一个元素
    elif dataset == 'kitti':  # kitti的gt_depths是一个dict，有多个元素
        gt_depth = gt_depths[id]
    # 输出最大深度和最小深度
    print("max_depth:", np.max(gt_depth[gt_depth < 80]))
    print("min_depth:", np.min(gt_depth[gt_depth > 0]))
    if save:
        VisulizeDepth(None, gt_depth, out_name='gt_test1', process=process)


def merge_folders(folder1, folder2):
    '''merge files in a folder to 000000.png, 000001.png, etc. must begin with 000000...'''
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))

    for i, file in enumerate(files1):
        os.rename(os.path.join(folder1, file), os.path.join(folder1, f"{str(i).zfill(6)}.{file.split('.')[1]}"))
    last_index = i

    for file in files2:
        index = int(file.split(".")[0]) + last_index + 1
        os.rename(os.path.join(folder2, file), os.path.join(folder1, f"{str(index).zfill(6)}.{file.split('.')[1]}"))


def merge_folder_recur(folder1, folder2):  # 目前是假递归，只有一层
    folders1 = os.listdir(folder1)
    folders2 = os.listdir(folder2)
    for f1, f2 in zip(folders1, folders2):
        merge_folders(os.path.join(folder1, f1), os.path.join(folder2, f2))


def depth_read(filename):
    # loads depth map from png file and returns it as a numpy array,

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert (np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    print("max depth:", np.max(depth), "min depth:", np.min(depth[depth > 0]))
    return depth


def depth_change(depth=None):
    print("注意此时是错位的，是debug用的")
    depth_path = '/data/cylin/wjy/dataset/KITTI/train/gt_depth/gt_depths.npz' if depth is None else None
    gt_depth = depth if depth is not None else np.load(depth_path, fix_imports=True, encoding='latin1')["data"]
    rate = [0, 0.01]
    val_gt_depth = gt_depth[int(rate[0] * len(gt_depth)):int(rate[1] * len(gt_depth))]
    np.save("/data/cylin/wjy/dataset/KITTI/train/gt_depth/debug_gt_train.npy", val_gt_depth)
    rate = [0.01, 0.015]
    test_gt_depth = gt_depth[int(rate[0] * len(gt_depth)):int(rate[1] * len(gt_depth))]
    np.save("/data/cylin/wjy/dataset/KITTI/train/gt_depth/debug_gt_val.npy", test_gt_depth)
    print("down!!!")


def vis_dif():
    coeffs_zip = zip(g.coeffs, g.coeffs)
    out3_images = []
    for coeff1, coeff2 in coeffs_zip:
        out1_image = Image.open("val/out1_" + coeff1 + ".jpg")
        out2_image = Image.open("val/out2_" + coeff2 + ".jpg")
        out3_image = ImageChops.subtract(out1_image, out2_image)
        out3_images.append(out3_image)
    for i, coeff in enumerate(g.coeffs):
        out3_image = out3_images[i]
        out3_image.save("val/out3_" + coeff + ".jpg")


def tensorboard_reader(org_path=None, epoch=0):
    path = os.path.join(g.default_log_dir, org_path, 'val')
    val_path = os.path.join(path, os.listdir(path)[0])
    # ea = event_accumulator.EventAccumulator(path, size_guidance={'scalars': 0})
    ea = None
    ea.Reload()
    ld_mode_map = g.out_load_map
    weight_mode = org_path.split("/")[0]
    weight_mode_map = g.out_name_map

    wt_mode, hav = '', False
    for key in weight_mode_map:
        if key in weight_mode:
            wt_mode += weight_mode_map[key]
            hav = True
    if not hav:
        wt_mode = '晴'
    epoch_data = {}
    for key in ea.scalars.Keys():
        events = ea.Scalars(key)
        for event in events:
            # print(event.step)
            if event.step == epoch:
                epoch_data[key] = event.value
                break
    out_text = ""
    for ld_mode in ld_mode_map:
        title = wt_mode + "\\rightarrow " + ld_mode_map[ld_mode] + str(epoch)
        out_text += title
        fmt = '&{: 8.3e}' if ld_mode == 'variance' else '&{: 8.3f}'
        out_text += (fmt * 7).format(
            epoch_data['{}/abs_rel'.format(ld_mode)],
            epoch_data['{}/sq_rel'.format(ld_mode)],
            epoch_data['{}/rmse'.format(ld_mode)],
            epoch_data['{}/rmse_log'.format(ld_mode)],
            epoch_data['{}/a1'.format(ld_mode)],
            epoch_data['{}/a2'.format(ld_mode)],
            epoch_data['{}/a3'.format(ld_mode)]
        ) + "\\\\" + "\n"
    print(out_text)
    print("-\\\\")


def tensor_out():
    list = g.weights_path.split(" ")
    for i in list:
        path = i.split("/")[0]
        epoch = i.split("_")[-1]
        tensorboard_reader(path, int(epoch))


def move_folder(basepath=None, path=None, mode=None):
    import shutil
    if mode is None:
        raise ValueError("mode is None")
    if path is None:
        raise ValueError("path is None")
    # 复制文件夹
    src = os.path.join(basepath.split('visulize')[0], path, mode)
    dst = os.path.join(basepath, path)
    shutil.copytree(src, dst)


def easy_visulize(basepath=None, mode=None):
    basepath = modify_opt(path=basepath)
    list = os.listdir(basepath)
    list = [i for i in list if i != 'visulize' and i != 'offical']
    basepath = os.path.join(basepath, 'visulize')
    remove_logfolder(basepath, True)
    os.mkdir(basepath)
    for path in list:
        move_folder(basepath, path, mode)


def meshgrid(x, y, indexing):
    grid_y, grid_x = torch.meshgrid(y, x)
    return (grid_x, grid_y)


# train main
def modify_opt(opt=None, path=None):
    """用于检测GPU类型，修改opt参数,也可以用于修改路径"""
    gpu_info = get_gpu_info()
    cuda_version = gpu_info[2].split('/')[-1].strip()  # 获取GPU名称
    if '10.1' in cuda_version:
        new_root = '/data/cylin/'
        org_root = '/opt/data/private/'
        if opt is not None:
            opt.log_dir = opt.log_dir.replace(org_root, new_root)
            opt.data_path = opt.data_path.replace(org_root, new_root)
            opt.data_path = opt.data_path.replace('backup/weather_datasets', 'dataset')
            return opt
        else:
            path = path.replace(org_root, new_root)
            path = path.replace('backup/weather_datasets', 'dataset')
            return path
    if opt is not None:
        return opt
    else:
        return path


# train main
def remove_logfolder(log_path, overwrite=False):
    """用于删除重名的log文件夹"""
    import shutil
    if os.path.exists(log_path) and overwrite:
        shutil.rmtree(log_path)
        print("Has Removed old log files at:  ", log_path)


class DataProvider():

    def __init__(self, dataset, batch_size, shuffle=True, num_workers=16, drop_last=True, pin_memory=True, worker_init_fn=None):
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_iter = None
        self.iter = 0
        self.epoch = 0
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.worker_init_fn = worker_init_fn

    def build(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            worker_init_fn=self.worker_init_fn)
        self.data_iter = _MultiProcessingDataLoaderIter(dataloader) if self.num_workers > 0 else iter(dataloader)

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iter += 1
            return batch

        except StopIteration:
            self.epoch += 1
            self.build()
            self.iter = 1
            batch = self.data_iter.next()
            return batch


def out_split(path, weather):
    path = os.path.join("/opt/data/private/wjy/backup/weather_datasets/drivingstereo", "foggy", "left-image-full-size")
    result = os.listdir(path)
    for file in result:
        print(file.split(".")[0].split("2018-10-25-07-37-26_2018-10-25-")[1])


def test_code():
    pass


if __name__ == '__main__':
    ini()

    # depth_change(None)
    # tensor_out()
    easy_visulize(modify_opt(path='/opt/data/private/wjy/PlaneDepth/log_err'), 'val')
    # out_split(None,None)
    # merge_folder_recur('/data/cylin/wjy/dataset/KITTI/test/0032','/data/cylin/wjy/dataset/KITTI/test/0056')
    # merge_folders(1,2)
    # crop('/data/cylin/wjy/dataset/KITTI/train/image_03fog','/data/cylin/wjy/dataset/KITTI/train/prev_3fogcrop')
    # val('C:/Users/Wang/Desktop/weather_datasets/2011_09_26_drive_0032_sync/rain/50mm/rainy_image/0000000000.png','F:/research/test/32_50mm/rainy_image/0000000000.png','crop')
    # val_depth()
    # val_gt_depth('/opt/data/private/wjy/backup/weather_datasets/kitti/gt_depths.npy', 225, save=True, process=None)
    # VisulizeDepth('/opt/data/private/wjy/backup/weather_datasets/cityscape/pred_depths/bonn/bonn_000000_000019_leftImg8bit.npy', process='crop', nwidth=1664, nheight=512, debug=2,
    #     out_name='bonn_000000_000019_leftImg8bit.png')
    # depth_read('/data/cylin/wjy/dataset/kitti/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000011.png')
    # test_code()
    # vis_dif()
