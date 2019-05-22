import time, datetime, argparse, cv2

from models.model_train_singan import Model_Train
from utils.utils import *
from models.ops import *
#tf.config.gpu.set_per_process_memory_fraction(0.6)
tf.config.gpu.set_per_process_memory_growth(True)

"""===========================================================================
                                configuaration
==========================================================================="""
start = time.time()
time_now = datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default=4)  # -1 for CPU
parser.add_argument("--target_size", type=list, default=250, nargs="+", help = 'Image size after crop.')
parser.add_argument("--batch_size", type=int, default=1, help= 'Minibatch size(global)')
parser.add_argument("--data_root_test", type=str, default= './datasets/test', help= 'Data root dir')
parser.add_argument("--image_file", type=str, default= './datasets/test/176039.jpg', help= 'Data root dir')
parser.add_argument("--channels", type=int, default= 3, help= 'Channel size')
parser.add_argument("--color_map", type=str, default= "RGB", help= 'Channel mode. [RGB, YCbCr]')
parser.add_argument("--model_tag", type=str, default= "default", help= 'Exp name to save logs/checkpoints.')
parser.add_argument("--checkpoint_dir", type=str, default='../__outputs/checkpoints/', help=  'Dir for checkpoints.')
parser.add_argument("--summary_dir", type=str, default='../__outputs/summaries/', help= 'Dir for tensorboard logs.')
parser.add_argument("--restore_file", type=str, default=None, help=  'file for resotration')
parser.add_argument("--graph_mode", type= bool, default=False, help=  'use graph mode for training')
config = parser.parse_args()


def generate_expname_automatically():
    name = "SinGAN_%s_%02d_%02d_%02d_%02d_%02d" % (config.model_tag,
            time_now.month, time_now.day, time_now.hour,
            time_now.minute, time_now.second)
    return name
expname  = generate_expname_automatically()
config.checkpoint_dir += "SinGAN "+ config.model_tag ; check_folder(config.checkpoint_dir)
config.summary_dir += expname ; check_folder(config.summary_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)


"""===========================================================================
                                prepare dataset
==========================================================================="""
""" read  image """
img = cv2.imread(config.image_file)[...,::-1] #conver to rgb

""" resize image """
H, W, _ =  img.shape
H,W = (250,int(W * 250/H)) if H < W else (int(H * 250/W), 250)
img = cv2.resize(img, (H,W))

""" reshape image """
img = np.expand_dims(img,axis=0)
img = normalize(img) ;print(img.shape)



"""===========================================================================
                                build model
==========================================================================="""



""" --------------------------------------------------------------------
train
---------------------------------------------------------------------"""
for i in range(8+1)[::-1]:
    """ build model """
    model = Model_Train(config, target_image=img)
    """ restore model """
    if config.restore_file is not None:
        model.ckpt.restore(os.path.join(config.checkpoint_dir + "ckpt-0"))

    for ii in range(10):
        """ train """
        N= i
        log = model.train_step(N = N, log_interval= 100)
        print("[train {}] step:{} {}".format(N,model.step.numpy(), log))

    """ save """
    #if model.step.numpy() % 1000 == 0:  save_path = model.save()
    save_path = model.save()
    model.step.assign_add(1)
