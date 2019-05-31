import time, datetime, argparse, cv2

from models.model_inference import Model_Inference
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
parser.add_argument("--num_scale", type=int, default=8, help= 'num_scale')
parser.add_argument("--image_file", type=str, default= './datasets/test/93_editing.png', help= 'Data root dir')
parser.add_argument("--channels", type=int, default= 3, help= 'Channel size')
parser.add_argument("--color_map", type=str, default= "RGB", help= 'Channel mode. [RGB, YCbCr]')
parser.add_argument("--model_tag", type=str, default= "default", help= 'Exp name to save logs/checkpoints.')
parser.add_argument("--checkpoint_dir", type=str, default='../__outputs/checkpoints/', help=  'Dir for checkpoints.')
parser.add_argument("--graph_mode", type= bool, default=False, help=  'use graph mode for training')
config = parser.parse_args()


config.checkpoint_dir += "SinGAN_" + config.model_tag ; check_folder(config.checkpoint_dir)
config.checkpoint_dir = "./pretrained/singan_fg_sky"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

"""===========================================================================
                                prepare dataset
==========================================================================="""
""" read  image """
config.image_file = './datasets/test/93.png'
#config.image_file = './datasets/test/93_paint.png'
#config.image_file = './datasets/test/93_harmonization.png'
#config.image_file = './datasets/test/93_editing.png'
#config.image_file = './datasets/test/93.png'

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
model = Model_Inference(config, target_image=img)
model.restore()



"""===========================================================================
                               inference
==========================================================================="""
'''
output, input_ = model.inference_sr()
cv2.imshow('image', np.concatenate([output, input_], axis=1)[..., ::-1])
cv2.waitKey(0)
'''
while True :
    #output, input_ = model.inference_paint_to_image()
    #output, input_ = model.inference_harmonization()
    #output, input_ = model.inference_editing()
    output, input_ = model.inference(N=0, start_N=8)
    cv2.imshow('image',np.concatenate([output,input_],axis=1)[...,::-1])
    cv2.waitKey(10)
