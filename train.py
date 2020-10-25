from dataloader import pascalVOC_parser
from pathlib import Path
from experiment import config
from optparse import OptionParser
from dataloader.data_generator import KerasDataGenerator, simple_genetator
import numpy as np

this_dir = Path.cwd()

parser = OptionParser()

parser.add_option("-p", "--path", dest="dataset_path", help="Path to training data.", default="dataset")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).",
                  action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90",
                  help="Augment with 90 degree rotations in training. (Default=false).", action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=20)
parser.add_option("--output_dir", dest="out_dir",
                  help="Location to save model and store all the metadata related to the training (to be used when testing).", default="output")
parser.add_option("--input_weight_path", dest="input_weight_path",
                  help="Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()

# if not options.train_path:   # if dataset path is not defined
#     parser.error('Error: path to training data must be specified. Pass --path to command line')

all_config = config.Config()

all_config.use_horizontal_flips = bool(options.horizontal_flips)
all_config.use_vertical_flips = bool(options.vertical_flips)
all_config.rot_90 = bool(options.rot_90)

all_config.model_path = this_dir / options.out_dir / "Frcnn_resnet50"
all_config.num_rois = int(options.num_rois)

all_imgs, classes_count, class_mapping = pascalVOC_parser.get_data(Path(options.dataset_path))

all_config.class_mapping = class_mapping
inv_map = {v: k for k, v in class_mapping.items()}

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

def get_img_output_length(width, height):
    def get_output_length(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    return get_output_length(width), get_output_length(height)

# train_datagen = KerasDataGenerator(train_imgs, all_config, get_img_output_length)
train_datagen = simple_genetator(train_imgs, all_config, get_img_output_length)

re = []
for i in range(10):
    X, y, meta_info = next(train_datagen)
    re.append(meta_info)


print()