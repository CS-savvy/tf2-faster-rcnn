import xml.etree.ElementTree as ET
from pathlib import Path
import traceback


def get_data(dataset_path):

    all_imgs = []
    classes_count = {}
    class_mapping = {}

    annot_path = dataset_path / 'Annotations'
    imgs_path = dataset_path / 'JPEGImages'
    trainval_imgset_path = dataset_path / 'ImageSets' / 'Main' / 'train.txt'
    test_imgset_path = dataset_path / 'ImageSets' / 'Main' / 'test.txt'

    trainval_files = []
    test_files = []
    try:
        with open(trainval_imgset_path, 'r') as f:
            for line in f:
                trainval_files.append(line.strip() + '.jpg')
            print("No of images in Train set: ", len(trainval_files))
    except Exception:
        print(traceback.format_exc())
    try:
        with open(test_imgset_path) as f:
            for line in f:
                test_files.append(line.strip() + '.jpg')
            print("No of images in Test set: ", len(test_files))
    except Exception:
        print(traceback.format_exc())

    annotation_files = annot_path.glob("*.json")
    for annot in annotation_files:
        try:
            et = ET.parse(annot)
            element = et.getroot()

            element_objs = element.findall('object')
            element_filename = element.find('filename').text
            element_width = int(element.find('size').find('width').text)
            element_height = int(element.find('size').find('height').text)

            if len(element_objs) < 1:
                continue

            annotation_data = {'filepath': imgs_path / element_filename, 'width': element_width,
                               'height': element_height, 'bboxes': []}
            if element_filename in trainval_files:
                annotation_data['imageset'] = 'trainval'
            elif element_filename in test_files:
                annotation_data['imageset'] = 'test'
            else:
                annotation_data['imageset'] = 'trainval'

            for element_obj in element_objs:
                class_name = element_obj.find('name').text
                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)

                obj_bbox = element_obj.find('bndbox')
                x1 = int(round(float(obj_bbox.find('xmin').text)))
                y1 = int(round(float(obj_bbox.find('ymin').text)))
                x2 = int(round(float(obj_bbox.find('xmax').text)))
                y2 = int(round(float(obj_bbox.find('ymax').text)))
                difficulty = int(element_obj.find('difficult').text) == 1
                annotation_data['bboxes'].append(
                    {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})

            all_imgs.append(annotation_data)

        except Exception:
            print(traceback.format_exc())

    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    return all_imgs, classes_count, class_mapping


if __name__ == '__main__':
    this_dir = Path.cwd()
    k, l, m = get_data( this_dir.parent / "dataset")
    print(len(k), l, m)