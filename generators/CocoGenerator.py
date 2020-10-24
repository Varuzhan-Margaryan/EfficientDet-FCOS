from pycocotools.coco import COCO
import cv2, os
import tensorflow as tf
import numpy as np
import preprocessing

from urllib.request import urlopen
from PIL import Image

class CocoGenerator:
    def __init__(self, data_dir=r'\data', set_name='val2017', img_size=(224, 224),
                 map_size=(112, 112), shuffle=True):
        self.data_dir = data_dir
        self.set_name = set_name
        self.img_size = img_size
        self.map_size = map_size
        
        if set_name in ['train2017', 'val2017']:
            self.coco = COCO(os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json'))
        else:
            self.coco = COCO(os.path.join(data_dir, 'annotations', 'image_info_' + set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        if shuffle:
            np.random.shuffle(self.image_ids)

        self.load_classes()

    def load_classes(self):
        """
        Loads the class to label mapping (and inverse) for COCO.
        """
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            id = len(self.classes)
            self.coco_labels[id] = c['id']
            self.coco_labels_inverse[c['id']] = id
            self.classes[c['name']] = id

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def size(self):
        """ Size of the COCO dataset.
        """
        return len(self.image_ids)

    def num_classes(self):
        """ Number of classes in the dataset. For COCO this is 80.
        """
        return 80

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def coco_label_to_label(self, coco_label):
        """ Map COCO label to the label as used in the network.
        COCO has some gaps in the order of labels. The highest label is 90, but there are 80 classes.
        """
        return self.coco_labels_inverse[coco_label]

    def coco_label_to_name(self, coco_label):
        """ Map COCO label to name.
        """
        return self.label_to_name(self.coco_label_to_label(coco_label))

    def label_to_coco_label(self, label):
        """ Map label as used by the network to labels as used by COCO.
        """
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        # {'license': 2, 'file_name': '000000259765.jpg', 'coco_url': 'http://images.cocodataset.org/test2017/000000259765.jpg', 'height': 480, 'width': 640, 'date_captured': '2013-11-21 04:02:31', 'id': 259765}
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_rand_image(self):
        """
        Load a random image.
        """
        # {'license': 2, 'file_name': '000000259765.jpg', 'coco_url': 'http://images.cocodataset.org/test2017/000000259765.jpg', 'height': 480, 'width': 640, 'date_captured': '2013-11-21 04:02:31', 'id': 259765}
        image_index = np.random.choice( len(self.image_ids))
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        # get ground truth annotations
        img_id = self.image_ids[image_index]
        img_info = self.coco.loadImgs(img_id)
        annotations_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        annotations = {'labels': np.empty((0,), dtype=np.float32), 'bboxes': np.empty((0, 4), dtype=np.float32),
                       'masks': np.empty((0, img_info['height'], img_info['width'], 3), dtype=np.float32)}

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        coco_annotations.sort(key='area', reverse=True)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotations['labels'] = np.concatenate(
                [annotations['labels'], [a['category_id'] - 1]], axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [a['bbox']]], axis=0)
            annotations['masks'] = np.concatenate([annotations['masks'], [self.coco.annToMask(a)]], axis=0)

        return annotations

    def load_img_and_anns(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.data_dir, 'images', self.set_name, img_info['file_name'])
        if os.path.exists(path):
            image = cv2.imread(path)
        else:
            try:
                image = np.array(Image.open(urlopen(img_info['coco_url'])))
            except:
                return None, None, None, None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        annotations_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        labels, boxes, masks = [], [], []

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return None, None, None, None

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        coco_annotations.sort(key=lambda x: x['area'], reverse=True)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 5 or a['bbox'][3] < 5:
                continue

            labels.append(self.coco_label_to_label(a['category_id']))
            boxes.append(a['bbox'])
            masks.append(self.coco.annToMask(a))

        if len(annotations_ids) == 0:
            return None, None, None, None

        return image, labels, boxes, masks

    def generate_data(self):
        for i in self.image_ids:
            image, labels, boxes, masks = self.load_img_and_anns(i)
            if image is None or len(masks)<1:
                continue
            class_map, box_map, centerness_map = preprocessing.get_resized_maps(labels, boxes, masks, self.map_size)
            image = cv2.resize(image, self.img_size)
            yield image, class_map, box_map, centerness_map

    def generate_test_data(self):
        for i in self.image_ids:
            image = self.load_image(i)
            if image is None:
                continue
            image = cv2.resize(image, self.img_size)
            yield image.astype('float32')
