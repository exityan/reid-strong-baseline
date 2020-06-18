# encoding: utf-8
import re
from pathlib import Path
import os.path as osp
from collections import defaultdict

from .bases import BaseImageDataset


class LPW(BaseImageDataset):
    """
    Labeled Pedestrian in the Wild
    URL: http://liuyu.us/dataset/lpw/index.html

    train set  : scene2 + scene3
    query set  : scene1(view2)
    gallery set: scene1(view1+view3)

    Dataset statistics:
      ----------------------------------------
      subset   | # ids | # images | # cameras
      ----------------------------------------
      train    |  1975 |   418739 |         8
      query    |   756 |    63118 |         1
      gallery  |   756 |   108690 |         2
      ----------------------------------------
    """

    dataset_dir = "lpw"

    def __init__(self, root, verbose=True, **kwargs):
        super(LPW, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        # data dirs = [(scene_no, view_no, camid), ...]
        self.train_dirs = [
            (1, 1, 0),
            (1, 2, 1),
            (1, 3, 2),
            (2, 1, 3),
            (2, 2, 4),
            (2, 3, 5),
            (2, 4, 6),
            (3, 1, 7),
            (3, 2, 8),
            (3, 3, 9),
            (3, 4, 10),
        ]
        self.query_dirs = [
        ]
        self.gallery_dirs = [
        ]

        self._check_before_run()

        train = self._process_train_dir(self.train_dirs)
        query = self._process_test_dir(self.query_dirs, relabel=False)
        gallery = self._process_test_dir(self.gallery_dirs, relabel=False)

        if verbose:
            print("=> LPW loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        (
            self.num_train_pids,
            self.num_train_imgs,
            self.num_train_cams,
        ) = self.get_imagedata_info(self.train)
        (
            self.num_query_pids,
            self.num_query_imgs,
            self.num_query_cams,
        ) = self.get_imagedata_info(self.query)
        (
            self.num_gallery_pids,
            self.num_gallery_imgs,
            self.num_gallery_cams,
        ) = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _process_train_dir(self, data_dirs):
        # dataset = []
        re_pid_dir = re.compile(r"\d+")

        dataset_dict = defaultdict(list)
        for scene_no, view_no, camid in data_dirs:
            pathname = f"scen{scene_no}/view{view_no}"
            dir_path = Path(osp.join(self.dataset_dir, pathname))
            for pid_dir in dir_path.iterdir():
                if pid_dir.is_dir() and re_pid_dir.match(pid_dir.name):
                    pid = int(pid_dir.name)
                    for item in pid_dir.iterdir():
                        m = re.match(r"\d+.jpg", item.name)
                        if m is None:
                            print(f"[WARN] {item.name} is not valid file name")
                            continue
                        dataset_dict[scene_no].append((str(item), pid, camid))
                        # dataset.append((str(item), pid, camid))

        scene_pid_set = set()
        for scene_no, data in dataset_dict.items():
            for img_path, pid, camid in data:
                scene_pid_set.add((scene_no, pid))
        scene_pid_to_label = {
            scene_pid: label for label, scene_pid in enumerate(sorted(scene_pid_set))
        }

        dataset_relabel = []
        for scene_no, data in dataset_dict.items():
            for img_path, pid, camid in data:
                dataset_relabel.append(
                    (img_path, scene_pid_to_label[(scene_no, pid)], camid)
                )
        return dataset_relabel

    def _process_test_dir(self, data_dirs, relabel=False):
        dataset = []
        re_pid_dir = re.compile(r"\d+")

        for scene_no, view_no, camid in data_dirs:
            pathname = f"scen{scene_no}/view{view_no}"
            dir_path = Path(osp.join(self.dataset_dir, pathname))
            for pid_dir in dir_path.iterdir():
                if pid_dir.is_dir() and re_pid_dir.match(pid_dir.name):
                    pid = int(pid_dir.name)
                    for item in pid_dir.iterdir():
                        m = re.match(r"\d+.jpg", item.name)
                        if m is None:
                            print(f"[WARN] {item.name} is not valid file name")
                            continue
                        dataset.append((str(item), pid, camid))

        if relabel is not True:
            return dataset

        pid_set = set()
        for _, pid, _ in dataset:
            pid_set.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_set))}

        dataset_relabel = []
        for img_path, pid, camid in dataset:
            dataset_relabel.append((img_path, pid2label[pid], camid))
        return dataset_relabel
