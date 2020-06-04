from .bases import BaseImageDataset


class CompositeDataset(BaseImageDataset):
    def __init__(self, datasets, verbose=True):
        self.train = []
        self.query = []
        self.gallery = []
        self._process_train_datasets(datasets)
        self._process_test_datasets(datasets)
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
        if verbose:
            print("=> Composite dataset loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

    def _process_train_datasets(self, datasets):
        pid_container = set()
        camid_container = set()
        for dataset in datasets:
            # NOTE: `data` = (img_path, pid, camid)
            for data in dataset.train:
                pid_container.add((dataset.dataset_dir, data[1]))
                camid_container.add((dataset.dataset_dir, data[2]))

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        camid2label = {camid: label for label, camid in enumerate(camid_container)}

        for dataset in datasets:
            for data in dataset.train:
                pid = pid2label[(dataset.dataset_dir, data[1])]
                camid = camid2label[(dataset.dataset_dir, data[2])]
                self.train.append((data[0], pid, camid))

    def _process_test_datasets(self, datasets):
        pid_container = set()
        camid_container = set()
        for dataset in datasets:
            for data in dataset.query:
                pid_container.add((dataset.dataset_dir, data[1]))
                camid_container.add((dataset.dataset_dir, data[2]))
            for data in dataset.gallery:
                pid_container.add((dataset.dataset_dir, data[1]))
                camid_container.add((dataset.dataset_dir, data[2]))

        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        camid2label = {camid: label for label, camid in enumerate(camid_container)}

        for dataset in datasets:
            for data in dataset.query:
                pid = pid2label[(dataset.dataset_dir, data[1])]
                camid = camid2label[(dataset.dataset_dir, data[2])]
                self.query.append((data[0], pid, camid))
            for data in dataset.gallery:
                pid = pid2label[(dataset.dataset_dir, data[1])]
                camid = camid2label[(dataset.dataset_dir, data[2])]
                self.gallery.append((data[0], pid, camid))
