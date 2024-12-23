import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np


def no_processing(data):
    return data


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. 

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing, frame_sample_mode='causal'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        return self.getitem()

    def getitem(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        attempts = 0
        max_attempts = 100  # حداکثر تعداد تلاش‌ها برای نمونه‌برداری
        while not valid and attempts < max_attempts:
            attempts += 1
            try:
                # Select a dataset
                dataset = random.choices(self.datasets, self.p_datasets)[0]

                is_video_dataset = dataset.is_video_sequence()

                # sample a sequence from the given dataset
                seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
                #print(seq_id, visible)
                if is_video_dataset:
                    template_frame_ids = None
                    search_frame_ids = None
                    gap_increase = 0

                    if self.frame_sample_mode == 'causal':
                        # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                        while search_frame_ids is None:
                            base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                     max_id=len(visible) - self.num_search_frames)
                            if base_frame_id is None or len(base_frame_id) == 0:
                                print(f"Warning: Could not sample base_frame_id in sequence {seq_id}")
                                break  # Break to re-sample a different sequence
                            prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                      min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                      max_id=base_frame_id[0])
                            if prev_frame_ids is None:
                                gap_increase += 5
                                continue
                            template_frame_ids = base_frame_id + prev_frame_ids
                            search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                      max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                      num_ids=self.num_search_frames)
                            # Increase gap until a frame is found
                            gap_increase += 5
                else:
                    # In case of image dataset, just repeat the image to generate synthetic video
                    template_frame_ids = [1] * self.num_template_frames
                    search_frame_ids = [1] * self.num_search_frames

                # دریافت فریم‌ها
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                #print('load images')

                # بررسی فریم‌ها برای اطمینان از عدم وجود None
                if template_frames is None or search_frames is None:
                    print(f"Error: Received None for frames in sequence {seq_id}")
                    continue  # تلاش برای نمونه‌برداری مجدد

                # بررسی هر فریم به صورت جداگانه
                invalid_frame = False
                for idx, frame in enumerate(template_frames):
                    if frame is None:
                        print(f"Error: Template frame at index {template_frame_ids[idx]} is None in sequence {seq_id}")
                        invalid_frame = True
                        break

                if invalid_frame:
                    continue  

                for idx, frame in enumerate(search_frames):
                    if frame is None:
                        print(f"Error: Search frame at index {search_frame_ids[idx]} is None in sequence {seq_id}")
                        invalid_frame = True
                        break

                if invalid_frame:
                    continue 

                H, W, _ = template_frames[0].shape

                data = TensorDict({
                    'template_images': template_frames,
                    'template_anno': template_anno['bbox'],
                    # 'template_masks': template_masks,
                    'search_images': search_frames,
                    'search_anno': search_anno['bbox'],
                    # 'search_masks': search_masks,
                    'dataset': dataset.get_name(),
                    # 'test_class': meta_obj_test.get('object_class_name')
                })
                data = self.processing(data)

                # بررسی اعتبار داده‌ها
                if 'valid' in data and data['valid']:
                    valid = True
                else:
                    print(f"Warning: Invalid data found in sequence {seq_id}")
                    valid = False

            except Exception as e:
                print(f"Exception encountered: {e}")
                valid = False

        if not valid:
            raise RuntimeError("Failed to get a valid sample after multiple attempts")

        return data

    def _make_aabb_mask(self, map_shape, bbox):
        mask = np.zeros(map_shape, dtype=np.float32)
        mask[int(round(bbox[1].item())):int(round(bbox[1].item() + bbox[3].item())),
             int(round(bbox[0].item())):int(round(bbox[0].item() + bbox[2].item()))] = 1
        return mask

    def get_center_box(self, H, W, ratio=1/8):
        cx, cy, w, h = W / 2, H / 2, W * ratio, H * ratio
        return torch.tensor([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])

    def sample_seq_from_dataset(self, dataset, is_video_dataset):
        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            if is_video_dataset:
                enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                        self.num_search_frames + self.num_template_frames) and len(visible) >= 20
            else:
                enough_visible_frames = True  

            if not is_video_dataset:
                break  

        return seq_id, visible, seq_info_dict
