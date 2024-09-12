# Augmentation functions
def choose_aug(aug, args):
    imsize = args["imsize"]
    a_end_tf = A.Compose(
        [
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    keep_aspect_resize = A.Compose(
        [
            A.LongestMaxSize(max_size=imsize),
            A.PadIfNeeded(min_height=imsize, min_width=imsize, border_mode=cv2.BORDER_CONSTANT, value=0),
        ],
        p=1.0,
    )

    if aug.startswith("aug-02"):
        apply_eq = "EQ" in aug
        apply_bw = "BW" in aug
        keep_aspect = "keep-aspect" in aug
        border = 0
        transform_test = A.Compose(
            [
                A.ToGray(p=1.0) if apply_bw else A.NoOp(),
                A.Equalize(p=1.0) if apply_eq else A.NoOp(),
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                a_end_tf,
            ]
        )
        transform_train = A.Compose(
            [
                A.ToGray(p=1.0) if apply_bw else A.NoOp(),
                A.Equalize(p=1.0) if apply_eq else A.NoOp(),
                A.Posterize(p=0.1),
                A.NoOp() if apply_eq else A.Equalize(p=0.2),
                A.CLAHE(clip_limit=2.0),
                A.OneOf(
                    [
                        A.GaussianBlur(),
                        A.Sharpen(),
                    ],
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Solarize(p=0.2),
                A.OneOf(
                    [
                        A.ColorJitter(),
                        A.RGBShift(
                            r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2
                        ),
                        A.NoOp() if apply_bw else A.ToGray(p=0.5),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.Rotate(limit=360, border_mode=border),
                        A.Perspective(pad_mode=border),
                        A.Affine(
                            scale=(0.5, 0.9),
                            translate_percent=0.1,
                            shear=(-30, 30),
                            rotate=360,
                        ),
                    ],
                    p=0.8,
                ),
                keep_aspect_resize if keep_aspect else A.Resize(imsize, imsize, p=1.0),
                A.CoarseDropout(
                    max_holes=30,
                    max_height=15,
                    max_width=15,
                    min_holes=1,
                    min_height=2,
                    min_width=2,
                ),
                A.RandomSizedCrop(
                    min_max_height=(int(0.5 * imsize), int(0.8 * imsize)),
                    height=imsize,
                    width=imsize,
                    p=0.3,
                ),
                a_end_tf,
            ]
        )
        tf_test = lambda x: transform_test(image=np.array(x))["image"]
        tf_train = lambda x: transform_train(image=np.array(x))["image"]
    else:
        raise ValueError(f"Invalid augmentation value {aug}")

    return tf_test, tf_train

# Dataset class
class Target(Dataset):
    def __init__(self, csv_file, root_dir, split, split_column, transform=None, aug=None, dataset_type='F1'):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.split = split
        self.split_column = split_column
        self.transform = transform
        self.aug = aug
        self.dataset_type = dataset_type
        self.data_frame = self.data_frame[self.data_frame[self.split_column] == self.split]
        self.class_names = pd.Categorical(self.data_frame['taxon']).categories.tolist()
        self.num_classes = len(pd.Categorical(self.data_frame['taxon']).categories)
        self.data_frame['label'] = pd.Categorical(self.data_frame['taxon']).codes
        

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if self.dataset_type == 'F1':
            taxon = self.data_frame.iloc[idx, 0]
            image_name = self.data_frame.iloc[idx, 2]
            img_path = os.path.join(self.root_dir, taxon, image_name)
            label = self.data_frame.iloc[idx, -1]
        else:  # F2
            individual = self.data_frame.iloc[idx, 0]
            taxon = self.data_frame.iloc[idx, 1]
            image_name = self.data_frame.iloc[idx, 6]
            img_path = os.path.join(self.root_dir, individual, image_name)
            label = pd.Categorical(self.data_frame['taxon']).codes[idx]

        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        if self.aug:
            augmented = self.aug(image=image)
            image = augmented['image']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# Augmentation setup
IMSIZE = 224
args = {"imsize": IMSIZE}
tf_test, tf_train = choose_aug("aug-02", args)