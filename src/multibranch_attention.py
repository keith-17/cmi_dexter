extractor = SequenceExtractor(
    output_format='chunks',

    # Accelerometer
    acc_modes='smoothed|velocity|displacement|jerk',
    use_acc_magnitude=False,
    use_linear_acc_magnitude=False,
    linear_acc_mode=None,
    use_highpass_fallback=True,

    # Rotation
    rotation_modes='quaternion|euler|angular_velocity',
    fix_quaternion_sign=True,

    # ToF
    tof_modes='pooled_stats|sensor_stats',
    tof_fill_mode='far_255',
    tof_n_sensors=5,

    # Thermo
    thm_modes='centered_diff',

    # Frame stats (irrelevant for chunks, kept for parity)
    frame_stats='mean,std,min,max,last,first,rms',
)

clf = multibranch_keras.MultiBranchSequenceClassifier(
    extractor=extractor,
    primary_target=TARGET_COL,
    random_state=random_state,
)

pipeline = Pipeline([
    ('augmentor', SensorAugmentor(
        sequence_col='sequence_id',
        counter_col='sequence_counter',
        prob=0.0,
        per_aug_prob=0.0,
        seed=random_state,
        # these are the new searchable knobs
        augmentations=None,
        warp_sigma=0.2,
        warp_num_knots=4,
        crop_frac_range=(0.5, 0.9),
    )),
    ('estimator', clf),
])