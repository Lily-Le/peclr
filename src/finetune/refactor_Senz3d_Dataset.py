'''
Ref：https://github.com/tannishpage/sign-language-detection/blob/b1906c799e293ab66e927ae7095f61b072af4833/CNN_For_Feature_Extraction/refactor_senz3D_dataset.py
'''


import os



FRAMES = "/home/zlc/cll/data/senz3d_dataset_cp/acquisitions"

labels = os.listdir(FRAMES)

def copy_all_frames(from_dir, to, attachment):
    files = os.listdir(from_dir)
    for file in files:
        old_file = open(os.path.join(from_dir, file), 'rb')
        new_file = open(os.path.join(to, attachment + "_" + file), 'wb')
        new_file.write(old_file.read())
        old_file.close()
        new_file.close()
        os.remove(os.path.join(from_dir, file))
    os.rmdir(from_dir)

for label in labels:
    gesture_sets = os.listdir(os.path.join(FRAMES, label))
    for gesture_set in gesture_sets:
        if not (os.path.exists(os.path.join(FRAMES, gesture_set))):
            os.mkdir(os.path.join(FRAMES, gesture_set))
        copy_all_frames(os.path.join(FRAMES, label, str(gesture_set)), os.path.join(FRAMES, gesture_set), str(label) + "_" + str(gesture_set))
    os.rmdir(os.path.join(FRAMES, label))