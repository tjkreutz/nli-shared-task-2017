import argparse
import os
from shutil import copytree


def check_required_paths_exist(shared_task_dir):
    pth_exts_to_check = ("data", "data/essays", "data/speech_transcriptions", 
                         "data/features", "data/features/essays", 
                         "data/features/speech_transcriptions", 
                         "data/features/ivectors", "data/labels")

    pths_to_create = ("data/features/essays/test", 
                      "data/features/speech_transcriptions/test",  
                      "data/features/speech_transcriptions+ivectors/test")

    for pth in pth_exts_to_check:
        full_path = os.path.join(shared_task_dir, pth)
        if not os.path.isdir(full_path):
            raise FileNotFoundError("Directory {} does not exist.".format(full_path))

    for pth in pths_to_create:
        full_path = os.path.join(shared_task_dir, pth) 
        os.mkdir(full_path)
        print("Created directory: '{}'".format(full_path))
    
    return True


def copy_test_data(shared_task_dir):
    pths_to_copy = ("data/essays", "data/speech_transcriptions", "data/labels", "data/features/ivectors")
    for pth in pths_to_copy:
        src = os.path.join(".", pth, "test")
        target = os.path.join(shared_task_dir, pth, 'test')
        copytree(src, target)
        print("Copied {} to {}".format(src, target))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("path_to_shared_task_directory")
    args = argparser.parse_args()
    project_dir = args.path_to_shared_task_directory
    check_required_paths_exist(project_dir)
    copy_test_data(project_dir)
