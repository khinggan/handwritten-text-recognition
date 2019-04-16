from glob import glob
import argparse
import shutil
import os


def norm_partitions(origin, target):
    origin_dir = os.path.join(
        origin, "largeWriterIndependentTextLineRecognitionTask")
    target_dir = os.path.join(target, "partitions")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    set_file = os.path.join(origin_dir, "trainset.txt")
    new_set_file = os.path.join(target_dir, "train.txt")
    shutil.copy(set_file, new_set_file)

    set_file = os.path.join(origin_dir, "validationset1.txt")
    new_set_file = os.path.join(target_dir, "validation.txt")
    shutil.copy(set_file, new_set_file)

    set_file = os.path.join(origin_dir, "testset.txt")
    new_set_file = os.path.join(target_dir, "test.txt")
    shutil.copy(set_file, new_set_file)


def norm_gt(origin, target):
    origin_dir = os.path.join(origin, "ascii")
    target_dir = os.path.join(target, "gt")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    set_file = os.path.join(origin_dir, "lines.txt")

    with open(set_file) as f:
        content = [x.strip() for x in f.readlines()]

        for line in content:
            if (not line or line[0] == "#"):
                continue

            splited = line.strip().split(' ')
            assert len(splited) >= 9

            file_name = splited[0]
            file_text = splited[len(splited)-1].replace("|", " ")

            new_set_file = os.path.join(target_dir, f"{file_name}.txt")

            with open(new_set_file, "w+") as f:
                f.write(file_text.strip())
                f.close()


def norm_lines(origin, target):
    origin_dir = os.path.join(origin, "lines")
    target_dir = os.path.join(target, "lines")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    glob_filter = os.path.join(origin_dir, "**", "*.*")
    files = [x for x in glob(glob_filter, recursive=True)]

    for f in files:
        shutil.copy(f, target_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    src = args.data_dir
    src_backup = f"{src}_backup"

    if not os.path.exists(src_backup):
        os.rename(src, src_backup)

    norm_partitions(src_backup, src)
    norm_gt(src_backup, src)
    norm_lines(src_backup, src)


if __name__ == '__main__':
    main()