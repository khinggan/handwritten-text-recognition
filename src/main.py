"""
Provides options via the command line to perform project tasks.
* `--source`: dataset/model name (bentham, iam, rimes, saintgall, washington)
* `--arch`: network to be used (puigcerver, bluche, flor)
* `--transform`: transform dataset to the HDF5 file
* `--cv2`: visualize sample from transformed dataset
* `--image`: predict a single image with the source parameter
* `--train`: train model with the source argument
* `--test`: evaluate and predict model with the source argument
* `--kaldi_assets`: save all assets for use with kaldi
* `--epochs`: number of epochs
* `--batch_size`: number of batches
"""

import argparse
import cv2
import h5py
import os
import string
import datetime
import numpy as np

from data import preproc as pp, evaluation
from data.generator import DataGenerator, Tokenizer
from data.reader import Dataset
from kaldiio import WriteHelper
from network.model import HTRModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--arch", type=str, default="puigcerver_words")

    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--cv2", action="store_true", default=False)
    parser.add_argument("--image", type=str, default="")

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--decode", choices=["greedy", "prefix_beam"], default="greedy")
    parser.add_argument("--kaldi_assets", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    raw_path = os.path.join("..", "raw", args.source)
    source_path = os.path.join("..", "data", f"{args.source}.hdf5")
    output_path = os.path.join("..", "output", args.source, args.arch)
    target_path = os.path.join(output_path, "checkpoint_weights.hdf5")

    # input_size = (1024, 128, 1)
    # max_text_length = 128
    input_size = (128, 32, 1)
    max_text_length = 32
    charset_base = string.printable[:79]

    if args.transform:
        assert os.path.exists(raw_path)
        print(f"The {args.source} dataset will be transformed...")

        ds = Dataset(source=raw_path, name=args.source)
        ds.read_partitions()

        print("Partitions will be preprocessed...")
        ds.preprocess_partitions(input_size=input_size)

        print("Partitions will be saved...")
        os.makedirs(os.path.dirname(source_path), exist_ok=True)

        # for i in ds.partitions:
        #     with h5py.File(source_path, "a") as hf:
        #         hf.create_dataset(f"{i}/dt", data=ds.dataset[i]['dt'], compression="gzip", compression_opts=9)
        #         hf.create_dataset(f"{i}/gt", data=ds.dataset[i]['gt'], compression="gzip", compression_opts=9)
        #         print(f"[OK] {i} partition.")

        # hdf5 变量;
        # train/dt, train/gt,     valid/dt, valid/gt,      test/dt, test/gt

        for partition in ds.partitions:
            for i, item in enumerate(ds.dataset[partition]['dt']):
                if item is None:
                    ds.dataset[partition]['dt'].pop(i)
                    ds.dataset[partition]['gt'].pop(i)
                    print(i, ds.dataset[partition]['gt'][i])

        hdf5_file = h5py.File(source_path, 'a')
        hdf5_file.create_dataset('train/dt', (len(ds.dataset['train']['dt']), 128, 32), np.int)
        hdf5_file.create_dataset('test/dt', (len(ds.dataset['test']['dt']), 128, 32), np.int)
        hdf5_file.create_dataset('valid/dt', (len(ds.dataset['valid']['dt']), 128, 32), np.int)

        string_type = h5py.special_dtype(vlen=str)
        hdf5_file.create_dataset('train/gt', (len(ds.dataset['train']['gt']), ), dtype=string_type)
        hdf5_file.create_dataset('test/gt', (len(ds.dataset['test']['gt']), ), dtype=string_type)
        hdf5_file.create_dataset('valid/gt', (len(ds.dataset['valid']['gt']), ), dtype=string_type)

        hdf5_file['train/dt'][...] = ds.dataset['train']['dt']
        hdf5_file['test/dt'][...] = ds.dataset['test']['dt']
        hdf5_file['valid/dt'][...] = ds.dataset['valid']['dt']

        hdf5_file['train/gt'][...] = ds.dataset['train']['gt']
        hdf5_file['test/gt'][...] = ds.dataset['test']['gt']
        hdf5_file['valid/gt'][...] = ds.dataset['valid']['gt']

        print(f"Transformation finished.")

    elif args.cv2:
        with h5py.File(source_path, "r") as hf:
            dt = hf['test']['dt'][:256]
            gt = hf['test']['gt'][:256]

        predict_file = os.path.join(output_path, "predict.txt")
        predicts = [''] * len(dt)

        if os.path.isfile(predict_file):
            with open(predict_file, "r") as lg:
                predicts = [line[5:] for line in lg if line.startswith("TE_P")]

        for x in range(len(dt)):
            print(f"Image shape:\t{dt[x].shape}")
            print(f"Ground truth:\t{gt[x].decode()}")
            print(f"Predict:\t{predicts[x]}\n")

            cv2.imshow("img", pp.adjust_to_see(dt[x]))
            cv2.waitKey(0)

    elif args.image:
        tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)

        img = pp.preproc(args.image, input_size=input_size)
        x_test = pp.normalization([img])

        model = HTRModel(architecture=args.arch,
                         input_size=input_size,
                         vocab_size=tokenizer.vocab_size,
                         top_paths=10)

        model.compile()
        model.load_checkpoint(target=target_path)

        predicts, probabilities = model.predict(x_test, ctc_decode=True)
        predicts = [[tokenizer.decode(x) for x in y] for y in predicts]

        print("\n####################################")
        for i, (pred, prob) in enumerate(zip(predicts, probabilities)):
            print("\nProb.  - Predict")

            for (pd, pb) in zip(pred, prob):
                print(f"{pb:.4f} - {pd}")

            cv2.imshow(f"Image {i + 1}", pp.adjust_to_see(img))
        print("\n####################################")
        cv2.waitKey(0)

    else:
        assert os.path.isfile(source_path) or os.path.isfile(target_path)
        os.makedirs(output_path, exist_ok=True)

        dtgen = DataGenerator(source=source_path,
                              batch_size=args.batch_size,
                              charset=charset_base,
                              max_text_length=max_text_length,
                              predict=args.test)

        model = HTRModel(architecture=args.arch,
                         input_size=input_size,
                         vocab_size=dtgen.tokenizer.vocab_size,
                         greedy=True)

        # set `learning_rate` parameter or get architecture default value
        model.compile(learning_rate=0.001)
        model.load_checkpoint(target=target_path)

        if args.train:
            model.summary(output_path, "summary.txt")
            callbacks = model.get_callbacks(logdir=output_path, checkpoint=target_path, verbose=1)

            start_time = datetime.datetime.now()

            h = model.fit(x=dtgen.next_train_batch(),
                          epochs=args.epochs,
                          steps_per_epoch=dtgen.steps['train'],
                          validation_data=dtgen.next_valid_batch(),
                          validation_steps=dtgen.steps['valid'],
                          callbacks=callbacks,
                          shuffle=True,
                          verbose=1)

            total_time = datetime.datetime.now() - start_time

            loss = h.history['loss']
            val_loss = h.history['val_loss']

            min_val_loss = min(val_loss)
            min_val_loss_i = val_loss.index(min_val_loss)

            time_epoch = (total_time / len(loss))
            total_item = (dtgen.size['train'] + dtgen.size['valid'])

            t_corpus = "\n".join([
                f"Total train images:      {dtgen.size['train']}",
                f"Total validation images: {dtgen.size['valid']}",
                f"Batch:                   {dtgen.batch_size}\n",
                f"Total time:              {total_time}",
                f"Time per epoch:          {time_epoch}",
                f"Time per item:           {time_epoch / total_item}\n",
                f"Total epochs:            {len(loss)}",
                f"Best epoch               {min_val_loss_i + 1}\n",
                f"Training loss:           {loss[min_val_loss_i]:.8f}",
                f"Validation loss:         {min_val_loss:.8f}"
            ])

            with open(os.path.join(output_path, "train.txt"), "w") as lg:
                lg.write(t_corpus)
                print(t_corpus)

        elif args.test:
            start_time = datetime.datetime.now()
            predicts = []
            ground_truths = []
            confidences = []
            if args.decode == "greedy":
                predicts, confidences = model.predict(x=dtgen.next_test_batch(),
                                                      steps=dtgen.steps['test'],
                                                      ctc_decode=True,
                                                      verbose=1)
                predicts = [dtgen.tokenizer.decode(x[0]) for x in predicts]
                ground_truths = dtgen.dataset['test']['gt']
                with open(os.path.join(output_path, "predict.txt"), "w") as lg:
                    for pd, gt, conf in zip(predicts, ground_truths, confidences):
                        lg.write(f"TE_L {gt}\nTE_P {pd}\nCONF {conf}\n")
            elif args.decode == "prefix_beam":
                model.greedy = False
                predicts, confidences = model.predict(x=dtgen.next_test_batch(),
                                                      steps=dtgen.steps['test'],
                                                      ctc_decode=True,
                                                      verbose=1)
                predicts = [dtgen.tokenizer.decode(x[0]) for x in predicts]

                new_predicts = []
                new_confidences = []
                new_gt = []
                for i, confidence in enumerate(confidences):
                    if confidence[0] != 0 and (confidence[1] != -np.float('inf') or confidence[1] != -np.float('inf')):
                        new_confidences.append((confidence[0] - confidence[1])/(confidence[0]+ confidence[1]))
                        new_predicts.append(predicts[i])
                        new_gt.append(dtgen.dataset['test']['gt'][i])

                confidences = [(x[0] - x[1]) / (x[1] + x[0]) for x in confidences]
                confidences = [x[0] for x in confidences]

                new_predicts = []
                new_confidences = []
                new_gt = []
                for i, confidence in enumerate(confidences):
                    if confidence[0] == 0:
                        if confidence[1] == 0:
                            new_confidences.append(0)
                            new_predicts.append(predicts[i])
                            new_gt.append(dtgen.dataset['test']['gt'][i])
                        else:
                            new_confidences.append(confidence[0])
                            new_predicts.append(predicts[i])
                            new_gt.append(dtgen.dataset['test']['gt'][i])
                    elif confidence[0] != 0 and (confidence[1] != -np.float('inf') or confidence[0] != -np.float('inf')):
                        new_confidences.append(1 - confidence[1] / confidence[0])
                        new_predicts.append(predicts[i])
                        new_gt.append(dtgen.dataset['test']['gt'][i])
                    else:
                        continue
                    with open(os.path.join(output_path, "predict.txt"), "w") as lg:
                        for pd, gt, conf in zip(predicts, dtgen.dataset['test']['gt'], new_confidences):
                            lg.write(f"TE_L {gt}\nTE_P {pd}\nCONF {conf}\n")
                predicts = new_predicts
                ground_truths = new_gt
                confidences = new_confidences
            total_time = datetime.datetime.now() - start_time

            evaluate = evaluation.ocr_metrics(predicts=predicts,
                                              ground_truth=dtgen.dataset['test']['gt'],
                                              norm_accentuation=False,
                                              norm_punctuation=False)

            e_corpus = "\n".join([
                f"Total test images:    {dtgen.size['test']}",
                f"Total time:           {total_time}",
                f"Time per item:        {total_time / dtgen.size['test']}\n",
                f"Metrics:",
                f"Character Error Rate: {evaluate[0]:.8f}",
                f"Word Error Rate:      {evaluate[1]:.8f}",
                f"Sequence Error Rate:  {evaluate[2]:.8f}"
            ])
            # draw ROC and calculate AUC
            y_test = [0 if pd == gt else 1 for pd, gt in zip(predicts, ground_truths)]
            evaluation.draw_roc(y_test, confidences)

            with open(os.path.join(output_path, "evaluate.txt"), "w") as lg:
                lg.write(e_corpus)
                print(e_corpus)

        elif args.kaldi_assets:
            predicts = model.predict(x=dtgen.next_test_batch(),
                                     steps=dtgen.steps['test'],
                                     ctc_decode=False,
                                     verbose=1)

            # get data and ground truth lists
            ctc_TK, space_TK = "<ctc>", "<space>"
            multigrams, multigrams_size = dict(), 0
            ground_truth = []

            # generate multigrams to compose the dataset
            for pt in dtgen.partitions:
                multigrams[pt] = [pp.generate_multigrams(x) for x in dtgen.dataset[pt]['gt']]
                multigrams[pt] = list(set([pp.text_standardize(y) for x in multigrams[pt] for y in x]))

                multigrams[pt] = [x for x in multigrams[pt] if Dataset.check_text(x)]
                multigrams_size += len(multigrams[pt])

                for x in multigrams[pt]:
                    ground_truth.append([space_TK if y == " " else y for y in list(f" {x} ")])

                for x in dtgen.dataset[pt]:
                    ground_truth.append([space_TK if y == " " else y for y in list(f" {x} ")])

            # define dataset size and default tokens
            train_size = dtgen.size['train'] + dtgen.size['valid'] + multigrams_size

            # get chars list and save with the ctc and space tokens
            chars = list(dtgen.tokenizer.chars) + [ctc_TK]
            chars[chars.index(" ")] = space_TK

            kaldi_path = os.path.join(output_path, "kaldi")
            os.makedirs(kaldi_path, exist_ok=True)

            with open(os.path.join(kaldi_path, "chars.lst"), "w") as lg:
                lg.write("\n".join(chars))

            ark_file_name = os.path.join(kaldi_path, "conf_mats.ark")
            scp_file_name = os.path.join(kaldi_path, "conf_mats.scp")

            # save ark and scp file (laia output/kaldi input format)
            with WriteHelper(f"ark,scp:{ark_file_name},{scp_file_name}") as writer:
                for i, item in enumerate(predicts):
                    writer(str(i + train_size), item)

            # save ground_truth.lst file with sparse sentences
            with open(os.path.join(kaldi_path, "ground_truth.lst"), "w") as lg:
                for i, item in enumerate(ground_truth):
                    lg.write(f"{i} {' '.join(item)}\n")

            # save indexes of the train/valid and test partitions
            with open(os.path.join(kaldi_path, "ID_train.lst"), "w") as lg:
                range_index = [str(i) for i in range(0, train_size)]
                lg.write("\n".join(range_index))

            with open(os.path.join(kaldi_path, "ID_test.lst"), "w") as lg:
                range_index = [str(i) for i in range(train_size, train_size + dtgen.size['test'])]
                lg.write("\n".join(range_index))
