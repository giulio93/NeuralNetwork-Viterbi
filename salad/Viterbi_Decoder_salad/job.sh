#!/bin/sh
source /venv/bin/activate

python3 train.py ----decoding_path=data/split1.train --results_path=result1/
python3 inference.py --decoding_path=data/split1.test --results_path=result1/
python eval.py --recog_dir=result1 --ground_truth_dir=data/groundTruth

python3 train.py ----decoding_path=data/split2.train --results_path=result2/
python3 inference.py --decoding_path=data/split2.test --results_path=result2/
python eval.py --recog_dir=result2 --ground_truth_dir=data/groundTruth

python3 train.py ----decoding_path=data/split3.train --results_path=result3/
python3 inference.py --decoding_path=data/split3.test --results_path=result3/
python eval.py --recog_dir=result3 --ground_truth_dir=data/groundTruth

python3 train.py ----decoding_path=data/split4.train --results_path=result4/
python3 inference.py --decoding_path=data/split4.test --results_path=result4/
python eval.py --recog_dir=result4 --ground_truth_dir=data/groundTruth

python3 train.py ----decoding_path=data/split5.train --results_path=result5/
python3 inference.py --decoding_path=data/split5.test --results_path=result5/
python eval.py --recog_dir=result5 --ground_truth_dir=data/groundTruth