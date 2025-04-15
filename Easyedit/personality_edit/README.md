# personality_edit

## Run

### Training

* Mend training

You can run follow commands:

```bash
CUDA_VISIBLE_DEVICES="2" \
python edit.py \
	--editing_method MEND_train \
	--hparams_path ./config/train/mend/llama3.yaml \
	--data_dir ./data
```

You can change the `hparams_path` to `llama-7b.yaml` for editing llama models, and modify the detail parameters in `.yaml` files.

### Testing

* Mend testing

After training , you can set the `archive` in the specific config file to your trained model for evaluation, and the run the following command:

```bash
CUDA_VISIBLE_DEVICES="1" python edit.py \
	--editing_method MEND_test \
	--hparams_path ./config/test/mend/llama2.yaml \
	--data_dir ./data \
	--TPEI \
	--PAE \
	--cls_path ./models/per-classifier \
	--metric_file ./MEND_test_metrics.json
```

* IKE testing

```bash
CUDA_VISIBLE_DEVICES="1" \
python edit.py \
	--editing_method IKE \
	--hparams_path ./config/test/ike/llama3-8B.yaml \
	--data_dir ./data \
	--TPEI \
	--PAE \
	--cls_path ./models/per-classifier
```
