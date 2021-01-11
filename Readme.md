# OpenPifPaf Human Action Prediction Plugin

### Installation
```
# install openpifpaf
git clone https://github.com/guarin/openpifpaf
cd openpifpaf
git checkout my_dev
pip install -e .[train,test]
cd ..

# install openpifpaf_action plugin
git clone openpifpaf_action
cd openpifpaf_action
pip install -e .

# download Pascal VOC 2012 devkit from http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/index.html
# move action_eval_fun.m into VOCdevkit directory
# install matlab engine for python (used for Pascal VOC Evaluation) https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

# download Pascal VOC 2012 data from http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/index.html
# download Stanford40 data from http://vision.stanford.edu/Datasets/40actions.html

# create pifpaf keypoint predictions for both datasets
python -m openpifpaf.predict --glob data/stanford40/images/*.jpg --checkpoint shufflenetv2k30 --seed-threshold 0.2 --force-complete-pose --json-output outputs/stanford_pifpaf_predictions/predictions
python -m openpifpaf.predict --glob data/voc2012/images/*.jpg --checkpoint shufflenetv2k30 --seed-threshold 0.2 --force-complete-pose --json-output outputs/voc_pifpaf_predictions/predictions

# create annotations (use --help for info on arguments)
python -m openpifpaf_action.data_preprocessing.pascal <arguments>
python -m openpifpaf_action.data_preprocessing.stanford <arguments>
``` 

### Training
Follow instructions at https://vita-epfl.github.io/openpifpaf/dev/train.html
Use `python -m openpifpaf.train --help` for an overview of all plugin arguments

### Evaluation
```
# Pascal VOC 2012
python -m openpifpaf.predict \
	--glob data/voc2012/val_images/*.jpg \
	--checkpoint <checkpoint> \
	--decoder action \
	--force-complete-pose \
	--seed-threshold 0.2 \
	--json-output outputs/<experiment name>/val_predictions_epoch000

python -m openpifpaf_action_prediction.voc_eval \
    --output-dir outputs/<experiment name>/ \
    --set-dir data/voc2012/image_sets/action/ \
    --anns-dir data/voc2012/annotations \
    --voc-devkit-dir <Pascal VOC devkit directory>


# Stanford 40
python -m openpifpaf.predict \
	--glob data/stanford40/val_images/*.jpg \
	--checkpoint <checkpoint> \
	--decoder action \
	--force-complete-pose \
	--seed-threshold 0.2 \
	--json-output outputs/<experiment name>/val_predictions_epoch000

python -m openpifpaf_action_prediction.stanford_eval \
    --output-dir outputs/<experiment name>/ \
    --set-dir data/stanford40/splits \
    --anns-dir data/stanford40/annotations
```

