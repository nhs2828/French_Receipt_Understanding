for segmentation, need to add training image on receipt that have fold in the middle, else the model would see it as 2 different receipts
x-anylabeling for annotation

layoutlmv2 needs detectrion2 CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

convert paddle ot onnx
paddle2onnx \
    --model_dir ./path/to/your/model_folder \
    --model_filename inference.json \
    --params_filename inference.pdiparams \
    --save_file ./model.onnx \
    --opset_version 14 \
    --enable_onnx_checker True