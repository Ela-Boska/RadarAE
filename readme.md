# Install Dependency
- python>=3.8.3
- torch==1.7.1
- timm==0.4.12
- scikit-learn==1.0.1

# Download Dataset
Download Link: 

put the dataset to predefined position
```
cp ${path-to-dataset} ${project-base-dir}/data/
```

# Congfig Files
There are three config files: "config/runPretrain.json", "config/runClassify.json", "config/runBenchmark.json". You must add a configuration in the corresponding config file before running the corresponding script: "src/pretrain.py", "src/classify.py", "src/benchamark.py".

This is an example of using the config file and script.

Pretrain the encoder:
```
# config/runPretrain.json
{
    "RadarAE_base": # the configuration name, used when running scripts
    {
        "pretrainCfg":
        {
            "seed": 3431, # random seed
            "batch_size": 128,
            "lr": 1e-3, # learning rate
            "n_epochs": 3600, # the number of pretrain epoches
            "warmup": 0.1, # ignore it
            "save_steps": 300, # ignore it
            "total_steps": 200000000 # ignore it
        },
        "maskCfg":
        {
            "mask_ratio": 0.75, # masking ratio
            "mask_prob": 0.8, # ignore it
            "patch_size": [3,10] # patch size
        },
        "dataset":
        {
            "path":"packs/", # ignore it
            "pipeline":"[PriVA(3e5, 5e4), TshiftV6(), Normalize(), Padding((72,20)), VITmask()]" # transformation performed on the input data. These methods are defined in "src/transformations/pipeline.py".
        },
        "encoder": {
            "type": "RadarAE", # encoder model. They are defined in "models.py".
            "channel" : 2, # num of channels of the input
            "input_shape" : [72,20,2], # shape of the input
            "patch_size" : [3,10], # patch size
            "hidden": 300, # hidden size of encoder
            "hidden_ff": 256, # ignore it
            "mlp_ratio":4, # ratio: numOfFeaturesOfMLP / hidden
            "n_layers": 3, # num of transformer encoder layers of RadarAE encoder.
            "n_heads": 12, # num of heads of Multi-Head Attention layers
            "emb_norm": false, # ignore it
            "decoderType":"TFC", # using transformer decoder, don't modify it
            "hidden_decoder":150, # hidden size of decoder
            "n_layers_decoder":1, # num of transformer encoder layers of the decoder.
            "n_heads_decoder":3, # num of heads of Multi-Head Attention layers of the decoder
            "shareWeight":true # whether using weight sharing.
        },
        "save_path":"saved/Pretrain/", # ignore it
        "pretrain_model": null, # ignore it
        "use_sup_data":false # ignore it
    }
}
```
```
# python src/pretrain.py ${configuration_name} -g ${GPU_id} 
python src/pretrain.py RadarAE_base -g 0
```

Train the classifier:
```
# config/runPretrain.json
{
    "RadarAE": # configuration name.
    {
        "pretrainCfgVersion":"RadarAE_base", # configuration name of pretraining. This name has to be exactly equal to the configuration name in "config/runPretrain.json"
        "classifyCfg":
        {
            "seed": 3431, # random seed
            "batch_size": 128, # batch size
            "lr": 1e-3, # learning rate
            "n_epochs": 700, # training epoches
            "warmup": 0.1, # ignore it
            "save_steps": 300, # ignore it
            "total_steps": 200000000 # ignore it
        },
        "dataset":
        {
            "pipelineTrain":"[PriVA(3e5, 5e4), TshiftV6(), Normalize(), Padding((72,20)), VITInput()]", # processing steps for training data
            "pipelineVali":"[PriVATest(3e5, 5e4), Normalize(), Padding((72,20)), VITInput()]", # processing steps for validating data
            "pipelineTest":"[PriVATest(3e5, 5e4), Normalize(), Padding((72,20)), VITInput()]" # processing steps for testing data
        },
        "decoder":
        {
            "type":"DecoderTransformer4", # classifer model defined in "models.py"
            "input_shape":[48,300], # [${num_of_patches_for_each_stmm}, ${hidden_size_of_encoder}]
            "n_layers":2 # number of transformer layers of the classifier
        },
        "save_path":"saved/Classify/" # ignore it
    }
}
```
```
# python src/classify.py ${configuration_name} ${labeling rate} -g ${GPU_id}
python src/classify.py RadarAE 0.01 -g 0
```

You should first pretrain an encoder, then train a classifier.

For a benchmark test, you only need one configuration and one script:

```
# config/runBenchmark.json
{
    "DCNN3_shift":
    {
        "classifyCfg":
        {
            "seed": 3431,
            "batch_size": 128,
            "lr": 1e-3,
            "n_epochs": 700,
            "warmup": 0.1,
            "save_steps": 300,
            "total_steps": 200000000
        },
        "dataset":
        {
            "pipelineTrain":"[PriVA(3e5, 5e4), TshiftV6(), Padding((72,20)), Normalize()]",
            "pipelineVali":"[PriVA(3e5, 5e4), Padding((72,20)), Normalize()]",
            "pipelineTest":"[PriVA(3e5, 5e4), Padding((72,20)), Normalize()]"
        },
        "decoder":
        {
            "type":"BenchmarkDCNN3"
        },
        "save_path":"saved/Benchmark/"
    }
}
```