# Post-Hoc Model Calibration

The directory contains post-hoc model calibration methods. When specified in the configuration files, post-hoc model calibration techniques are initiated after training is completed and after re-training is completed.

## Currently Implemented Post-Hoc Model Calibration Methods

This module currently contains the following post-hoc methods for model calibration:
    - Temperature Scaling

## Enabling Post-Hoc Model Calibration Methods

To use model calibration during training and/or active learning, edit the config file. The `model_config` dictionary in the config (shown below) should be modified to specify the desired `calibration` method. 

```yaml 
    model_config:
        model_name: ActiveModel
        architecture: CifarResnet20
        calibration: TemperatureScaling
```