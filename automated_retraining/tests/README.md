# Tests

This folder contains scripts used for testing

## Config Tests
The config tester is used to make sure user-entered parameters and configurations are compatible with the code structure. Each config file key is and its sub-parameters are checked for consistent typing, existence of directories, etc. 

Each parameter is first checked for the proper type, and then checked for acceptable values. 

Tests not fully developed simply `pass` until a useful test is added.

## Running Config Tests
### To test all config keys and parameters, run:

with default config file (`base_config.yaml`):

`python test_configs.py`

with custom config file:

`python test_configs.py --config $PATH_TO_CONFIG_FILE`

The default config file is stored in: `../configs/base_config.yaml`

To test an active or training configuration, it is easiest to have all parameters in one config file (`base_config.yaml`), and only change the `config` key to be either `active_learning` or `training`. 

### To test a single config key, run:
`python test_configs.py --test_key $CONFIG_KEY_TO_TEST`

where `$CONFIG_KEY_TO_TEST` may be any top level key in the config file, e.g. `model_config`, `training_params`, `training_config`, `dataset_config`, or `active_params`. All config keys are tested by default. 

## Adding new parameters and tests
As new config file parameters are added, additional test should be put in place. Test the type, and values separately. One assertion per method is ideal, however there may be cases where multiple assertions are necessary. 
