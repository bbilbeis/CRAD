"""
Federated Learning Client Abstraction.

Represents an individual participant in the decentralized network. 
Handles the local data state and provides a standardized interface for 
the orchestrator to trigger local training and evaluation.
"""
import copy
import numpy as np
import torch


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data,
                 local_sample_number, args, device, model_trainer, logger,
                 arch_name: str = "resnet18", local_val_data=None):
        self.logger               = logger
        self.client_idx           = client_idx
        self.local_training_data  = local_training_data
        self.local_test_data      = local_test_data
        self.local_val_data       = local_val_data if local_val_data is not None else local_test_data
        self.local_sample_number  = local_sample_number
        self.args                 = args

        # Assign device round-robin across available GPUs if multigpu is enabled (only when not using Ray)
        if getattr(args, "multigpu", False) and torch.cuda.is_available() and not getattr(args, "use_ray", False):
            num_gpus = torch.cuda.device_count()
            self.device = torch.device(f"cuda:{client_idx % num_gpus}")
        else:
            self.device               = device
        self.model_trainer        = model_trainer
        self.arch_name            = arch_name

        # Tell the trainer which arch it owns
        self.model_trainer._arch_name = arch_name

        self.logger.info(
            f"Client {client_idx} | arch={arch_name} | "
            f"samples={local_sample_number}"
        )

    def update_local_dataset(self, client_idx, local_training_data,
                             local_test_data, local_sample_number, local_val_data=None):
        self.client_idx          = client_idx
        self.local_training_data = local_training_data
        self.local_test_data     = local_test_data
        self.local_val_data      = local_val_data if local_val_data is not None else local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_local, neighbor_info, round, global_test_data=None, class_weights=None):
        """
        Entry point for the client's local training phase.

        - Decentralized FedAvg: `neighbor_info` is None. `w_local` is the pre-aggregated 
          neighborhood model.
        - Decentralized KD (TypicalKD): `w_local` is the client's own model. `neighbor_info` 
          contains neighbor teacher models.

        Args:
            w_local       : The initial state_dict for this round's training.
            neighbor_info : List of (state_dict, arch_name) tuples for KD, or None for FedAvg.
            round         : Current communication round index.
            global_test_data : Optional centralized dataset for isolated testing.
            class_weights : Class weight vector [C] for curriculum distillation.
        """
        self.model_trainer.set_model_params(w_local)
        self.model_trainer.set_id(self.client_idx)
        
        # ALL client validation statistics during distillation are computed strictly on local_val_data
        val_data_to_use = global_test_data if global_test_data is not None else self.local_val_data

        val_metrics = self.model_trainer.train(
            self.local_training_data,
            val_data_to_use,
            self.device,
            self.args,
            round,
            neighbor_info,
            class_weights=class_weights,
        )
        weights = self.model_trainer.get_model_params()
        return weights, val_metrics

    def local_val(self, w):
        """Compute validation statistics strictly using local_val_data."""
        self.model_trainer.set_model_params(w)
        return self.model_trainer.test(self.local_val_data, self.device, self.args)

    def local_test(self, w, b_use_test_dataset=True):
        """Compute final evaluation metrics strictly using local_test_data."""
        test_data = self.local_test_data if b_use_test_dataset else self.local_training_data
        self.model_trainer.set_model_params(w)
        return self.model_trainer.test(test_data, self.device, self.args)

