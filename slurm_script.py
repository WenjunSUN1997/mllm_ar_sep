from train import train
import submitit

def task():
    config = {'task': 'doc_ner',
              'type': 'unmasked',
              'device': 'cuda:0',
              'batch_size': 2,
              'max_token_num': 1024,
              'half': True,
              'lr': 0.0001,
              'weight': True,
              'sim_dim': 4096,
              'model_name': 'MAGAer13/mplug-owl-llama-7b',
              'dataset_name': 'nielsr/funsd-layoutlmv3'}
    train(config)

if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder='/Utilisateurs/wsun01/logs/')  # Can specify cluster='debug' or 'local' to run on the current node instead of on the cluster
    executor.update_parameters(
        timeout_min=120,
        nodes=1,
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=1,
        mem_gb=80,
        slurm_partition='general',
        slurm_additional_parameters={
            'nodelist': 'l3icalcul07'
        })
    executor.submit(task)