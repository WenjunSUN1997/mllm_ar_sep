# import submitit
# from utils.datasetor_ner_doc import test
#
# def task():
#     test()
#
# executor = submitit.AutoExecutor(folder='/Utilisateurs/wsun01/logs') # Can specify cluster='debug' or 'local' to run on the current node instead of on the cluster
# executor.update_parameters(
#   # timeout_min=120, # Time out in minutes
#   # nodes=1,
#   # gpus_per_node=1,
#   # tasks_per_node=1,
#   # cpus_per_task=6,
#   # mem_gb=16,
#   slurm_partition='general',
#   slurm_additional_parameters={
#       'gres': 'gpu:4', # Specify the gpus to use
#       # 'mail-user': 'first.last@univ-lr.fr',
#       # 'mail-type': 'FAIL,REQUEUE',
#       'nodelist': 'l3icalcul07'
#       # Please refer to the SLURM documentation for additional parameters
#   })
# executor.submit(task)