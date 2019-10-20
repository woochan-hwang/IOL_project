# Induction of Labour

## Objective

* predict outcome given the current information
* predict time to delivery given current information

### Prerequisites:
-	Conda

### Environment setup:
Pytorch Cuda is not part of the environment yaml. If the device is gpu enabled, you should install the latest pytorch cuda binaries to enable gpu use.

- Using conda:
```bash
conda env create -f environment.yml
```
- To activate this environment:
```bash
conda activate iol-project
```

### Data
The patient data is excluded from the git. The train.py will load IOL_clean_data.xlsx if it is located in the same folder.

### Preprocessing
All preprocessing is defined in the dataloader script and runs in train.py when create_dataset() method is called.

Each sample in the dataset output will be a dictionary that includes:
'id':{'patient_id':patient_id, 'series':t}; 'metadata'; 'action'; 'time_to_delivery'; 'outcome'.

Each patient will have a series of actions done during the induction process, indicated by 'series'. The action will include
all the actions that has been performed until that time point. For example, if the patient had action A followed by Action B.
This patient will have 2 samples ['seres':1, 'action':[A]], ['seres':2, 'action':[A,B]].

### Contributions
Code written by Woochan H. & Simon R.
Data collection contributed by Bomee K. & Elle S.
