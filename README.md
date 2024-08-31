### GSL-Project

A system to translate Greek Sign Language. This system uses CNN and LSTM as the model architecture.

## Videos Referenced in This Project

The following videos provide context as you work through this documentation. The links to these videos are included later in the document when you will need to view them:

1. **Video explaining the effects of the `params.yaml` file**: This video explains how changes to the `params.yaml` file affect the pipeline and model training.
2. **Video explaining the use of DVC**: This video walks through the process and advantages of using DVC for managing and running the pipeline.
3. **Video explaining the code (Notebooks)**: This video provides an overview of how to navigate and use the Jupyter notebooks included in the project.
4. **Video explaining the code (Pipeline and the rest of the code)**: This video details how the pipeline is built and how the rest of the code integrates to form the complete system.

## Prerequisites

1. **Install Python Version 3.8**: Ensure that Python 3.8 is installed on your system.
2. **Install Git**: Git is required for version control and to clone the repository.
3. **Install Visual Studio Code (VSCode)**: VSCode is recommended for editing and running the code.

## How to Get Started

1. **Fork the Repository**:
   - Fork this repository on GitHub to create your own copy.

2. **Clone the Repository**:
   - From your GitHub account, get the clone URL of the forked repository.
   - Clone the repository to your local machine in a folder of your choice.

   ```bash
   git clone https://github.com/<your-username>/GSL-Project.git
   ```

   - Be sure to replace the clone URL above with the URL from your forked repository.

3. **Open the Project in VSCode**:
   - After cloning, open the folder containing the repository in Visual Studio Code.

## How to Run

### STEP 01 - Create a Virtual Environment After Opening the Repository

1. **Open the Terminal in VSCode**:
   - With VSCode open, access the terminal. The shortcut is usually `Ctrl + (backtick)` or `Ctrl + J` on Windows/Linux or `Cmd + (backtick)` on macOS. (backtick is this character -> `)

2. **Verify Python Installation**:
   - Check to ensure you have Python 3.8 installed globally by running the following command:

   ```bash
   python --version
   ```

3. **Create the Virtual Environment**:
   - Proceed to create a virtual environment in the repository folder:

   ```bash
   python -m venv .venv
   ```

4. **Activate the Virtual Environment**:
   - Activate the virtual environment:

   ```bash
   .\.venv\Scripts\Activate
   ```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 03 - Set Up DagsHub

To ensure the code works correctly, you will need a DagsHub account.

1. **Create an Account on DagsHub**:
   - Visit [DagsHub](https://dagshub.com/) and click the "Start Free" button.
   - Sign up to create an account. I recommend signing up with GitHub, as it links your GitHub account directly, which will be useful later.

2. **Create a New Repository on DagsHub**:
   - Once signed up, create a new repository in DagsHub.
   - Select "**Connect a Repository**" from the available options.
   - Choose to connect from GitHub, which will redirect you to the GitHub website.
   - Select your GitHub account to configure. If you have two-factor authentication (2FA) enabled, authorize DagsHub to access your account.
   - Scroll down to "**Repository Access**", click "**Only select repositories**", and search for the forked repo.
   - Click "**Save**" and continue the process until you're redirected back to DagsHub.

3. **Link to MLflow Tracking**:
   - Back in DagsHub, you should see a copy of your repo as it appears on GitHub.
   - Click the green "**Remote**" button and select the "**Experiments**" tab from the dropdown.
   - There should be a section titled "**Using MLflow Tracking**" with information similar to this:

   ```python
   import dagshub
   dagshub.init(repo_owner='username', repo_name='<repo-name>', mlflow=True)
   ```

   - You should also see a "**Go to MLflow UI**" button. Click it to open a new tab where you can view the experiments of your project. When you run an experiment, it will appear here. Simply refresh the page after running an experiment to see the changes.

4. **Update Configuration**:
   - Keep a note of your username and repo name.
   - In the cloned repo on your laptop, go to `config/config.yaml`, scroll to the bottom, and replace `<username>` with your DagsHub username and `<repo-name>` with your repo name.
   - Go to `setup.py` and replace the information missing below.
   ```python
    REPO_NAME = "<repo-name>"
    AUTHOR_USER_NAME = "<username>"
    SRC_REPO = "gslTranslater"
    AUTHOR_EMAIL = "<email>"
   ```


### STEP 04 - Select the Dataset

You need to choose the dataset you will use for the project. 

- The [*original dataset*](https://drive.google.com/file/d/1I7NNpDR3e1SIQbG1E_E93si1bclrKfuc/view?usp=sharing) is over 20GB and will take longer for the pipeline to download and complete the training.
- For demonstration purposes, a [*smaller subset*](https://drive.google.com/file/d/1-Bf5UCLhM_ahXHy4RDfm5PcMly9lKX7_/view?usp=sharing) of the original dataset, approximately 200MB in size, has been provided.

Depending on your choice, select the appropriate URL below:

1. **Original dataset URL (20GB+)**: 
URL -> https://drive.google.com/file/d/1I7NNpDR3e1SIQbG1E_E93si1bclrKfuc/view?usp=sharing
2. **Smaller dataset URL (~200MB)**: 
URL -> https://drive.google.com/file/d/1-Bf5UCLhM_ahXHy4RDfm5PcMly9lKX7_/view?usp=sharing

Next, go to the `config/config.yaml` file in the cloned repository on your laptop. Replace the placeholder `dataset-URL` in the following section with the selected URL:

```yaml
source_URL: dataset-URL # Replace before execution
```

### STEP 05 - Run the Pipeline

This project is built as a pipeline consisting of four stages:

>1. Data Ingestion
>2. Base Model Preparation
>3. Model Training
>4. Model Evaluation

You can run the pipeline in two ways:
>1. With MLflow tracking
>2. Without MLflow tracking

You also have the option to execute the pipeline stage by stage using Jupyter notebooks or to run the entire pipeline at once using DVC, which will track any changes in the project structure. 

**Please read through both options before deciding to follow one. Regardless of the option you choose, it is recommended to first run the data ingestion notebook, analyze the dataset, and adjust the parameters as and if needed. Once you’ve finalized the parameters, you can proceed with the remaining notebooks or switch to using DVC directly.**

**Note:** DVC will keep track of all files, so if a step has already been completed, it will be skipped whether you choose option 1 or 2.

#### Option 1 - Running the Pipeline Using Notebooks

To view and run the pipeline stage by stage:

1. Navigate to the `research` folder, where the notebooks are labeled in the order they should be executed (ignore `trials.ipynb`).
   
2. If you choose to run the pipeline **without MLflow tracking**, open the notebook `research/04_model_evaluation_with_mlflow.ipynb` and comment out the following line in the last cell:

   ```python
   model_evaluation.log_into_mlflow(avg_loss, avg_accuracy)
   ```

3. If you want to use MLflow tracking, you can skip the above step.

4. To run the pipeline:
   - Open the notebooks in order, starting with the data ingestion notebook.
   - Select the virtual environment kernel.
   - Run all the cells in the first notebook (you will run the other notebooks later) and review the output in the final cell.

5. Depending on the dataset you select, you may need to modify the `params.yaml` file in the project root directory. To make an informed decision, first review the contents of the newly created `artifacts/data_ingestion/GSL_Analysis` folder. Pay particular attention to the `dataset_analysis.txt` file and the plots in the `plot_images` directory. Watch this video [link to video] to understand how the parameters affect the rest of the project.

6. Once you’ve analyzed the data, update the `params.yaml` file based on your dataset choice:

   **Case 1: 20GB+ Dataset Selected**

   ```yaml
   AUGMENTATION: True
   IMAGE_SIZE: [144, 192, 3]
   BATCH_SIZE: 4
   INCLUDE_TOP: False
   EPOCHS: 1
   CLASSES: 250
   WEIGHTS: imagenet
   LEARNING_RATE: 0.001
   MAX_SEQ_LENGTH: 130
   NUM_UNIQUE_WORDS: 250

   # Data Ingestion Parameters
   MAX_INSTANCES_PER_CLASS: 10
   TRAIN_SPLIT: 0.7
   TEST_SPLIT: 0.2
   VALIDATE_SPLIT: 0.1
   ```

   **Case 2: ~200MB Dataset Selected**

   ```yaml
   AUGMENTATION: True
   IMAGE_SIZE: [144, 192, 3]
   BATCH_SIZE: 4
   INCLUDE_TOP: False
   EPOCHS: 1
   CLASSES: 25
   WEIGHTS: imagenet
   LEARNING_RATE: 0.001
   MAX_SEQ_LENGTH: 130
   NUM_UNIQUE_WORDS: 25

   # Data Ingestion Parameters
   MAX_INSTANCES_PER_CLASS: 10
   TRAIN_SPLIT: 0.7
   TEST_SPLIT: 0.2
   VALIDATE_SPLIT: 0.1
   ```

7. **If the parameters need changing after running the first notebook, rerun it after making the changes.** Once you’re satisfied with the parameters, you can proceed by running the second, third, and fourth notebooks. Alternatively, you can skip directly to running the pipeline with DVC. Note that only the first notebook is mandatory before proceeding with DVC.

#### Option 2 - Running the Pipeline with DVC

To see what happens when DVC is used, watch this video [link to video]

To run the pipeline with DVC:

1. If you choose to run **without MLflow tracking**, go to `src/gslTranslater/pipeline/stage_04_model_evaluation.py` and comment out the following line:

   ```python
   model_evaluation.log_into_mlflow(avg_loss, avg_accuracy)
   ```

2. To run DVC **with MLflow tracking**, ignore the step above.

3. Initialize DVC for your project by running:

   ```bash
   dvc init
   ```

4. Start the pipeline by running:

   ```bash
   dvc repro
   ```

This command will begin the pipeline and complete all stages without requiring further input.

### STEP 06 - View Experiment Results (Optional)

After running the pipeline, you can return to the MLflow page opened earlier in Step 3. Refresh the page to view the details of your experiment, including metrics, parameters, and any logged artifacts.

### STEP 07 - Running Different Experiments (Optional)

To conduct different experiments:

1. Modify the values in the `params.yaml` file as needed.
   
2. Run the pipeline again using the same method you chose previously (either via notebooks or DVC).

**Note:** Be cautious when changing parameters related to the dataset, as doing so will trigger the entire pipeline to run again, including the dataset feature extraction process.

## Development Workflow

This section regards the development process and can be ignored if your goal is not reproducing the code from scratch.

1. Update config.yaml
2. Update secrets.yaml [Optional] - We keep secrets and keys here if needed
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml

To understand these steps, watch these videos explaining the code structure.
 - [Video explaining the code (Notebooks)] [link to video]
 - [Video explaining the code (Pipeline and the rest of the code)] [link to video]
 
Follow these steps to ensure the project is properly configured and updated.

1. **Update `config.yaml`**: Ensure all configuration settings are accurate and up to date.

2. **Update `secrets.yaml` [Optional]**: If your project uses secrets or keys, update this file with any new or modified credentials.

3. **Update `params.yaml`**: Modify the parameters as needed for your specific experiment or use case.

4. **Update the Entity**: Make sure that any entity definitions or changes are reflected in your codebase.

5. **Update the Configuration Manager in `src/config`**: Verify that the configuration manager is properly set up to handle any changes made to configuration files.

6. **Update the Components**: If there are changes in any modules or functions, ensure these are reflected across the project components.

7. **Update the Pipeline**: Adjust the pipeline code to accommodate any changes in the data flow or processing logic.

8. **Update `main.py`**: Ensure the main script reflects any changes to the overall execution flow or logic.

9. **Update `dvc.yaml`**: Modify the DVC pipeline configuration file to track any changes in data dependencies or processing stages.
