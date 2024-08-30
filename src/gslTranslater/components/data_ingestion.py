import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import tarfile
import gdown
import pandas as pd
import re
from tqdm import tqdm
from scipy.stats import mode
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
from gslTranslater.constants import *
from gslTranslater.utils.common import read_yaml, create_directories, get_size, save_json
from gslTranslater import logger
from gslTranslater.entity.config_entity import (DataIngestionConfig)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        # Ensure the analysis and plot directories exist
        os.makedirs(self.config.analysis_dir, exist_ok=True)
        os.makedirs(self.config.plot_dir, exist_ok=True)

    def download_file(self):
        if os.path.exists(self.config.unzip_dir):
            logger.info(f"Data already exists at {self.config.unzip_dir}, skipping download and extraction.")
            return
        try:
            dataset_url = self.config.source_URL
            tar_download_dir = self.config.local_data_file
            os.makedirs(self.config.root_dir, exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {tar_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, tar_download_dir, quiet=False)

            logger.info(f"Downloaded data from {dataset_url} into file {tar_download_dir}")

        except Exception as e:
            raise e

    def extract_tar_file(self):
        if os.path.exists(self.config.unzip_dir):
            logger.info(f"Data already extracted to {self.config.unzip_dir}, skipping extraction.")
            return
        extract_path = self.config.unzip_dir
        os.makedirs(extract_path, exist_ok=True)
        with tarfile.open(self.config.local_data_file, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_path)

    def merge_and_clean_csv_files(self):
        # List all extracted_annotation CSV files
        csv_files = [f for f in os.listdir(self.config.data_dir) if f.startswith('extracted_annotations_')]
        merged_df = pd.DataFrame()

        def clean_text(text):
            if isinstance(text, str):
                # Replace pipe symbols with commas
                text = text.replace('|', ',')
                # Remove any content within parentheses, including the parentheses
                text = re.sub(r'\([^)]*\)', '', text)
                # Remove semicolons
                text = text.replace(';', '')
            return text

        for csv_file in csv_files:
            # Load the CSV file
            df = pd.read_csv(os.path.join(self.config.data_dir, csv_file), header=None, names=['Raw'])

            # Clean the 'Raw' text
            df['Cleaned'] = df['Raw'].apply(clean_text)

            # Split the cleaned text into two columns: Path and Gloss
            split_df = df['Cleaned'].str.split(',', expand=True)
            
            if split_df.shape[1] == 2:
                split_df.columns = ['Path', 'Gloss']
            else:
                logger.error(f"Unexpected format in file {csv_file}. Skipping this file.")
                continue

            # Remove any leading/trailing whitespace in 'Path' and 'Gloss'
            split_df['Path'] = split_df['Path'].str.strip()
            split_df['Gloss'] = split_df['Gloss'].str.strip()

            # Concatenate the split_df into the merged_df
            merged_df = pd.concat([merged_df, split_df], ignore_index=True)

        if merged_df.empty:
            logger.error("Merged DataFrame is empty. No valid data was found in the CSV files.")
            raise ValueError("Merged DataFrame is empty. Ensure that the extracted_annotation CSV files contain valid data.")

        # Save the merged and cleaned CSV with appropriate headers
        merged_df.to_csv(self.config.merged_csv, index=False, encoding='utf-8')
        logger.info(f"Merged and cleaned annotations saved to {self.config.merged_csv}")

        return merged_df

    def check_image_paths(self, merged_df):
        confirmed_rows, missing_rows = [], []

        for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
            path = row['Path'].strip()
            image_dir = os.path.join(self.config.data_dir, path)

            if os.path.exists(image_dir):
                images = [img for img in os.listdir(image_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(images) > 0:
                    confirmed_rows.append(row)
                else:
                    missing_rows.append(row)
            else:
                missing_rows.append(row)

        confirmed_df = pd.DataFrame(confirmed_rows, columns=merged_df.columns)
        missing_df = pd.DataFrame(missing_rows, columns=merged_df.columns)

        # Save the confirmed and missing CSVs
        confirmed_df.to_csv(self.config.confirmed_csv, index=False)
        missing_df.to_csv(self.config.missing_csv, index=False)
        logger.info(f"Confirmed annotations saved to {self.config.confirmed_csv}")
        logger.info(f"Missing annotations saved to {self.config.missing_csv}")

        return confirmed_df

    def analyze_raw_gloss_distribution(self, merged_df):
        gloss_counts = Counter(merged_df['Gloss'])
        gloss_distribution_df = pd.DataFrame(list(gloss_counts.items()), columns=['Gloss', 'Count'])
        raw_gloss_distribution_plot = self.config.plot_dir / 'raw_gloss_distribution_plot.png'

        # Plotting the gloss distribution
        sorted_df = gloss_distribution_df.sort_values(by='Count', ascending=False)
        plt.figure(figsize=(12, 16))
        plt.barh(sorted_df['Gloss'], sorted_df['Count'], color='skyblue')
        plt.gca().invert_yaxis()
        for index, value in enumerate(sorted_df['Count']):
            plt.text(value, index, f'{sorted_df["Gloss"].iloc[index]} ({value})', va='center')
        plt.xlabel('Count')
        plt.ylabel('Glosses')
        plt.title('Raw Gloss Count Distribution (Highest to Lowest)')

        # Save the plot instead of displaying it
        plt.savefig(raw_gloss_distribution_plot, bbox_inches='tight')
        logger.info(f"Raw gloss distribution plot saved to {raw_gloss_distribution_plot}")
        plt.close()

    def analyze_frames(self, confirmed_df):
        confirmed_df['Frame_Count'] = confirmed_df['Path'].apply(self.count_frames)
        confirmed_df.to_csv(self.config.frame_count_csv, index=False, encoding='utf-8')
        logger.info(f"CSV with frame counts saved to {self.config.frame_count_csv}")
        
        analysis_data = self.calculate_analysis(confirmed_df)
        with open(self.config.analysis_txt, 'w', encoding='utf-8') as f:
            f.write(analysis_data)
        logger.info(f"Dataset analysis saved to {self.config.analysis_txt}")

    
    def count_frames(self, path):
        image_dir = os.path.join(self.config.data_dir, path)
        frames = [img for img in os.listdir(image_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        return len(frames)
    
    def calculate_analysis(self, df):
        frame_counts = df['Frame_Count']
        glosses = df['Gloss']

        # Calculate average image dimensions
        dimensions = df['Path'].apply(self.get_image_dimensions)
        avg_width = np.mean([d[0] for d in dimensions])
        avg_height = np.mean([d[1] for d in dimensions])
        
        # Calculate mode safely
        mode_result = mode(frame_counts)
        mode_value = mode_result.mode[0] if mode_result.count[0] > 1 else 'No mode'

        analysis_data = {
            'Max Frames': np.max(frame_counts),
            'Min Frames': np.min(frame_counts),
            'Average Frames': np.mean(frame_counts),
            'Median Frames': np.median(frame_counts),
            'Mode Frames': mode_value,
            'Gloss with Max Frames': glosses[frame_counts.idxmax()],
            'Gloss with Min Frames': glosses[frame_counts.idxmin()],
            'Unique Glosses': df['Gloss'].nunique(),
            'Total Instances': len(df),
            'Average Image Width': avg_width,
            'Average Image Height': avg_height
        }
        
        analysis_txt = (
            f"Max Frame Count: {analysis_data['Max Frames']}\n"
            f"Gloss with Max Frame Count: {analysis_data['Gloss with Max Frames']}\n"
            f"Min Frame Count: {analysis_data['Min Frames']}\n"
            f"Gloss with Min Frame Count: {analysis_data['Gloss with Min Frames']}\n"
            f"Average Frame Count: {analysis_data['Average Frames']:.2f}\n"
            f"Median Frame Count: {analysis_data['Median Frames']}\n"
            f"Mode Frame Count: {analysis_data['Mode Frames']}\n"
            f"Number of Unique Glosses: {analysis_data['Unique Glosses']}\n"
            f"Total Number of Instances: {analysis_data['Total Instances']}\n"
            f"Average Image Width: {analysis_data['Average Image Width']:.2f}\n"
            f"Average Image Height: {analysis_data['Average Image Height']:.2f}\n"
        )
        
        return analysis_txt

    def get_image_dimensions(self, path):
        image_dir = os.path.join(self.config.data_dir, path)
        first_image = os.listdir(image_dir)[0]
        image_path = os.path.join(image_dir, first_image)
        image = Image.open(image_path)
        return image.size  # (width, height)

    def analyze_gloss_distribution(self, confirmed_df):
        gloss_counts = Counter(confirmed_df['Gloss'])
        gloss_distribution_df = pd.DataFrame(list(gloss_counts.items()), columns=['Gloss', 'Count'])
        gloss_distribution_df.to_csv(self.config.gloss_distribution_csv, index=False)
        logger.info(f"Gloss distribution saved to {self.config.gloss_distribution_csv}")

        # Plotting the gloss distribution
        sorted_df = gloss_distribution_df.sort_values(by='Count', ascending=False)
        plt.figure(figsize=(12, 16))
        plt.barh(sorted_df['Gloss'], sorted_df['Count'], color='skyblue')
        plt.gca().invert_yaxis()
        for index, value in enumerate(sorted_df['Count']):
            plt.text(value, index, f'{sorted_df["Gloss"].iloc[index]} ({value})', va='center')
        plt.xlabel('Count')
        plt.ylabel('Glosses')
        plt.title('Gloss Count Distribution (Highest to Lowest)')

        # Save the plot instead of displaying it
        plt.savefig(self.config.gloss_distribution_plot, bbox_inches='tight')
        logger.info(f"Gloss distribution plot saved to {self.config.gloss_distribution_plot}")
        plt.close()

    def create_balanced_dataset(self, confirmed_df):
        balanced_df = pd.DataFrame()
        summary_data = []

        for gloss, group in confirmed_df.groupby('Gloss'):
            count = len(group)
            if count > self.config.max_instances_per_class:
                # Trim the group to the max_instances_per_class
                selected_group = group.sample(n=self.config.max_instances_per_class, random_state=42)
            else:
                selected_group = group
            
            balanced_df = pd.concat([balanced_df, selected_group], ignore_index=True)
            summary_data.append({"Gloss": gloss, "Count": len(selected_group)})

        if balanced_df.empty:
            logger.error("Balanced DataFrame is empty after trimming. Please check the parameters and the dataset.")
            raise ValueError("Balanced DataFrame is empty after trimming. Please check the parameters and the dataset.")

        summary_df = pd.DataFrame(summary_data)
        balanced_df.to_csv(self.config.balanced_csv, index=False, encoding='utf-8')
        summary_df.to_csv(self.config.summary_csv, index=False, encoding='utf-8')
        logger.info(f"Balanced dataset saved to {self.config.balanced_csv}")

        return balanced_df


    def split_data(self, balanced_df):
        if balanced_df.empty:
            logger.error("Balanced DataFrame is empty. Cannot proceed with data splitting.")
            raise ValueError("Balanced DataFrame is empty. Ensure the dataset is properly balanced before splitting.")

        total_instances = self.config.max_instances_per_class
        train_instances = int(self.config.train_split * total_instances)
        test_instances = int(self.config.test_split * total_instances)
        validate_instances = total_instances - train_instances - test_instances

        train_rows, test_rows, val_rows = [], [], []

        for i in range(0, len(balanced_df), total_instances):
            group = balanced_df.iloc[i:i+total_instances]
            if len(group) == total_instances:
                train_rows.append(group.iloc[:train_instances])
                test_rows.append(group.iloc[train_instances:train_instances+test_instances])
                val_rows.append(group.iloc[train_instances+test_instances:train_instances+test_instances+validate_instances])

        if not train_rows or not test_rows or not val_rows:
            logger.error("No valid groups found for splitting. Check the balance and size of the dataset.")
            raise ValueError("No valid groups found for splitting. Check the balance and size of the dataset.")

        train_df = pd.concat(train_rows, ignore_index=True)
        test_df = pd.concat(test_rows, ignore_index=True)
        val_df = pd.concat(val_rows, ignore_index=True)

        train_df.to_csv(self.config.train_csv, index=False)
        test_df.to_csv(self.config.test_csv, index=False)
        val_df.to_csv(self.config.validate_csv, index=False)

        logger.info(f"Training set saved to {self.config.train_csv}")
        logger.info(f"Testing set saved to {self.config.test_csv}")
        logger.info(f"Validation set saved to {self.config.validate_csv}")
