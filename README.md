# Koyeb Serverless GPUs: Fine-Tune MistralAI Model on Financial Data

## Introduction

[MistralAI](https://mistral.ai/) is an advanced language model designed for tasks like text generation, sentiment analysis, translation, summarization, and more.

By default, MistralAI is trained on general language data. It **performs even better when fine-tuned to specific domains** like finance, law, or medicine.

Fine-tuning retrains the model on domain-specific data, enabling it to understand the specific terms, patterns, and concepts used in that field. For instance, in finance, supplementary information for retraining a model includes financial reports, stock market data, or legal documents.

With fine-tuning, MistralAI becomes more accurate at understanding complex financial terms, market trends, and regulatory requirements. This enhancement makes the model more adept at predicting financial outcomes, generating insightful analysis, and supporting decision-making in areas like trading or risk management.

## Requirements

To successfully complete this tutorial, you will need the following:

- **GitHub Account**: Needed for managing the fine-tuning code. Sign up at [GitHub](https://github.com/signup) if you don’t have one.
- **Koyeb Account**: Required for accessing Koyeb’s cloud infrastructure, including GPU resources. Create an account at [Koyeb](https://app.koyeb.com/signup) if you don’t have one.
- **Koyeb GPU Access**: Make sure your Koyeb account has access to GPU instances for fine-tuning and to deploy GPU-enabled instances through the Koyeb dashboard.
- **Basic Knowledge**: Familiarity with **Python** (running scripts, setting up virtual environments). Basic understanding of **Docker**.
- **[NewsAPI.org](http://NewsAPI.org) API Key**: Access to a NewsAPI API Key. This is needed to retrieve content for the financial data set.
- **MistralAI**: Access to a Mistral AI API key. This will be needed to prepare the financial data set with AI.

## Steps

This tutorial is divided into the following steps:

- [Koyeb Serverless GPUs: Fine-Tune MistralAI Model on Financial Data](#koyeb-serverless-gpus-fine-tune-mistralai-model-on-financial-data)
  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Steps](#steps)
  - [Cloning and Exploring the GitHub Repository](#cloning-and-exploring-the-github-repository)
  - [Understanding the Fine-Tuning Workflow](#understanding-the-fine-tuning-workflow)
    - [1. Prepare the Dataset](#1-prepare-the-dataset)
    - [2. Prepare Training and Evaluation Datasets](#2-prepare-training-and-evaluation-datasets)
    - [3. Configure the Training Script](#3-configure-the-training-script)
    - [4. Verify the Dataset (Training + Evaluation)](#4-verify-the-dataset-training--evaluation)
    - [5. Train the Model](#5-train-the-model)
  - [Preparing the Financial Dataset](#preparing-the-financial-dataset)
  - [Preparing Training and Evaluation Datasets](#preparing-training-and-evaluation-datasets)
  - [Configure the Training Script](#configure-the-training-script)
  - [Deploying to Koyeb GPU](#deploying-to-koyeb-gpu)
    - [Create a Dockerfile](#create-a-dockerfile)
    - [Create the repository](#create-the-repository)
    - [Deploy to Koyeb](#deploy-to-koyeb)
  - [Running the Fine-Tuning process](#running-the-fine-tuning-process)
  - [Evaluating the Fine-Tuned Model](#evaluating-the-fine-tuned-model)
    - [Impact on Domain Knowledge](#impact-on-domain-knowledge)
  - [Conclusion](#conclusion)

## Cloning and Exploring the GitHub Repository

To start fine-tuning MistralAI, the first step is to clone the official GitHub repository, which has all the necessary scripts and settings for training.

Clone the repository with the following command:

```bash
git clone https://github.com/mistralai/mistral-finetune.git
```

**Exploring the Key Files and Folders**

After cloning the repository, take a moment to look around its structure. Understanding these files will help you customize and run the fine-tuning process effectively.

The file `example/7B.yaml` is particularly important:

- This configuration file defines the model architecture and settings such as batch size, and number of training cycles.
- It is crucial for setting up the training environment and should be reviewed and adjusted based on your specific needs, especially if you are fine-tuning for a specialized area like finance.

Other important files are:

- `validate_data.py`:
  - This script is used to validate the dataset before training. It ensures that the data is complete, correctly formatted, and free of errors that could impact the training process.
  - Running this script helps identify and resolve any issues with the dataset, ensuring smooth training.
- `reformat_data.py`:
  - This script is used to reformat the dataset if necessary. It ensures that the data is in the correct format required by the model for training.
  - This step is important to maintain consistency and accuracy in the dataset, which is essential for effective fine-tuning.

Understanding and properly configuring the `7B.yaml` file is essential for effective fine-tuning of the MistralAI model. We will see later on the necessary settings for our fine-tune process for the Mistral 7B model.

## Understanding the Fine-Tuning Workflow

Fine-tuning a language model like MistralAI involves a systematic workflow to ensure the model is properly adapted to a specific domain or task.

### 1. Prepare the Dataset

- **Gather the Data (Content)**: Collect domain-specific data relevant to the task you are fine-tuning the model for. This could include financial reports, market data, customer reviews, or any other textual data that reflects the language and concepts you want the model to learn. Ensure the data is comprehensive and diverse enough to cover different scenarios and contexts within the domain.
- **Proper Formatting**: Format the collected data into a structure that the model can process. This typically involves organizing the text into a sequence of interactions (e.g., question-answer pairs) or continuous text segments. Ensure consistency in formatting across all data samples to avoid confusion during training. This might include tokenization, lowercasing, and handling special characters or symbols.

### 2. Prepare Training and Evaluation Datasets

- **Splitting the Original Dataset**: Divide the prepared dataset into two parts: the training set and the evaluation (or validation) set. The training set is used to teach the model, while the evaluation set is used to monitor performance and avoid overfitting. A common split ratio is 80/20 or 90/10, but this can be adjusted based on the size of your dataset and the specific requirements of your task. Ensure that both sets are representative of the full dataset, covering all relevant aspects of the domain.
- **Balancing the Datasets**: Check for balance in the training and evaluation datasets. For example, if the data includes different categories (e.g., different financial instruments or market conditions), ensure that each category is well-represented in both the training and evaluation sets. This step is crucial to avoid bias in the model's predictions.

### 3. Configure the Training Script

- **Set Batch Size**: Batch size determines how many samples are processed before the model's weights are updated. Larger batch sizes can make training faster but require more memory, while smaller batches can lead to better generalization but might make training slower. Experiment with different batch sizes to find the optimal setting for your hardware and data.
- **Define Training Steps**: Specify the number of training steps or epochs. This determines how many times the model will iterate over the entire training dataset. Monitor the model's performance during training to decide whether more or fewer steps are needed.

### 4. Verify the Dataset (Training + Evaluation)

- **Check Data Integrity**: Verify that the data in both training and evaluation sets is complete and correctly formatted. Look for missing values, corrupted files, or inconsistencies that could impact training. Run preliminary checks to ensure that the data loads correctly into the training pipeline.
- **Validate Data Distribution**: Confirm that the distribution of data in the training and evaluation sets aligns with the expected real-world distribution. This is especially important in domains like finance, where different market conditions need to be represented.

### 5. Train the Model

- **Initiate Training**: Begin the fine-tuning process by running the configured training script.
- **Monitor Performance**: Regularly evaluate the model's performance on the validation set during training.
- **Save Checkpoints**: Save model checkpoints at regular intervals to preserve the model’s state at different points in training.

## Preparing the Financial Dataset

To prepare the dataset for training, it needs to be structured in a specific format that the MistralAI fine-tune process can understand. The format typically follows a structure similar to this:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "User interaction n°1 contained in document n°1"
    },
    {
      "role": "assistant",
      "content": "Bot interaction n°1 contained in document n°1"
    },
    {
      "role": "user",
      "content": "User interaction n°2 contained in document n°1"
    },
    {
      "role": "assistant",
      "content": "Bot interaction n°2 contained in document n°1"
    }
  ]
}
{
  "messages": [
    {
      "role": "user",
      "content": "User interaction n°1 contained in document n°2"
    },
    {
      "role": "assistant",
      "content": "Bot interaction n°1 contained in document n°2"
    },
    {
      "role": "user",
      "content": "User interaction n°2 contained in document n°2"
    },
    {
      "role": "assistant",
      "content": "Bot interaction n°2 contained in document n°2",
      "weight": 0,  # don't train on n°2
    },
    {
      "role": "user",
      "content": "User interaction n°3 contained in document n°2"
    },
    {
      "role": "assistant",
      "content": "Bot interaction n°3 contained in document n°2"
    }
  ]
}
```

Each JSON object contains a list of messages, with each message having a `role` field to indicate the speaker (either "user" or "assistant") and a `content` field to store the text of the interaction.

This file type is called a JSONL (JSON lines files), because it contains several JSON objects separated by a newline.

## Preparing Training and Evaluation Datasets

To gather the data (content), we will use an API from [NewsAPI.org](http://newsapi.org/) to get financial news content. You will need to register to get a free API key from [NewsAPI.org](http://newsapi.org/) if you don’t have one.

You will also need access to an API key from MistralAI. You can register for one [here](https://auth.mistral.ai/ui/registration) if you don’t have one.

Then you can write the `dataset_processing.py` script.

````python
import json
import requests
import pandas as pd
import os
from decouple import config
from mistralai import Mistral

NEWS_API_KEY = config("NEWS_API_KEY")
MISTRAL_API_KEY = config("MISTRAL_API_KEY")

# Function to fetch financial news data related to a specific topic
def fetch_financial_news(api_key, query="financial market", page_size=100):
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={page_size}&apiKey={api_key}"
    response = requests.get(url)
    news_data = response.json()
    articles = news_data.get('articles', [])
    return articles

# Function to process and save the data to a CSV file
def save_news_to_csv(articles, topic):
    # Create a DataFrame from the articles
    data = {
        "source": [article['source']['name'] for article in articles],
        "author": [article['author'] for article in articles],
        "title": [article['title'] for article in articles],
        "description": [article['description'] for article in articles],
        "url": [article['url'] for article in articles],
        "publishedAt": [article['publishedAt'] for article in articles],
        "content": [article['content'] for article in articles],
    }
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    output_file = os.path.join(f"{topic.replace(' ', '_')}_news.csv")
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

def process_csv_to_jsonl(input_file):
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        # Process each row and append each content to a list
        json_list = []
        for index, row in df.iterrows():
            client = Mistral(api_key=MISTRAL_API_KEY)
            result = client.chat.complete(model="mistral-small-latest", messages=[
                {
                    "content": """You will receive news article. Analyze it and generated user and assistant
                                interactions in a chat like format. The output should be a JSON with the
                                following format:
                               '{
                                  "messages": [
                                    {
                                      "role": "user",
                                      "content": "User interaction n°1 contained in document n°1"
                                    },
                                    {
                                      "role": "assistant",
                                      "content": "Bot interaction n°1 contained in document n°1"
                                    },
                                    {
                                      "role": "user",
                                      "content": "User interaction n°2 contained in document n°1"
                                    },
                                    {
                                      "role": "assistant",
                                      "content": "Bot interaction n°2 contained in document n°1"
                                    }
                                  ]
                                }'
                               Return only the JSON and nothing else, and no JSON tags.
                               """
                    ,
                    "role": "system",
                },
                {
                    "content": f"Here is the news content: {row['content']}",
                    "role": "user",
                },
            ])
            result_txt = result.choices[0].message.content
            result_txt = result_txt.replace("```json", "").replace("```", "")
            json_content = json.loads(result_txt)
            print(json_content)
            json_list.append(json_content)
        # Save the list to a JSONL file
        output_file = input_file.replace(".csv", ".jsonl")
        with open(output_file, "w") as f:
            for item in json_list:
                f.write(json.dumps(item) + "\n")
    except Exception as e:
        print(e)
        return "", ""

def separate_jsonl_train_eval(input_file):
    try:
        # Read the JSONL file and separate into training and evaluation sets using pandas
        df = pd.read_json(input_file, lines=True)
        df_train = df.sample(frac=0.90, random_state=200)
        df_eval = df.drop(df_train.index)

        # Save the training and evaluation sets to JSONL files keeping the same file name
        train_output_file = input_file.replace(".jsonl", "_train.jsonl")
        eval_output_file = input_file.replace(".jsonl", "_eval.jsonl")
        df_train.to_json(train_output_file, orient="records", lines=True)
        df_eval.to_json(eval_output_file, orient="records", lines=True)
    except Exception as e:
        print(e)
        return "", ""

# Main function to run the data processing pipeline
def main():
    # Specify the topic you want to search for
    topic = "financial market"

    # Fetch data
    articles = fetch_financial_news(NEWS_API_KEY, query=topic)

    # Save the data to a CSV file
    save_news_to_csv(articles, topic)

    # Process the CSV file to generate a JSONL file
    process_csv_to_jsonl("financial_market_news.csv")

    # Separate the JSONL file into training and evaluation sets
    separate_jsonl_train_eval("financial_market_news.jsonl")

# Entry point of the script
if __name__ == "__main__":
    main()

````

This Python script automates the process of collecting financial news data, formatting it for fine-tuning a MistralAI language model, and preparing it into training and evaluation datasets:

- **`fetch_financial_news`**: This function fetches financial news articles from NewsAPI based on a specified topic.
- **`save_news_to_csv`**: This function saves the fetched news data into a CSV file.
- **`process_csv_to_jsonl`**: This function converts the CSV data into a JSONL format, generating chat-based interactions using MistralAI.
- **`separate_jsonl_train_eval`**: This function splits the JSONL data into training and evaluation datasets.

The main function orchestrates the entire workflow:

- Fetching news data.
- Saving the data to a CSV file.
- Converting the CSV data to JSONL format.
- Splitting the JSONL data into training and evaluation datasets.

This script will generate the following dataset files, which will be used to train the model (you will run this script later on the remote machine):

- **`financial_market_news_train.jsonl`**: Contains the training data, in this case, 90 records of questions and answers related to news source data.
- **`financial_market_news_eval.jsonl`**: Contains the evaluation data, in this case, 10 records of questions and answers related to news source data.

## Configure the Training Script

Before we deploy to the Koyeb CPU to validate the dataset and train the model, you can start preparing the training configuration file. This file, which is a YAML file, will include all the necessary settings for the training process, as mentioned earlier.

So, go ahead and create a `7B.yaml` file:

```yaml
# data
data:
  instruct_data: "/mistral-finetune/financial_market_news_train.jsonl" # Fill
  data: "" # Optionally fill with pretraining data
  eval_instruct_data: "/mistral-finetune/financial_market_news_eval.jsonl" # Optionally fill

# model
model_id_or_path: "/mistral-finetune/mistral_models/" # Change to downloaded path
lora:
  rank: 64

# optim
seq_len: 32768
batch_size: 1
max_steps: 300
optim:
  lr: 6.e-5
  weight_decay: 0.1
  pct_start: 0.05

# other
seed: 0
log_freq: 1
eval_freq: 100
no_eval: False
ckpt_freq: 100

save_adapters: True # save only trained LoRA adapters. Set to `False` to merge LoRA adapter into the base model and save full fine-tuned model

run_dir: "/mistral-finetune/chat_test" # Fill
```

This is the important information that you need to fill in:

- **`instruct_data`**: This is the path to the training dataset. This dataset will be generated when you run the **`dataset_processing.py`** script on the remote machine.
- **`eval_instruct_data`**: This is the path to the evaluation dataset. This dataset will also be generated when you run the **`dataset_processing.py`** script on the remote machine.
- **`model_id_or_path`**: This is the identifier or path of the model you will be training. You will download this model later on the remote machine.
- **`batch_size`**: You can adjust this if needed, but a batch size of 1 will work well for this case.
- **`max_steps`**: This is the number of steps to train the model with. The default of 300 steps provides a good balance between speed and training capabilities. You can reduce to 100 steps for faster processing at a cost of less accuracy.
- **`run_dir`**: This is the directory where the trained model will be saved.

After deployment, you will need to download the model to train, execute the dataset script, and then train the model. These settings are prepared for the commands you will execute later on.

## Deploying to Koyeb GPU

To deploy the fine-tuning process to Koyeb, you will need to create a Dockerfile that sets up the environment for training the model, a repository to store the code, and finally deploy the app to Koyeb via git and built using the Dockerfile.

### Create a Dockerfile

We'll start by preparing a Dockerfile to ensure we have all the necessary dependencies installed, especially for GPU support. Create a `Dockerfile` with the following contents:

```docker
# Use the official Python base image
FROM python:3.11

# Clone the repository
RUN git clone https://github.com/mistralai/mistral-finetune.git

# Set the working directory
WORKDIR /mistral-finetune

# Update pip, install torch and other dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the 7B.yaml file
COPY 7B.yaml /mistral-finetune/example/7B.yaml

# Script to prepare the training data
COPY dataset_processing.py /mistral-finetune/dataset_processing.py
```

This Dockerfile is designed to set up an environment for fine-tuning the MistralAI language model. It automates the process of cloning the necessary repository, installing dependencies, and copying both the training configuration and the training dataset script.

### Create the repository

The final step is to create a new repository on GitHub to store the project files.

Once you're ready, run the following commands in your terminal to commit and push your code to the repository:

```docker
echo "# MistralFineTuning" >> README.md
git init
git add .
git commit -m "First Commit"
git branch -M main
git remote add origin [Your GitHub repository URL]
git push -u origin main
```

You should now have all your local code in your remote repository. Now it is time to deploy the Dockerfile.

### Deploy to Koyeb

In the [Koyeb control panel](https://app.koyeb.com/), while on the **Overview** tab, initiate the app creation and deployment process by clicking **Create App.** You can select a Worker application.

On the App deployment page:

1. Select **GitHub** as your deployment method.
2. Choose the repository where your code resides. For example, `MistralFineTuning`.
3. Select the GPU you wish to use, for example, `A100`. The training might work on other GPUs, but for performance and training accuracy, this is the recommended.
4. In the **Builder** section, choose **Dockerfile**.
5. In the **Service name** section, choose an appropriate name.
6. Finally, click **Deploy**.

## Running the Fine-Tuning process

Once the deployment is complete, you can start preparing and running the fine-tuning of the model.

The Dockerfile deployment has set up the base system needed to train the model, but it didn’t download a model, so that will be one of the first steps.

Since the next commands need interaction with the remote machine, you'll use the Koyeb CLI to access the remote machine through the terminal.

First, make sure you have the Koyeb CLI installed. You can find the installation instructions [**here**](https://www.koyeb.com/docs/build-and-deploy/cli/installation). Then, generate an API Token, which you can do [**here**](https://app.koyeb.com/user/settings/api/).

Now you are ready to log in with the Koyeb CLI:

```bash
koyeb login
```

First, input your API token key when asked for it.

To see a list of running instances, use the following command:

```bash
koyeb instances list
```

Note the instance ID you want to connect to. Then, create a remote terminal session to the remote machine:

```bash
koyeb instances exec <instance_id> /bin/bash
```

You now have an active remote session to the remote machine. All commands executed from now on will be on the remote machine.

As mentioned, the first step is to download the model to train, in this case the Mistral 7B Instruct:

```bash
mkdir mistral_models
wget https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-Instruct-v0.3.tar && tar -xf mistral-7B-Instruct-v0.3.tar -C mistral_models
```

It might take a couple of minutes for the model to be downloaded and extracted.

Next, to ensure proper compatibility, make sure that the Numpy package installed is at version 1.26.4:

```bash
pip install numpy==1.26.4
```

Now you can install the necessary libraries for executing the dataset script:

```bash
pip install requests pandas mistralai python-decouple
```

You can then copy the necessary information for the **`.env`** file:

```bash
echo "NEWS_API_KEY=<YOUR_NEWS_API_KEY>" > .env
echo "MISTRAL_API_KEY=<YOUR_MISTRAL_API_KEY>" >> .env

```

Make sure to replace the values with your own API keys.

And then you can execute the **`dataset_processing.py`** script:

```bash
python3 dataset_processing.py
```

It might take a couple of minutes to prepare the dataset. After it finishes, you should have two JSONL files corresponding to the training and evaluation datasets.

You can now validate those datasets with:

```bash
python3 -m utils.validate_data --train_yaml example/7B.yaml
```

You should get an estimate on the ETA for the training and there should not be any validation errors. If there are errors, you can fix them with:

```bash
python3 -m utils.reformat_data cryptocurrency_market_news_train.jsonl
python3 -m utils.reformat_data cryptocurrency_market_news_eval.jsonl
```

Validate the dataset again (if needed) and now there should be no errors:

```bash
python3 -m utils.validate_data --train_yaml example/7B.yaml
```

Everything is now ready to train the model, which you can do with:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.run --nproc-per-node 1 --master_port 65123 -m train example/7B.yaml
```

The **`CUDA_VISIBLE_DEVICES=0`** is necessary to make sure the training script recognizes the GPU on the remote machine.

This process will take several minutes, possibly even hours. It will show an estimate of the remaining processing time.

After it is finished, you will be able to evaluate the trained model against the standard model, which we will see how to do in the next section.

## Evaluating the Fine-Tuned Model

To evaluate the fine-tuned model, we first need to establish a baseline with the default model.

First, you need to install the necessary package on the remote machine:

```bash
pip install mistral_inference
```

Now you can test the default model by running:

```bash
mistral-chat /mistral-finetune/mistral_models/ --max_tokens 1024 --temperature 0.7 --instruct
```

It will ask you for a prompt. Let’s try this one:

```bash
Prompt: how is x adverstising doing?

To provide an accurate answer, I would need specific details about X advertising. However, I can share some general insights about advertising in today's digital landscape:

1. Digital Advertising: With the rise of the internet and mobile devices, digital advertising has become a significant part of the marketing mix. Platforms like Google, Facebook, Instagram, and YouTube offer targeted advertising options based on user demographics, interests, and online behavior.

2. Social Media Advertising: Social media platforms are popular advertising channels due to their vast user base and the ability to target ads based on user profiles. Brands use social media ads to increase brand awareness, drive traffic to their websites, and generate leads.

3. Influencer Marketing: Influencer marketing has grown in popularity, as brands partner with social media influencers to promote their products or services. This strategy allows brands to tap into the influencer's audience and leverage their credibility.

4. Traditional Advertising: While digital advertising is on the rise, traditional advertising methods like television, radio, print, and outdoor advertising still have a place in the marketing mix, especially for reaching broader audiences or those who are less tech-savvy.

5. Measuring Success: Advertising success is often measured by metrics such as reach, impressions, click-through rates, conversion rates, and return on investment (ROI). Brands use these metrics to evaluate the effectiveness of their advertising campaigns and make data-driven decisions.

Without more specific information about X advertising, it's difficult to say how they are doing. It's essential for brands to regularly analyze their advertising performance and adjust their strategies as needed to maximize their return on investment.

```

As you can see, the default model gave a very generic answer.

Now let’s run the fine-tuned model with:

```bash
mistral-chat /mistral-finetune/mistral_models/ --max_tokens 1024 --temperature 0.7 --instruct --lora_path /mistral-finetune/chat_test/checkpoints/checkpoint_000300/consolidated/lora.safetensors
```

And we use the same prompt:

```bash
Prompt: how is x adverstising doing?

X advertising is currently seeing a slight increase in performance.
```

As you can see, the fine-tuned model gave a much more accurate and precise answer.

### Impact on Domain Knowledge

Fine-tuning MistralAI on financial data significantly improves the model's ability to understand and operate within the financial domain. This process transforms the model into a specialized tool that has a deep understanding of the financial domain.

Here we have just exposed the model to a subset of recent news, but by exposing the model to more domain-specific data, such as financial reports, market analysis, and regulatory documents, it learns the precise meanings and nuances of financial terminology.

Fine-tuning also helps the model stay current with ongoing trends in the financial industry. This includes understanding the implications of market movements, economic indicators, and geopolitical events on financial markets.

This enhanced understanding and specialization enable the model to perform a wide range of finance-related tasks with greater accuracy, relevance, and compliance. This makes it an invaluable asset for financial professionals and organizations, helping them to make more informed decisions and improve their overall performance in the financial domain.

## Conclusion

You've just completed this tutorial on fine-tuning MistralAI on Koyeb Serverless GPUs.

While this guide focused on fine-tuning MistralAI for finance, the approach and techniques covered here are the same across various domains. Whether you're working with healthcare data, legal documents, technical manuals, or customer service interactions, fine-tuning can significantly improve the relevance and accuracy of AI models.

Have fun experimenting with your own datasets and seeing how fine-tuning can add value and improve performance in your specific area of interest!
