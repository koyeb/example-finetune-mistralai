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
