import json
import boto3
import logging
from tqdm import tqdm
import argparse
import time
from bs4 import BeautifulSoup
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("scenario_generation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_preferences(text):
    """
    Parse the generated conversation text containing XML-like task elements
    and extract preference, question, and explanation information.

    Args:
        text (str): The generated conversation text containing XML-like elements

    Returns:
        list: A list of dictionaries containing parsed preferences
    """
    soup = BeautifulSoup(text, "html.parser")
    tasks = soup.find_all("task")

    preferences = []
    for task in tasks:
        scenario = {
            "preference": task.find("preference").text.strip() if task.find("preference") else "",
            "question": task.find("question").text.strip() if task.find("question") else "",
            "explanation": task.find("explanation").text.strip() if task.find("explanation") else "",
        }
        preferences.append(scenario)

    return preferences


def generate_message(
    bedrock_runtime,
    model_id,
    system_prompt=None,
    messages=None,
    max_tokens=3000,
    temperature=1,
    max_retries=10,
):
    retries = 0
    while retries < max_retries:
        try:
            body = json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "system": system_prompt,
                    "messages": messages,
                    "temperature": temperature,
                }
            )
            response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
            response_body = json.loads(response.get("body").read())
            return response_body["content"][0]["text"]
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            if "reduce" in str(e):
                raise Exception(f"max context length is exceeded")
            if retries == max_retries - 1:
                logger.info("Sleeping 30 seconds before retry")
                time.sleep(30)
                retries = 0
            retries += 1
            time.sleep(10)


def save_preferences(preferences, topic, batch_num):
    """
    Save preferences to a JSON file with timestamp and batch number
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{topic}_preferences_batch_{batch_num}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(preferences, f, indent=4)
    logger.info(f"Saved batch {batch_num} to {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract conversations from a dataset.")
    parser.add_argument("--num_generation", type=int, default=5)
    parser.add_argument("--preferences_per_generation", type=int, default=10)
    args = parser.parse_args()

    client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    # Read prompt template
    gen_data_file = "gen_data_prompt.txt"
    with open(gen_data_file, "r") as f:
        gen_data_prompt = f.read()
    all_topics = {
        "Home_Cooking": "Home_Cooking, such as Recipe Modifications, Kitchen Equipment, Meal Planning, Cooking Techniques, Food Storage",
        "Personal_Finance": "Personal_Finance, such as Bill Management, Credit Score, Daily Budgeting, Banking Issues, Savings Tips",
        "Home_Maintenance": "Home_Maintenance, such as Appliance Repairs, Plumbing Issues, Cleaning Methods, Power Problems, HVAC Care",
        "Family_Care": "Family_Care, such as Child Development, Elderly Support, Family Activities, Parenting Tips, Work-Life Balance",
        "Digital_Services": "Digital_Services, such as App Troubleshooting, Account Security, Subscription Management, Device Setup, Software Updates",
        "Weather_Planning": "Weather_Planning, such as Daily Forecasts, Event Planning, Storm Preparation, Seasonal Activities, Travel Weather",
        "Personal_Documents": "Personal_Documents, such as ID Renewal, Document Filing, Form Completion, Legal Paperwork, Record Keeping",
        "Shopping_Assistance": "Shopping_Assistance, such as Price Comparison, Product Reviews, Warranty Info, Return Policies, Discount Finding",
        "Time_Management": "Time_Management, such as Schedule Planning, Deadline Tracking, Calendar Organization, Task Prioritization, Routine Building",
        "Communication_Help": "Communication_Help, such as Email Writing, Message Drafting, Call Scripts, Meeting Planning, Social Media Posts",
        "Medical_Care": "Medical_Care, such as Symptom Checking, Medication Info, Appointment Booking, Insurance Questions, Health Records",
        "Entertainment_Planning": "Entertainment_Planning, such as Event Tickets, Party Planning, Holiday Activities, Group Gatherings, Weekend Ideas",
        "Personal_Growth": "Personal_Growth, such as Habit Formation, Goal Setting, Self Improvement, Skill Development, Career Planning",
        "Local_Services": "Local_Services, such as Business Hours, Service Booking, Location Finding, Price Inquiries, Review Checking",
        "Relationship_Advice": "Relationship_Advice, such as Communication Tips, Conflict Resolution, Dating Guidance, Friend Issues, Family Dynamics",
        "Smart_Home": "Smart_Home, such as Device Control, Automation Setup, Network Issues, Energy Management, Security Settings",
        "Personal_Safety": "Personal_Safety, such as Emergency Plans, Safety Precautions, Security Tips, First Aid Help, Risk Assessment",
        "Moving_Relocation": "Moving_Relocation, such as Planning Timeline, Service Booking, Address Changes, Packing Tips, Setup Utilities",
        "Gift_Giving": "Gift_Giving, such as Gift Ideas, Occasion Planning, Budget Options, Wrapping Tips, Delivery Tracking",
        "Seasonal_Tasks": "Seasonal_Tasks, such as Holiday Planning, Weather Preparation, Wardrobe Changes, Decoration Ideas, Activity Planning",
    }
    for topic, topic_prompt in all_topics.items():
        all_preferences = []
        for batch in tqdm(range(args.num_generation), desc="Generating scenario batches"):
            logger.info(f"Starting batch {batch + 1}/{args.num_generation}")

            # Modify prompt to request specific number of preferences
            modified_prompt = gen_data_prompt.replace("{topic}", topic_prompt)
            modified_prompt = modified_prompt.replace("{num_prefs}", str(args.preferences_per_generation))
            messages = [
                {
                    "role": "user",
                    "content": f"{modified_prompt}",
                }
            ]
            system_prompt = "You are a helpful assistant."

            # Generate and parse preferences
            generated_conversation = generate_message(
                client,
                model_id=model_id,
                system_prompt=system_prompt,
                messages=messages,
                temperature=0.5,  # Adjust temperature for diversity
            )

            batch_preferences = parse_preferences(generated_conversation)

            if len(batch_preferences) != args.preferences_per_generation:
                logger.warning(
                    f"Expected {args.preferences_per_generation} preferences but got {len(batch_preferences)}"
                )

            all_preferences.extend(batch_preferences)

            time.sleep(2)

        final_output = f"{topic}_preferences.json"
        with open(final_output, "w") as f:
            json.dump(all_preferences, f, indent=4)

        logger.info(f"Generation complete. Total preferences generated: {len(all_preferences)} for topic: {topic}")
        logger.info(f"{topic} preferences saved to {final_output}")
