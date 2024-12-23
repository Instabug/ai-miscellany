import os
import time
import yaml
from typing import List, Dict

import base64

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

BROKEN_FUNCTIONALITY_PROMPT = """
Context:
You are given a lot of screenshots related to a user's interaction with our application.
Your task is to analyze this data and provide a detailed description of the user's steps, actions, and the sequence in which they occurred.
The goal is to understand the user's behavior and identify any issues or patterns.

Instructions:

1. Detailed Steps:
    Based on your analysis, provide a step-by-step description of the user's journey within the application. For each step, include the following details:

    a. Action: What the user did (e.g., clicked a button, entered text).
    b. Context: Where in the application the action took place (e.g., on which screen or page).
    c. Outcome: What happened as a result of the action (e.g., navigated to a new screen, received an error message).
    d. Corroboration: Reference the screenshots that support this step.

2. Issues and Observations:
    Highlight any potential issues or unusual behaviors you observe. This could include errors, unexpected actions, or anything that deviates from typical usage patterns.

3. Summary:
    Conclude with a summary of the user's overall session, including the main goals they appeared to be pursuing and any challenges they encountered.

Screenshots:
"""


# Utility Functions
def encode_image(image_path: str) -> str:
    """Encodes an image to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        return ""


def get_screenshots_urls(session_path: str) -> List[str]:
    screenshots_filenames = [
        f for f in os.listdir(session_path) if f.lower().endswith((".jpeg", ".png"))
    ]

    # Sort the screenshots chronologically
    sorted_screenshots = sorted(
        screenshots_filenames,
        key=lambda x: int(x.split(".")[0]),
    )

    screenshots_urls = (
        os.path.join(session_path, screenshot) for screenshot in sorted_screenshots
    )

    return screenshots_urls


def call_model(llm: BaseChatModel, screenshots_urls: list[str]):
    """Sets up the LangChain LLMChain with a chat model and prompt template."""

    encoded_screenshots = (
        encode_image(screenshot_url) for screenshot_url in screenshots_urls
    )

    system_message = SystemMessage(content=BROKEN_FUNCTIONALITY_PROMPT)

    images_messages = (
        HumanMessage(
            content=[
                # {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                },
            ],
        )
        for image in encoded_screenshots
    )

    messages = [system_message, *images_messages]

    response = llm.invoke(messages)
    return response


def process_session(
    session_id: str, session_path: str, llm: BaseChatModel
) -> Dict[str, str]:
    """Processes a single session using LangChain."""
    screenshots_urls = get_screenshots_urls(session_path)

    if not screenshots_urls:
        print(f"No screenshots found for session: {session_id}")
        return {}

    try:
        start_time = time.time()
        response = call_model(llm, screenshots_urls)
        end_time = time.time()

        return {
            "session_id": session_id,
            "response": response.content,
            "elapsed_time": round(end_time - start_time, 2),
            "usage": response.usage_metadata,
        }
    except Exception as e:
        print(f"Error processing session {session_id}: {e}")
        return {}


def benchmark_broken_functionality(base_directory: str, app_name: str) -> Dict:
    """Processes all sessions in the base directory and returns results."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0, max_tokens=4000)

    results = {
        "app_name": app_name,
        "sessions": [],
    }

    for session_id in os.listdir(base_directory):
        session_path = os.path.join(base_directory, session_id, session_id)
        if os.path.isdir(session_path):
            session_result = process_session(
                session_id=session_id,
                session_path=session_path,
                llm=llm,
            )
            if session_result:
                results["sessions"].append(session_result)

    return results


def save_to_yaml(object: Dict, output_file: str):
    """Saves a dictionary to a YAML file."""

    class BlockStyleDumper(yaml.SafeDumper):
        def represent_str(self, data):
            if "\n" in data:  # Use block style for multiline strings
                return self.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return super().represent_str(data)

    yaml.add_representer(str, BlockStyleDumper.represent_str, Dumper=BlockStyleDumper)

    with open(output_file, "w") as yaml_file:
        yaml.dump(object, yaml_file, Dumper=BlockStyleDumper, allow_unicode=True)


if __name__ == "__main__":
    BASE_DIRECTORY = "data/demo_data"
    APP_NAME = "App1"
    OUTPUT_FILE = f"session_analysis_results_{APP_NAME}.yaml"

    results = benchmark_broken_functionality(BASE_DIRECTORY, APP_NAME)
    save_to_yaml(results, OUTPUT_FILE)
    print(f"Results saved to {OUTPUT_FILE}")
