from scrapegraphai.graphs import SmartScraperGraph
from scrapegraph_py import ScrapeGraphAI
from dotenv import load_dotenv
from typing import Optional
from openai import OpenAI
import json
import os
import time

# Load environment variables
load_dotenv()

# Define what you want the AI to extract
prompt = """
Extract the detailed job description from this page. 
I specifically need the 'About the job' section, 'Minimum qualifications', 'Preferred qualifications', and 'Responsibilities'.
"""

# 2. Change the schema to match the data you actually want
schema = {
    "type": "object",
    "properties": {
        "job_title": {"type": "string", "description": "The title of the job"},
        "minimum_qualifications": {
            "type": "array", 
            "items": {"type": "string"}, 
            "description": "List of minimum qualifications"
        },
        "preferred_qualifications": {
            "type": "array", 
            "items": {"type": "string"}, 
            "description": "List of preferred qualifications"
        },
        "responsibilities": {
            "type": "array", 
            "items": {"type": "string"}, 
            "description": "List of job responsibilities"
        },
        "about_the_job": {"type": "string", "description": "The general 'About the job' description paragraph"}
    },
    "required": ["job_title", "minimum_qualifications", "responsibilities"]
}


# Give it the URL
urls = [
    "https://www.google.com/about/careers/applications/jobs/results/84332207971148486-research-engineer-tool-use-deepmind?company=DeepMind&utm_source=deepmind&utm_medium=jobposting&page=3",
    "https://www.google.com/about/careers/applications/jobs/results/107269257477137094-research-engineer-human-understanding-deepmind?company=DeepMind&utm_source=deepmind&utm_medium=jobposting&page=4",
    "https://www.google.com/about/careers/applications/jobs/results/107641669796405958-staff-research-engineer-and-scientist-gemini-posttraining-deepmind?company=DeepMind&utm_source=deepmind&utm_medium=jobposting&page=5",
    "https://www.google.com/about/careers/applications/jobs/results/141902048390980294-staff-software-engineer-gemini-app-data-engineering-deepmind?company=DeepMind&utm_source=deepmind&utm_medium=jobposting&page=5",
    "https://www.google.com/about/careers/applications/jobs/results/105870811830592198-research-engineer-humanoids-deepmind?company=DeepMind&utm_source=deepmind&utm_medium=jobposting&page=5",
    "https://www.google.com/about/careers/applications/jobs/results/78472631333855942-research-engineer-pretraining-deepmind?company=DeepMind&utm_source=deepmind&utm_medium=jobposting&page=5",
    "https://www.google.com/about/careers/applications/jobs/results/143298054496101062-research-engineer-multimodal-reasoning-for-information-literacy-deepmind?company=DeepMind&utm_source=deepmind&utm_medium=jobposting",
    "https://www.google.com/about/careers/applications/jobs/results/76114032863388358-research-engineer-winslow-deepmind?company=DeepMind&utm_source=deepmind&utm_medium=jobposting",
    "https://www.google.com/about/careers/applications/jobs/results/95635593379095238-research-engineer-agi-safety-and-alignment-deepmind?company=DeepMind&utm_source=deepmind&utm_medium=jobposting&page=2",
    "https://www.google.com/about/careers/applications/jobs/results/102552346151527110-research-engineer-agi-safety-and-alignment-deepmind?company=DeepMind&utm_source=deepmind&utm_medium=jobposting&page=3",
    "https://www.google.com/about/careers/applications/jobs/results/113676455329571526-research-engineer-deepmind?company=DeepMind&utm_source=deepmind&utm_medium=jobposting&page=2"
]

# Create the scraper

sgai = ScrapeGraphAI(api_key=os.environ.get("SCRAPE_GRAPH_API"))

all_scraped_jobs = []

print(f"Starting to scrape {len(urls)} URLs...")

# 3. Loop through each URL
for i, url in enumerate(urls):
    print(f"\n[{i+1}/{len(urls)}] Scraping: {url}")
    
    try:
        # Run the extraction
        res = sgai.extract(
            prompt=prompt,
            url=url,
            schema=schema,
        )

        # Check if it was successful
        if res.status == "success":
            # Add the source URL to the data so you know where it came from
            job_data = res.data.json_data
            job_data['source_url'] = url 
            
            all_scraped_jobs.append(job_data)
            print(f"✅ Success: Extracted '{job_data.get('job_title', 'Unknown Title')}'")
        else:
            print(f"❌ Failed: {res.error}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

    # 4. Be polite! Wait 2-3 seconds before scraping the next URL
    if i < len(urls) - 1: # No need to wait after the very last URL
        print("Waiting 3 seconds before next request...")
        time.sleep(3)

# 5. Save all the collected data to a JSON file
print("\n" + "="*40)
print(f"Finished! Successfully scraped {len(all_scraped_jobs)} out of {len(urls)} jobs.")

if all_scraped_jobs:
    with open("jobs_data.json", "w", encoding="utf-8") as f:
        json.dump(all_scraped_jobs, f, indent=4, ensure_ascii=False)
    print("📁 Data saved to 'jobs_data.json'")