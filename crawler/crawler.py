import re
from typing import Dict, Union
import json
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import random
import time


def parse_job_posting_structured(content: str) -> Dict[str, Union[str, int]]:
    lines = content.strip().split('\n')
    parsed_data = {}
    
    if len(lines) > 0:
        parsed_data['title'] = lines[0].strip()
    
    if len(lines) > 1:
        company_location = lines[1].strip()
        if ',' in company_location:
            parts = company_location.split(',')
            parsed_data['company'] = parts[0].strip()
            parsed_data['location_short'] = ','.join(parts[1:]).strip()
    
    if len(lines) > 2 and ('PKR' in lines[2] or '$' in lines[2] or 'salary' in lines[2].lower()):
        parsed_data['salary_range'] = lines[2].strip()
    
    structured_output = {
        'title': '',
        'location': '',
        'department': '',
        'salary_range': '',
        'company_profile': '',
        'description': '',
        'requirements': '',
        'benefits': '',
        'telecommuting': 0,
        'has_company_logo': 0,
        'has_questions': 0,
        'employment_type': '',
        'required_experience': '',
        'required_education': '',
        'industry': '',
        'function': ''
    }
    
    current_section = None
    section_content = []
    
    for line in lines:
        line = line.strip()
        if not line or line in ['Apply for job', 'Share this job']:
            continue
        
        if any(keyword in line.lower() for keyword in ['remote', 'work from home', 'telecommuting', 'virtual']):
            structured_output['telecommuting'] = 1
            
        if line == 'Job Description':
            current_section = 'job_description'
            section_content = []
        
        elif line == 'Key Responsibilities:':
            if current_section == 'job_description':
                structured_output['description'] = '\n'.join(section_content).strip()
            current_section = 'key_responsibilities'
            section_content = []
        
        elif line in ["You're a Good Fit If You Have:", "Requirements:", "Qualifications:"]:
            if current_section == 'key_responsibilities':
                key_resp = '\n'.join(section_content).strip()
                if structured_output['description']:
                    structured_output['description'] += '\n' + key_resp
                else:
                    structured_output['description'] = key_resp
            current_section = 'requirements'
            section_content = []
        
        elif line == 'Education Requirement:':
            if current_section == 'requirements':
                structured_output['requirements'] = '\n'.join(section_content).strip()
            current_section = 'education_requirement'
            section_content = []
        
        elif line in ['What We Offer:', 'Benefits:', 'Perks:']:
            if current_section == 'education_requirement':
                edu_req = '\n'.join(section_content).strip()
                if structured_output['requirements']:
                    structured_output['requirements'] += '\n' + edu_req
                else:
                    structured_output['requirements'] = edu_req
            current_section = 'benefits'
            section_content = []
        
        elif line == 'Job Skills':
            if current_section == 'benefits':
                structured_output['benefits'] = '\n'.join(section_content).strip()
            current_section = 'job_skills'
            section_content = []
        
        elif line == 'Job Details':
            if current_section == 'job_skills':
                skills = [skill.strip() for skill in section_content if skill.strip()]
                if skills:
                    skills_text = 'Required Skills: ' + ', '.join(skills)
                    if structured_output['requirements']:
                        structured_output['requirements'] += '\n' + skills_text
                    else:
                        structured_output['requirements'] = skills_text
            current_section = 'job_details'
            section_content = []
        
        elif current_section == 'job_details':
            line_clean = line.strip()
            
            if ':' in line_clean and line_clean.endswith(':'):
                field_name = line_clean[:-1].strip().lower().replace(' ', '_')
                if not hasattr(parse_job_posting_structured, '_field_queue'):
                    parse_job_posting_structured._field_queue = []
                parse_job_posting_structured._field_queue.append(field_name)
            elif ':' in line_clean and not line_clean.endswith(':'):
                key, value = line_clean.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key == 'industry':
                    structured_output['industry'] = value
                elif key == 'functional_area':
                    structured_output['function'] = value
                    if not structured_output['department']:
                        structured_output['department'] = value
                elif key in ['job_type', 'employment_type']:
                    structured_output['employment_type'] = value
                elif key == 'job_location':
                    structured_output['location'] = value
                elif key == 'minimum_experience':
                    structured_output['required_experience'] = value
                elif key == 'minimum_education':
                    structured_output['required_education'] = value
            else:
                if hasattr(parse_job_posting_structured, '_field_queue') and parse_job_posting_structured._field_queue:
                    field_key = parse_job_posting_structured._field_queue.pop(0)
                    
                    if field_key == 'industry':
                        structured_output['industry'] = line_clean
                    elif field_key == 'functional_area':
                        structured_output['function'] = line_clean
                        if not structured_output['department']:
                            structured_output['department'] = line_clean
                    elif field_key in ['job_type', 'employment_type']:
                        structured_output['employment_type'] = line_clean
                    elif field_key == 'job_location':
                        structured_output['location'] = line_clean
                    elif field_key == 'minimum_experience':
                        structured_output['required_experience'] = line_clean
                    elif field_key == 'minimum_education':
                        structured_output['required_education'] = line_clean
        else:
            if current_section and line:
                section_content.append(line)
    
    if current_section == 'benefits' and section_content:
        structured_output['benefits'] = '\n'.join(section_content).strip()
    elif current_section == 'job_skills' and section_content:
        skills = [skill.strip() for skill in section_content if skill.strip()]
        if skills:
            skills_text = 'Required Skills: ' + ', '.join(skills)
            if structured_output['requirements']:
                structured_output['requirements'] += '\n' + skills_text
            else:
                structured_output['requirements'] = skills_text
    
    if hasattr(parse_job_posting_structured, '_field_queue'):
        delattr(parse_job_posting_structured, '_field_queue')
    
    if parsed_data.get('title'):
        structured_output['title'] = parsed_data['title']
    if parsed_data.get('salary_range'):
        structured_output['salary_range'] = parsed_data['salary_range']
    
    if not structured_output['location'] and parsed_data.get('location_short'):
        structured_output['location'] = parsed_data['location_short']
    
    if not structured_output['department']:
        if structured_output['function']:
            structured_output['department'] = structured_output['function']
        elif structured_output['title']:
            title_lower = structured_output['title'].lower()
            if any(word in title_lower for word in ['marketing', 'social media']):
                structured_output['department'] = 'Marketing'
            elif any(word in title_lower for word in ['sales', 'business development']):
                structured_output['department'] = 'Sales'
            elif any(word in title_lower for word in ['hr', 'human resource']):
                structured_output['department'] = 'Human Resources'
            elif any(word in title_lower for word in ['it', 'developer', 'engineer']):
                structured_output['department'] = 'Technology'
            elif 'intern' in title_lower:
                structured_output['department'] = 'Other'
    
    if not structured_output['function']:
        structured_output['function'] = structured_output['department'] if structured_output['department'] else 'Other'
    
    if structured_output['description'] and len(structured_output['description']) > 500:
        desc_parts = structured_output['description'].split('\n\n', 1)
        if len(desc_parts) > 1 and len(desc_parts[0]) > 200:
            structured_output['company_profile'] = desc_parts[0]
            structured_output['description'] = desc_parts[1]
    
    if (structured_output['company_profile'] or 
        (structured_output['description'] and len(structured_output['description']) > 300)):
        structured_output['has_company_logo'] = 1
    
    full_content = content.lower()
    if '?' in full_content or 'question' in full_content or 'interview' in full_content:
        structured_output['has_questions'] = 1
    
    return structured_output

def append_job_to_file(job_data: dict, filename: str = "jobs.json"):
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    with open(filename, "r", encoding="utf-8") as f:
        try:
            jobs_list = json.load(f)
        except json.JSONDecodeError:
            jobs_list = []

    jobs_list.append(job_data)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(jobs_list, f, indent=2)

LISTING_URL = "https://www.rozee.pk/job/jsearch/q/all/fc/1184"

def check_job_detail_loaded(driver):
    try:
        job_detail = driver.find_element(By.ID, "jobDetail")
        content = job_detail.text.strip()
        
        if content:
            print("Job detail loaded. Content length:", len(content))
            parsed = parse_job_posting_structured(content)
            append_job_to_file(parsed)
            return True
        else:
            print("Job detail div found but empty")
            return False
    except:
        print("Job detail div not found")
        return False
    

def main():
    options = Options()
    # options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    driver.set_page_load_timeout(30)

    try:
        print("Navigating to the listing page...")
        driver.get(LISTING_URL)
        time.sleep(3)

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.job"))
        )

        job_cards = driver.find_elements(By.CSS_SELECTOR, "div.job")
        print("Found", len(job_cards), "job cards")

        start_index = 5
        test_cards = 10

        for i in range(start_index, min(len(job_cards), start_index + test_cards)):
            card_number = i + 1
            print("Testing job card #", card_number)
            try:
                job_card = job_cards[i]
                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});", job_card
                )
                time.sleep(1)
                job_card.click()
                time.sleep(2)
                check_job_detail_loaded(driver)
            except Exception as e:
                print("Error with card #", card_number, ":", e)

    finally:
        print("Closing browser...")
        driver.quit()


if __name__ == "__main__":
    main()

