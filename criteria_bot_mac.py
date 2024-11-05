import os
import re
import logging
import threading
import csv
import tkinter as tk
from tkinter import Tk, Label, Entry, Button, Text, filedialog, StringVar, Frame, END, W, E, S, N, scrolledtext
from pdfminer.high_level import extract_text
from textstat import flesch_reading_ease
from docx import Document
import pandas as pd
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO, filename='evaluation.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')


def extract_emails_from_text(text):
    """
    Extracts individual emails from a text string.
    Assumes that emails are separated by headers starting with 'From: '.
    Adjust the pattern as needed based on the actual format.
    """
    # Regular expression pattern to identify email headers
    email_pattern = r'(From: .+?)(?=From: |\Z)'
    emails = re.findall(email_pattern, text, flags=re.DOTALL | re.MULTILINE)
    logging.info(f"Extracted {len(emails)} emails from text.")
    return emails


def extract_emails_from_pdf(pdf_path):
    """
    Extracts emails from a PDF file.
    """
    try:
        text = extract_text(pdf_path)
        emails = extract_emails_from_text(text)
        return emails
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return []


def extract_emails_from_docx(docx_path):
    """
    Extracts emails from a Word (.docx) file.
    """
    try:
        doc = Document(docx_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        emails = extract_emails_from_text(text)
        return emails
    except Exception as e:
        logging.error(f"Error extracting text from {docx_path}: {e}")
        return []


def extract_emails_from_txt(txt_path):
    """
    Extracts emails from a text (.txt) file.
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        emails = extract_emails_from_text(text)
        return emails
    except Exception as e:
        logging.error(f"Error reading {txt_path}: {e}")
        return []


def extract_emails_from_csv(csv_path):
    """
    Extracts emails from a CSV file.
    Assumes that the email content is in a column named 'email' or 'Email'.
    """
    emails = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                email_content = row.get('email') or row.get('Email')
                if email_content:
                    emails.append(email_content)
        logging.info(f"Extracted {len(emails)} emails from {csv_path}")
        return emails
    except Exception as e:
        logging.error(f"Error reading {csv_path}: {e}")
        return []


def evaluate_comment_locally(comment_text, criteria, comment_id, filename):
    """
    Evaluates a policy comment based on the specified criteria.
    Returns the evaluation data as a dictionary.
    """
    # The maximum score is the number of criteria
    max_score = len(criteria)
    score = 0

    # Split comment into sentences
    sentences = re.split(r'(?<=[.!?])\s+', comment_text.strip())

    # Evaluate each criterion
    evaluation_data = {}
    for criterion, details in criteria.items():
        details['score'] = 0  # Reset score
        details['sentences'] = []  # Reset sentences

        if criterion == 'Writing Quality':
            # Use readability score and sentence length as proxies for writing quality
            readability = flesch_reading_ease(comment_text)
            avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            if readability >= 60 and avg_sentence_length <= 20:
                details['score'] = 1
        else:
            # Check for presence of keywords in sentences
            for sentence in sentences:
                for keyword in details['keywords']:
                    if keyword.lower() in sentence.lower():
                        details['score'] = 1
                        details['sentences'].append(sentence.strip())
                        break  # Avoid duplicate sentences for the same keyword

        # Record whether the criterion was met
        evaluation_data[criterion] = 'Yes' if details.get('score', 0) else 'No'

    # Calculate total score
    score = sum(details.get('score', 0) for details in criteria.values())

    # Ensure the score does not exceed max_score
    score = min(score, max_score)

    evaluation_data['Quality Score'] = score

    # Sentiment Analysis
    blob = TextBlob(comment_text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        evaluation_data['Sentiment'] = 'Positive'
    elif sentiment < 0:
        evaluation_data['Sentiment'] = 'Negative'
    else:
        evaluation_data['Sentiment'] = 'Neutral'

    # Extract Keywords (Top 5 most frequent words excluding stopwords)
    from collections import Counter
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\b\w+\b', comment_text.lower())
    words = [word for word in words if word not in stop_words]
    word_counts = Counter(words)
    top_keywords = [word for word, count in word_counts.most_common(5)]
    evaluation_data['Keywords'] = ', '.join(top_keywords)

    # Assign Themes (based on predefined keywords)
    themes = []
    theme_keywords = {
        'Housing Costs': ['housing', 'room', 'board', 'deduction'],
        'Scheduling Flexibility': ['schedule', 'flexibility', 'hours', 'time'],
        'Au Pair Program': ['au pair', 'program', 'cultural exchange'],
        # Add more themes as needed
    }
    for theme, keywords in theme_keywords.items():
        for keyword in keywords:
            if keyword.lower() in comment_text.lower():
                themes.append(theme)
                break
    evaluation_data['Themes'] = ', '.join(set(themes)) if themes else 'N/A'

    # Stakeholder (Assuming we can detect based on content)
    if 'host family' in comment_text.lower():
        evaluation_data['Stakeholder'] = 'Host Family'
    elif 'au pair' in comment_text.lower():
        evaluation_data['Stakeholder'] = 'Au Pair'
    else:
        evaluation_data['Stakeholder'] = 'N/A'

    # Market (Assuming we can detect based on content)
    if 'tier 3' in comment_text.lower() or 'tier 4' in comment_text.lower():
        evaluation_data['Market'] = 'Tier 3 or 4 Market'
    else:
        evaluation_data['Market'] = 'N/A'

    # Priority (Assign based on score or other criteria)
    if score >= max_score - 1:
        evaluation_data['Priority'] = 'High'
    elif score >= max_score / 2:
        evaluation_data['Priority'] = 'Medium'
    else:
        evaluation_data['Priority'] = 'Low'

    # Recommendations (Assuming any sentences with 'should' or 'recommend')
    recommendations = []
    for sentence in sentences:
        if 'should' in sentence.lower() or 'recommend' in sentence.lower():
            recommendations.append(sentence.strip())
    evaluation_data['Recommendations'] = ' '.join(recommendations) if recommendations else 'N/A'

    # Add other fields
    evaluation_data['Comment ID'] = comment_id
    evaluation_data['Text'] = comment_text
    evaluation_data['Source'] = filename

    # Add Email Excerpt for reference (first 200 characters)
    evaluation_data['Email Excerpt'] = comment_text[:200]

    # Logging the email content for traceback
    logging.info(f"Evaluated Comment ID {comment_id} from {filename}. Comment content: {comment_text[:200]}...")

    return evaluation_data


def process_files_in_directory(directory_path, criteria, output_text_widget):
    """
    Processes all files in the given directory and updates the output_text_widget with evaluations.
    Supports PDF, TXT, DOCX, and CSV files.
    Also collects evaluation data to export to an Excel file.
    """
    evaluation_results = []
    comment_id = 1
    try:
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            if os.path.isfile(filepath):
                emails = []
                if filename.lower().endswith('.pdf'):
                    logging.info(f"Processing PDF file: {filename}")
                    output_text_widget.insert(END, f"Processing file: {filename}\n\n")
                    emails = extract_emails_from_pdf(filepath)
                elif filename.lower().endswith('.docx'):
                    logging.info(f"Processing Word file: {filename}")
                    output_text_widget.insert(END, f"Processing file: {filename}\n\n")
                    emails = extract_emails_from_docx(filepath)
                elif filename.lower().endswith('.txt'):
                    logging.info(f"Processing TXT file: {filename}")
                    output_text_widget.insert(END, f"Processing file: {filename}\n\n")
                    emails = extract_emails_from_txt(filepath)
                elif filename.lower().endswith('.csv'):
                    logging.info(f"Processing CSV file: {filename}")
                    output_text_widget.insert(END, f"Processing file: {filename}\n\n")
                    emails = extract_emails_from_csv(filepath)
                else:
                    logging.info(f"Unsupported file type: {filename}")
                    continue  # Skip unsupported file types

                logging.info(f"Extracted {len(emails)} emails from {filename}")

                for idx, email_content in enumerate(emails):
                    output_text_widget.insert(END, f"Evaluating email {idx+1}/{len(emails)} in {filename}\n")
                    try:
                        evaluation_data = evaluate_comment_locally(email_content, criteria, comment_id, filename)
                        if evaluation_data:
                            # Display in GUI
                            output_text_widget.insert(END, f"Comment ID: {comment_id}\n")
                            output_text_widget.insert(END, f"Email Excerpt: {email_content[:200]}...\n")
                            output_text_widget.insert(END, f"Score: {evaluation_data['Quality Score']}\n")
                            output_text_widget.insert(END, f"Sentiment: {evaluation_data['Sentiment']}\n")
                            output_text_widget.insert(END, '-' * 80 + '\n')
                            # Collect evaluation data
                            evaluation_results.append(evaluation_data)
                            comment_id += 1
                    except Exception as e:
                        logging.error(f"Error evaluating email {idx+1} in {filename}: {e}")
                        output_text_widget.insert(END, f"Error evaluating email {idx+1} in {filename}: {e}\n")
                        output_text_widget.insert(END, f"Email content: {email_content[:200]}...\n")
                        output_text_widget.insert(END, '-' * 80 + '\n')
                    output_text_widget.update()
        # Check if evaluation_results has data
        if evaluation_results:
            # After processing all files, export results to Excel
            export_excel(evaluation_results)
            logging.info("Evaluation completed successfully.")
            output_text_widget.insert(END, "Evaluation completed successfully.\n")
            output_text_widget.insert(END, f"Results have been saved to 'evaluation_results.xlsx'.\n")
        else:
            logging.warning("No evaluation results to export.")
            output_text_widget.insert(END, "No evaluation results to export.\n")
    except Exception as e:
        logging.error(f"Error processing files: {e}")
        output_text_widget.insert(END, f"Error processing files: {e}\n")


def export_excel(evaluation_results):
    """
    Exports the evaluation results to an Excel file.
    """
    if not evaluation_results:
        logging.warning("No evaluation results to export.")
        return
    df = pd.DataFrame(evaluation_results)
    # Rearranging columns to match the desired output
    columns_order = [
        'Comment ID', 'Email Excerpt', 'Text', 'Themes', 'Sentiment', 'Quality Score', 'Stakeholder',
        'Market', 'Priority', 'Keywords', 'Recommendations', 'Source'
    ]
    # Verify all columns are present
    missing_columns = set(columns_order) - set(df.columns)
    if missing_columns:
        logging.error(f"Missing columns in DataFrame: {missing_columns}")
        print(f"Missing columns in DataFrame: {missing_columns}")
        return
    df = df[columns_order]
    try:
        df.to_excel('evaluation_results.xlsx', index=False)
        logging.info("Evaluation results have been exported to 'evaluation_results.xlsx'.")
    except Exception as e:
        logging.error(f"Error exporting results to Excel: {e}")
        print(f"Error exporting results to Excel: {e}")


def start_evaluation_thread(directory_path, criteria, output_text_widget):
    """
    Starts the evaluation in a separate thread to keep the GUI responsive.
    """
    threading.Thread(target=process_files_in_directory, args=(directory_path, criteria, output_text_widget)).start()


def create_gui():
    """
    Creates the GUI interface.
    """
    root = Tk()
    root.title("Policy Comment Evaluator")
    root.geometry("800x600")

    # Criteria and their default keywords
    criteria = {
        'Clear Cause-and-Effect Reasoning': {
            'description': 'Comments that clearly explain how specific policy changes might lead to anticipated outcomes.',
            'keywords': ['because', 'therefore', 'thus', 'hence', 'consequently', 'as a result', 'lead to', 'result in'],
        },
        'Constructive Criticism': {
            'description': 'Detailed critiques or feedback that provide actionable insights or suggestions for improvement.',
            'keywords': ['suggest', 'recommend', 'advise', 'propose', 'should', 'could', 'improve', 'enhance'],
        },
        'Writing Quality': {
            'description': 'Comments written in a clear, concise, and well-organized manner.',
        },
        'Objective Data and Facts': {
            'description': 'Comments that include factual evidence, specific figures, or credible references to support arguments.',
            'keywords': ['data', 'evidence', 'statistics', 'research', 'study', 'percent', 'number', 'figure', 'according to'],
        },
        'Provisions Mentioned': {
            'description': 'Comments that mention specific provisions from 22 CFR Part 62.',
            'keywords': [
                # Add the provision keywords here
                '62.31(a)', 'Au Pairs', 'Purpose', 'Part-time program', 'Full-time program',
                '62.31(b)', 'Program Designation',
                '62.31(c)', 'Program Conditions', 'Standard Operating Procedures', 'Foreign Third Parties', 'Vetting',
                'Rematch', 'Emergency Procedures', 'Local Coordinator',
                '62.31(d)', 'Au Pair Eligibility', 'Suitability', 'Interview', 'Driver\'s License', 'Physical Exam', 'Vaccinations',
                '62.31(e)', 'Au Pair Placement', 'Host Family Agreement', 'Personal Space',
                '62.31(f)', 'Au Pair Orientation', 'Pre-Departure Materials', 'Post-Arrival Orientation',
                'Travel Arrangements', 'Compensation', 'Benefits', 'Deductions', 'Work Hours', 'Child Care Duties', 'Taxes',
                '62.31(g)', 'Au Pair Training', 'Child Safety', 'Child Development', 'Driving',
                '62.31(h)', 'Host Family Selection', 'English Proficiency', 'Residency', 'Criminal Background Checks',
                '62.31(i)', 'Host Family Orientation', 'Host Family Agreement', 'IRS Taxation', 'Child Care Hours',
                '62.31(j)', 'Host Family Agreement', 'Fees', 'Duties', 'Weekly Schedule', 'Weekends', 'Paid Time Off',
                'Compensation', 'Room and Board', 'In-kind Benefits', 'Training', 'Home Environment', 'Changes', 'Rematch',
                '62.31(k)', 'Au Pair Limitations', 'Protections', 'Rest', 'Paid Time Off', 'Sick Leave', 'Privacy',
                'Identification Papers', 'Sexual Harassment', 'Exploitation', 'Abuse', 'Nanny Cam', 'Photography',
                '62.31(l)', 'Rematch', 'Standard Operating Procedures', 'Refund', 'Program Duration',
                '62.31(m)', 'Hours', 'Host Family Agreement', 'Overtime', 'Exigent Circumstances', 'Child Care', 'Timesheets',
                '62.31(n)', 'Compensation', 'Weekly Payments', 'Tiers', 'Minimum Wage', 'Overtime', 'Deductions',
                'Room and Board', 'In-kind Benefits',
                '62.31(o)', 'Educational Component', 'Academic Option', 'Online Classes', 'Continuing Education',
                'Customized Courses', 'Community Service', 'Educational Stipend',
                '62.31(p)', 'Monitoring', 'Records', 'Local Coordinators', 'Personal Contact', 'Host Family Visits',
                'Communication',
                '62.31(q)', 'Duration', 'Extensions',
                '62.31(r)', 'Reporting Requirements', 'SEVIS', 'Surveys', 'Complaints', 'Removals', 'Rematches',
                'Promotional Materials', 'Fees', 'Foreign Third Parties',
                '62.31(s)', 'Repeat Participation',
                '62.31(t)', 'Relationship to State and Local Laws', 'Preemption', 'Federalism', 'Minimum Wage', 'Overtime',
                'Unemployment Insurance', 'Employment Training Taxes',
                '62.31(u)', 'Severability',
                '62.31(v)', 'Transition Period', 'Grandfathering', 'Implementation',
            ],
        },
    }

    # Function to select directory
    def select_directory():
        directory = filedialog.askdirectory()
        directory_var.set(directory)
        logging.info(f"Selected directory: {directory}")

    # Function to start evaluation
    def start_evaluation():
        directory_path = directory_var.get()
        if not directory_path:
            output_text.insert(END, "Please select a directory containing files.\n")
            return

        # Update criteria with user inputs
        for criterion in criteria:
            if 'keywords' in criteria[criterion]:
                keywords_str = criteria_vars[criterion].get()
                criteria[criterion]['keywords'] = [kw.strip() for kw in keywords_str.split(',')]
                logging.info(f"Keywords for '{criterion}': {criteria[criterion]['keywords']}")

        # Clear output text
        output_text.delete(1.0, END)

        # Start evaluation in a separate thread
        start_evaluation_thread(directory_path, criteria, output_text)

    # Variables
    directory_var = StringVar()

    # Configure root grid
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    # Layout
    frame = Frame(root)
    frame.grid(row=0, column=0, padx=10, pady=10, sticky=(N, S, E, W))

    # Make the frame expand
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)
    frame.columnconfigure(2, weight=1)

    # Directory selection
    Label(frame, text="Select Files Directory:").grid(row=0, column=0, sticky=W)
    Entry(frame, textvariable=directory_var, width=50).grid(row=0, column=1, padx=5, sticky=(E, W))
    Button(frame, text="Browse", command=select_directory).grid(row=0, column=2)

    # Criteria keyword inputs
    criteria_vars = {}
    row_idx = 1
    for criterion in criteria:
        if 'keywords' in criteria[criterion]:
            Label(frame, text=f"{criterion} Keywords:").grid(row=row_idx, column=0, sticky=W, pady=5)
            keywords_str = ', '.join(criteria[criterion]['keywords'])
            criteria_var = StringVar(value=keywords_str)
            criteria_vars[criterion] = criteria_var
            Entry(frame, textvariable=criteria_var, width=70).grid(row=row_idx, column=1, columnspan=2, pady=5, sticky=(E, W))
            row_idx += 1

    # Start Evaluation Button
    Button(frame, text="Start Evaluation", command=start_evaluation).grid(row=row_idx, column=0, columnspan=3, pady=10)

    # Output Text Widget
    output_text = scrolledtext.ScrolledText(root, wrap='word')
    output_text.grid(row=1, column=0, padx=10, pady=10, sticky=(N, S, E, W))

    # Configure root to expand
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    root.mainloop()


if __name__ == '__main__':
    # Install required packages if not already installed
    try:
        import pdfminer
        import textstat
        import docx
        import pandas
        import nltk
        import textblob
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pdfminer.six', 'textstat', 'python-docx', 'pandas', 'nltk', 'textblob'])
        print("Packages installed. Please run the script again.")
        exit()

    create_gui()
