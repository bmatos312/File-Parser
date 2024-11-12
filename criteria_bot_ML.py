import os
import re
import logging
import threading
import tkinter as tk
from tkinter import Tk, Label, Entry, Button, filedialog, StringVar, Frame, END, W, E, S, N, scrolledtext, Toplevel, Text, Scrollbar, VERTICAL, HORIZONTAL, RIGHT, LEFT, Y, X, BOTH
import pandas as pd
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import tkinter.messagebox as messagebox
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
from fuzzywuzzy import fuzz
import sys
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and PyInstaller """
    try:
        # PyInstaller creates a temporary folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Initialize NLTK components
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Add custom negative phrases to VADER's lexicon
custom_negatives = {
    'this would ruin us': -4.0,
    'devastating': -3.5,
    'destroy': -3.5,
    'harmful': -3.0,
    'unacceptable': -3.0,
    'catastrophic': -4.0,
    'disaster': -3.5,
    'severe impact': -3.0,
    'negative effect': -2.5,
    'ruin': -3.5,
    'financial burden': -3.0,
    'detrimental': -3.0,
    'hurt': -2.5,
    'oppose': -2.5,
    'object to': -2.5,
    'serious concerns': -2.5,
    'absolutely terrible': -4.0,
    'abysmal': -4.0,
    'awful': -3.5,
    'bad': -2.0,
    'blame': -2.5,
    'broken': -2.5,
    'conflict': -3.0,
    'concerning': -2.5,
    'crisis': -3.5,
    'damaging': -3.0,
    'dangerous': -3.0,
    'degraded': -2.5,
    'defective': -2.5,
    'delayed': -2.0,
    'deny': -2.0,
    'depressed': -3.0,
    'desperate': -3.0,
    'destruction': -3.5,
    'disadvantage': -2.5,
    'discouraging': -2.5,
    'disgrace': -3.0,
    'dishonest': -2.5,
    'dislike': -2.0,
    'disruptive': -2.5,
    'dissatisfied': -2.5,
    'downturn': -3.0,
    'failure': -3.0,
    'frustrating': -2.5,
    'hurtful': -3.0,
    'inferior': -2.5,
    'insufficient': -2.5,
    'invalid': -2.5,
    'jeopardize': -3.0,
    'leak': -2.5,
    'loss': -3.0,
    'malfunction': -3.0,
    'mistake': -2.5,
    'misunderstanding': -2.5,
    'negative': -2.0,
    'poor': -2.0,
    'problem': -2.0,
    'risky': -2.5,
    'ruinous': -3.5,
    'scandal': -3.0,
    'shame': -3.0,
    'slow': -2.0,
    'stagnant': -2.5,
    'struggle': -2.5,
    'substandard': -2.5,
    'tragedy': -3.5,
    'unfortunate': -2.5,
    'unstable': -2.5,
    'unreliable': -2.5,
    'weakness': -2.5,
    'worsen': -2.5,
    'abuse': -3.0,
    'annoying': -2.0,
    'atrocious': -4.0,
    'betray': -3.0,
    'betrayal': -3.0,
    'chaos': -3.5,
    'complicated': -2.0,
    'confused': -2.0,
    'contaminate': -3.0,
    'contradict': -2.5,
    'corrupt': -3.0,
    'defame': -3.0,
    'degenerate': -3.0,
    'deny responsibility': -3.0,
    'dirty': -2.0,
    'disappointing': -3.0,
    'disarray': -3.0,
    'disgraceful': -3.5,
    'disturbing': -3.0,
    'dreadful': -3.5,
    'embarrassing': -2.5,
    'enrage': -3.0,
    'fiasco': -4.0,
    'flawed': -2.5,
    'fraud': -3.5,
    'hate': -3.0,
    'hostile': -2.5,
    'insult': -2.5,
    'irritate': -2.0,
    'mistreatment': -3.0,
    'perilous': -3.0,
    'rejected': -2.0,
    'resent': -2.0,
    'sabotage': -3.5,
    'stressed': -2.5,
    'terror': -4.0,
    'toxic': -3.0,
    'trouble': -2.5,
    'unethical': -3.5,
    'unfair': -3.0,
    'unsafe': -3.0,
    'unsatisfactory': -3.0,
    'upset': -2.5,
    'vandalize': -3.5,
    'violent': -3.5,
    'woeful': -3.5,
}

sia.lexicon.update(custom_negatives)

# Configure logging
logging.basicConfig(level=logging.INFO, filename='evaluation.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained BERT model and tokenizer
try:
    model_save_path = './best_bert_model'
    tokenizer = BertTokenizer.from_pretrained(model_save_path)
    model = BertForSequenceClassification.from_pretrained(model_save_path)
    model.to(device)
    model.eval()
    logging.info("Trained BERT model and tokenizer loaded successfully.")
    # Load label mapping
    label_mapping_path = os.path.join(model_save_path, 'label_mapping.json')
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    # Convert label_mapping keys to strings and values to integers
    label_mapping = {str(k): int(v) for k, v in label_mapping.items()}
    # Invert the label mapping
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
except Exception as e:
    logging.error(f"Error loading BERT model: {e}")
    model = None
    tokenizer = None
    label_mapping = None
    inverse_label_mapping = None

def evaluate_comment_with_ml(comment_text):
    """
    Uses the trained BERT model to classify the comment.
    """
    if model is None or tokenizer is None or inverse_label_mapping is None:
        return 'Model not available'
    try:
        # Tokenize the input text
        inputs = tokenizer.encode_plus(
            comment_text,
            add_special_tokens=True,
            max_length=128,  # Use the same MAX_LEN as during training
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predicted_class = torch.max(logits, dim=1)
            predicted_class = predicted_class.cpu().item()
        
        # Map the predicted class index to the label
        predicted_label = inverse_label_mapping[predicted_class]
        return predicted_label
    except Exception as e:
        logging.error(f"Error in evaluate_comment_with_ml: {e}")
        return 'Unknown'

def remove_email_headers(text):
    """
    Removes specified email headers, skips links, and processes the email content.
    """
    try:
        # Headers to identify and remove
        headers = [
            r'^From:.*$', 
            r'^Sent:.*$', 
            r'^To:.*$', 
            r'^Subject:.*$', 
            r'^Categories:.*$',
            r'^Cc:.*$',
            r'^Bcc:.*$',
        ]
        header_pattern = '|'.join(headers)
        
        # Remove any text before the first header occurrence
        first_header_match = re.search(header_pattern, text, flags=re.MULTILINE)
        if first_header_match:
            start_index = first_header_match.start()
            text = text[start_index:]
        
        # Remove headers
        text_without_headers = re.sub(header_pattern, '', text, flags=re.MULTILINE)
        
        # Remove links
        text_without_links = re.sub(r'https?://\S+|www\.\S+|<.*?>', '', text_without_headers)
        
        # Remove any remaining email footers or metadata
        footer_patterns = [
            r'^On.*wrote:$',  # Common email reply format
            r'^-{2,}.*Original Message.*-{2,}$',  # Email reply separator
            r'^>.*$',  # Quoted lines
            r'\[cid:.*?\]',  # Inline images or attachments
            r'&\S+;',  # HTML entities
            r'^.*unsubscribe.*$',  # Unsubscribe lines
            r'^.*To ensure receipt.*$',  # Common footer lines
            r'^.*PLEASE NOTE.*$',  # Disclaimer lines
            r'^.*This message.*$',  # Common email footers
            r'^.*Attachment:.*$',  # Attachment lines
            r'^.*E-mail Disclaimer.*$',  # Email disclaimer
            r'^(Document ID|First Name|Last Name|City|State/Province|Zip/Postal Code|Country|Comment):?.*$',  # Field names
        ]
        footer_pattern = '|'.join(footer_patterns)
        clean_text = re.sub(footer_pattern, '', text_without_links, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove extra whitespace and blank lines
        clean_text = re.sub(r'\n\s*\n', '\n', clean_text)
        
        return clean_text.strip()
    except Exception as e:
        logging.error(f"Error in remove_email_headers: {e}")
        return text.strip()

def extract_comments_from_docx(docx_path):
    """
    Extracts comments from a Word (.docx) file.
    Splits the text at 'From:' and processes each email separately.
    """
    try:
        from docx import Document
        doc = Document(docx_path)
        full_text = []
    
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                full_text.append(text)
    
        text = '\n'.join(full_text)
    
        # Split the text into individual emails/comments at "From:"
        email_split_pattern = r'(?=^From:)'  # Splits at 'From:' that appears at the beginning of a line
        emails = re.split(email_split_pattern, text, flags=re.MULTILINE)
        comments = []
    
        for idx, email_text in enumerate(emails):
            email_text = email_text.strip()
            if email_text:
                logging.info(f"Processing email {idx+1} in {docx_path}")
                # Remove headers and metadata
                email_text = remove_email_headers(email_text)
                # Add the cleaned email text as a comment if it's not empty
                if email_text:
                    comments.append(email_text)
            else:
                logging.info(f"Skipped empty email at index {idx} in {docx_path}")
        logging.info(f"Extracted {len(comments)} comments from {docx_path}")
        return comments
    except Exception as e:
        logging.error(f"Error extracting text from {docx_path}: {e}")
        return []

def extract_comments_from_pdf(pdf_path):
    """
    Extracts comments from a PDF file.
    Splits the text at 'From:' and processes each email separately.
    """
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path)
    
        # Split the text into individual emails/comments at "From:"
        email_split_pattern = r'(?=^From:)'  # Splits at 'From:' that appears at the beginning of a line
        emails = re.split(email_split_pattern, text, flags=re.MULTILINE)
        comments = []
    
        for idx, email_text in enumerate(emails):
            email_text = email_text.strip()
            if email_text:
                logging.info(f"Processing email {idx+1} in {pdf_path}")
                # Remove headers and metadata
                email_text = remove_email_headers(email_text)
                # Add the cleaned email text as a comment if it's not empty
                if email_text:
                    comments.append(email_text)
            else:
                logging.info(f"Skipped empty email at index {idx} in {pdf_path}")
        logging.info(f"Extracted {len(comments)} comments from {pdf_path}")
        return comments
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return []

def evaluate_comment_locally(comment_text, criteria, comment_id, filename, provision_keywords, custom_negative_phrases):
    """
    Evaluates a policy comment based on the specified criteria.
    Returns the evaluation data as a dictionary.
    """
    try:
        # The maximum score is the number of criteria
        max_score = len(criteria)
        score = 0

        # Split comment into sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', comment_text.strip())
        total_sentences = len(sentences)

        # Evaluate each criterion
        evaluation_data = {}

        keyword_frequency = {}

        for criterion, details in criteria.items():
            details['score'] = 0  # Reset score
            details['sentences'] = set()  # Use a set to store unique sentences

            if criterion == 'Writing Quality':
                # Use readability score and sentence length as proxies for writing quality
                try:
                    from textstat import flesch_reading_ease
                    readability = flesch_reading_ease(comment_text)
                except Exception as e:
                    logging.error(f"Error calculating readability: {e}")
                    readability = 0  # If calculation fails, set to 0
                avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
                if readability >= 60 and avg_sentence_length <= 20:
                    details['score'] = 1
                # Record whether the criterion was met
                evaluation_data[criterion] = 'Yes' if details.get('score', 0) else 'No'
                keyword_frequency[criterion] = f"Readability: {readability:.2f}, Avg Sentence Length: {avg_sentence_length:.2f}"
            else:
                for sentence in sentences:
                    for keyword in details['keywords']:
                        if keyword.lower() in sentence.lower():
                            details['sentences'].add(sentence.strip())
                # Calculate the percentage of sentences containing keywords
                keyword_count = len(details['sentences'])
                if total_sentences > 0:
                    keyword_percentage = (keyword_count / total_sentences) * 100
                else:
                    keyword_percentage = 0
                keyword_frequency[criterion] = f"{keyword_percentage:.2f}%"
                # Decide if the criterion is met based on a threshold, e.g., 5%
                if keyword_percentage >= 5:
                    details['score'] = 1
                # Record whether the criterion was met
                evaluation_data[criterion] = 'Yes' if details.get('score', 0) else 'No'

        # Calculate total score
        score = sum(details.get('score', 0) for details in criteria.values())

        # Ensure the score does not exceed max_score
        score = min(score, max_score)

        evaluation_data['Quality Score'] = score

        # Sentiment Analysis using VADER on the whole comment text
        sentiment_scores = sia.polarity_scores(comment_text)
        compound_score = sentiment_scores['compound']

        # Check for custom negative phrases
        negative_flag = False
        for phrase in custom_negative_phrases:
            if phrase in comment_text.lower():
                negative_flag = True
                break

        if compound_score >= 0.05 and not negative_flag:
            evaluation_data['Sentiment'] = 'Positive'
        elif compound_score <= -0.05 or negative_flag:
            evaluation_data['Sentiment'] = 'Negative'
        else:
            evaluation_data['Sentiment'] = 'Neutral'

        # Extract Keywords (Top 5 most frequent words excluding stopwords)
        words = re.findall(r'\b\w+\b', comment_text.lower())
        words = [word for word in words if word not in stop_words]
        word_counts = pd.Series(words).value_counts()
        top_keywords = word_counts.head(5).index.tolist()
        evaluation_data['Keywords'] = ', '.join(top_keywords)

        # Keyword Frequency Counter
        keyword_freq_str = ', '.join(f"{k}: {v}" for k, v in keyword_frequency.items())
        evaluation_data['Keyword Frequency'] = keyword_freq_str if keyword_freq_str else 'N/A'

        # Themes (Placeholder for your own theme extraction logic)
        themes = []
        theme_keywords = {
            'Housing and Financial Deductions': [
                'housing costs', 'deductions', 'regional disparities', 'financial expectations'
            ],
            'Program Costs and Childcare Accessibility': [
                'program costs', 'childcare shortage', 'accessibility', 'rising costs'
            ],
            'Scheduling and Flexibility': [
                'scheduling flexibility', 'administrative burden', 'inflexible scheduling', 'adjustments'
            ],
            'Worker Protections and Financial Expectations': [
                'worker protections', 'fair wage', 'financial expectations', 'adequate protections'
            ]
        }
        themes = []
        for theme, keywords in theme_keywords.items():
            if any(keyword in comment_text.lower() for keyword in keywords):
                themes.append(theme)
        evaluation_data['Themes'] = ', '.join(set(themes)) if themes else 'N/A'


            # Use the ML model to get the classification
        ml_classification = evaluate_comment_with_ml(comment_text)
        evaluation_data['ML Classification'] = ml_classification

        # Determine Relevant Provisions
        relevant_provisions = determine_relevant_provisions(comment_text, provision_keywords)
        evaluation_data['Relevant Provisions'] = ', '.join(relevant_provisions) if relevant_provisions else 'N/A'

        # Add other fields
        evaluation_data['Comment ID'] = comment_id
        evaluation_data['Text'] = comment_text
        evaluation_data['Source'] = filename

        # Email Excerpt (First five sentences)
        email_excerpt_sentences = re.split(r'(?<=[.!?])\s+', comment_text.strip())
        email_excerpt = ' '.join(email_excerpt_sentences[:5])
        evaluation_data['Email Excerpt'] = email_excerpt

        return evaluation_data
    except Exception as e:
        logging.error(f"Error in evaluate_comment_locally for Comment ID {comment_id}: {e}")
        return None

def determine_relevant_provisions(comment_text, provision_keywords):
    """
    Determines which provisions are relevant to the comment text based on keywords using fuzzy matching.
    """
    relevant_provisions = set()
    try:
        comment_text_lower = comment_text.lower()
        for provision, keywords in provision_keywords.items():
            for keyword in keywords:
                # Use fuzzy matching to compare keyword and comment text
                score = fuzz.partial_ratio(keyword.lower(), comment_text_lower)
                if score >= 80:  # Threshold can be adjusted
                    relevant_provisions.add(provision)
                    break  # Move to next provision if a match is found
        return sorted(relevant_provisions)
    except Exception as e:
        logging.error(f"Error in determine_relevant_provisions: {e}")
        return []

def process_files_in_directory(directory_path, criteria, output_text_widget, provision_keywords, custom_negative_phrases):
    """
    Processes all files in the given directory and updates the output_text_widget with evaluations.
    """
    evaluation_results = []
    comment_id = 1
    try:
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            if os.path.isfile(filepath):
                comments = []
                if filename.lower().endswith('.pdf'):
                    output_text_widget.insert(END, f"Processing file: {filename}\n\n")
                    comments = extract_comments_from_pdf(filepath)
                elif filename.lower().endswith('.docx'):
                    output_text_widget.insert(END, f"Processing file: {filename}\n\n")
                    comments = extract_comments_from_docx(filepath)
                else:
                    continue  # Skip unsupported file types

                for idx, comment_text in enumerate(comments):
                    output_text_widget.insert(END, f"Evaluating comment {idx+1}/{len(comments)} in {filename}\n")
                    try:
                        evaluation_data = evaluate_comment_locally(comment_text, criteria, comment_id, filename, provision_keywords, custom_negative_phrases)
                        if evaluation_data:
                            # Display in GUI
                            output_text_widget.insert(END, f"Comment ID: {comment_id}\n")
                            output_text_widget.insert(END, f"Excerpt: {evaluation_data['Email Excerpt']}...\n")
                            output_text_widget.insert(END, f"ML Classification: {evaluation_data['ML Classification']}\n")
                            output_text_widget.insert(END, f"Relevant Provisions: {evaluation_data['Relevant Provisions']}\n")
                            output_text_widget.insert(END, f"Score: {evaluation_data['Quality Score']}\n")
                            output_text_widget.insert(END, f"Sentiment: {evaluation_data['Sentiment']}\n")
                            output_text_widget.insert(END, '-' * 80 + '\n')
                            # Collect evaluation data
                            evaluation_results.append(evaluation_data)
                            comment_id += 1
                        else:
                            logging.warning(f"No evaluation data for comment {idx+1} in {filename}")
                    except Exception as e:
                        logging.error(f"Error evaluating comment {idx+1} in {filename}: {e}")
                        output_text_widget.insert(END, f"Error evaluating comment {idx+1} in {filename}: {e}\n")
                        output_text_widget.insert(END, f"Comment content: {comment_text[:200]}...\n")
                        output_text_widget.insert(END, '-' * 80 + '\n')
                    output_text_widget.update()
        # Check if evaluation_results has data
        if evaluation_results:
            # After processing all files, export results to Excel and CSV
            export_results(evaluation_results)
            output_text_widget.insert(END, "Evaluation completed successfully.\n")
            output_text_widget.insert(END, f"Results have been saved to 'evaluation_results.xlsx' and 'evaluation_results.csv'.\n")
        else:
            output_text_widget.insert(END, "No evaluation results to export.\n")
    except Exception as e:
        logging.error(f"Error processing files in directory: {e}")
        output_text_widget.insert(END, f"Error processing files: {e}\n")

def export_results(evaluation_results):
    """
    Exports the evaluation results to an Excel and CSV file.
    """
    if not evaluation_results:
        logging.warning("No evaluation results to export.")
        return
    try:
        df = pd.DataFrame(evaluation_results)
        # Rearranging columns to match the desired output
        columns_order = [
            'Comment ID', 'Email Excerpt', 'Text', 'ML Classification', 'Relevant Provisions', 'Themes', 'Sentiment', 'Quality Score', 'Keyword Frequency',
            'Stakeholder', 'Market', 'Priority', 'Keywords', 'Recommendations', 'Source',
        ]
        # Ensure all columns are present
        available_columns = df.columns.tolist()
        columns_to_use = [col for col in columns_order if col in available_columns]
        df = df[columns_to_use]

        # Export to Excel
        df.to_excel('evaluation_results.xlsx', index=False)

        # Export to CSV
        df.to_csv('evaluation_results.csv', index=False)
    except Exception as e:
        logging.error(f"Error exporting results: {e}")

def start_evaluation_thread(directory_path, criteria, output_text_widget, provision_keywords, custom_negative_phrases):
    """
    Starts the evaluation in a separate thread to keep the GUI responsive.
    """
    threading.Thread(target=process_files_in_directory, args=(directory_path, criteria, output_text_widget, provision_keywords, custom_negative_phrases)).start()

def create_gui():
    """
    Creates the GUI interface.
    """
    root = Tk()
    root.title("Policy Comment Evaluator")
    root.geometry("900x700")

    # Criteria and their default keywords
    criteria = {
        'Clear Cause-and-Effect Reasoning': {
            'description': 'Comments that clearly explain how specific policy changes might lead to anticipated outcomes.',
            'keywords': ['because', 'therefore', 'thus', 'hence', 'consequently', 'as a result', 'lead to', 'result in', 'due to', 'since'],
        },
        'Constructive Criticism': {
            'description': 'Detailed critiques or feedback that provide actionable insights or suggestions for improvement.',
            'keywords': ['I suggest', 'recommend', 'advise', 'propose', 'should', 'could', 'improve', 'enhance', 'modify', 'consider'],
        },
        'Writing Quality': {
            'description': 'Comments written in a clear, concise, and well-organized manner.',
        },
        'Objective Data and Facts': {
            'description': 'Comments that include factual evidence, specific figures, or credible references to support arguments.',
            'keywords': ['data', 'evidence', 'statistics', 'research', 'study', 'percent', 'number', 'figure', 'according to', 'report'],
        },
    }

    # Custom negative phrases
    custom_negative_phrases = [
        'this would ruin us',
        'devastating',
        'destroy',
        'harmful',
        'unacceptable',
        'catastrophic',
        'disaster',
        'severe impact',
        'negative effect',
        'ruin',
        'financial burden',
        'detrimental',
        'hurt',
        'oppose',
        'object to',
        'serious concerns',
    ]

    # Provision keywords mapping (will be editable via GUI)
    provision_keywords = {
        'Section 62.31(a)': ['au pairs', 'purpose', 'part-time program', 'full-time program', 'program goals', 'objectives', 'mission'],
        # ... (other provisions)
        'Section 62.31(v)': ['transition period', 'grandfathering', 'implementation', 'effective date', 'phasing in'],
    }

    # Function to select directory
    def select_directory():
        directory = filedialog.askdirectory()
        directory_var.set(directory)

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

        # Update provision keywords with user inputs
        provision_keywords_str = provision_text.get("1.0", END).strip()
        if provision_keywords_str:
            try:
                # Parse the provision keywords from the text
                new_provision_keywords = {}
                lines = provision_keywords_str.split('\n')
                for line in lines:
                    if ':' in line:
                        provision, keywords_str = line.split(':', 1)
                        keywords = [kw.strip() for kw in keywords_str.split(',')]
                        new_provision_keywords[provision.strip()] = keywords
                # Update the provision_keywords dictionary
                provision_keywords.clear()
                provision_keywords.update(new_provision_keywords)
            except Exception as e:
                messagebox.showerror("Error", f"Error parsing provisions: {e}")
                return

        # Clear output text
        output_text.delete(1.0, END)

        # Start evaluation in a separate thread
        start_evaluation_thread(directory_path, criteria, output_text, provision_keywords, custom_negative_phrases)

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

    # Provisions Editing Section
    Label(frame, text="Edit Provisions and Keywords (Format: Provision: keyword1, keyword2, ...)").grid(row=row_idx, column=0, columnspan=3, sticky=W, pady=5)
    row_idx += 1
    provision_text = Text(frame, height=10, width=70)
    provision_text.grid(row=row_idx, column=0, columnspan=3, pady=5, sticky=(E, W))
    # Insert the existing provisions into the text widget
    provision_lines = []
    for provision, keywords in provision_keywords.items():
        provision_line = f"{provision}: {', '.join(keywords)}"
        provision_lines.append(provision_line)
    provision_text.insert(END, '\n'.join(provision_lines))

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
        import nltk
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pdfminer.six', 'textstat', 'python-docx', 'nltk', 'fuzzywuzzy[accelerate]'])
        print("Packages installed. Please run the script again.")
        exit()

    create_gui()

