"""
Convert pgeval-benchmark trajectory data to NeoBinder format with exact schema.

This version uses the exact schema specified for retrieval-focused questions.
"""

import json
import re
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional


def parse_trajectory_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single trajectory line into structured format matching the exact schema.
    
    Schema:
    - timestamp TIMESTAMP
    - action_type VARCHAR(50): 'type', 'click', 'Add to Cart', 'purchase'
    - asin VARCHAR(255)
    - product_name TEXT
    - brand VARCHAR(255)
    - product_type VARCHAR(100)
    - price DECIMAL(10,2)
    - prime BOOLEAN
    - review_rating DECIMAL(2,1)
    - review_count INT
    - search_query TEXT (only for action_type='type')
    """
    line = line.strip()
    if not line:
        return None
    
    # Extract timestamp (first part before action type)
    timestamp_match = re.match(r'^([\d-]+\s+[\d:]+)', line)
    if not timestamp_match:
        return None
    
    timestamp = timestamp_match.group(1)
    remaining = line[len(timestamp):].strip()
    
    event = {
        'timestamp': timestamp,
        'action_type': None,
        'asin': None,
        'product_name': None,
        'brand': None,
        'product_type': None,
        'price': None,
        'prime': None,
        'review_rating': None,
        'review_count': None,
        'search_query': None
    }
    
    # Parse different action types
    if remaining.startswith('type [Search Amazon'):
        # type [Search Amazon | query]
        match = re.match(r'type \[Search Amazon \| ([^\]]+)\]', remaining)
        if match:
            event['action_type'] = 'type'
            event['search_query'] = match.group(1).strip()
            return event
    
    elif remaining.startswith('click [asin'):
        # click [asin | product_name] (attributes)
        match = re.match(r'click \[asin \| ([^\]]+)\](.*)', remaining)
        if match:
            event['action_type'] = 'click'
            event['product_name'] = match.group(1).strip()
            # Extract ASIN from product name if it's an actual ASIN pattern
            # For now, we'll use a hash of the product name as asin
            event['asin'] = f"ASIN_{hash(event['product_name']) % 10000000:07d}"
            attributes = match.group(2).strip()
            parse_attributes(event, attributes)
            return event
    
    elif remaining.startswith('click [Add to Cart'):
        # click [Add to Cart | product_name] (attributes)
        match = re.match(r'click \[Add to Cart \| ([^\]]+)\](.*)', remaining)
        if match:
            event['action_type'] = 'Add to Cart'
            event['product_name'] = match.group(1).strip()
            event['asin'] = f"ASIN_{hash(event['product_name']) % 10000000:07d}"
            attributes = match.group(2).strip()
            parse_attributes(event, attributes)
            return event
    
    elif remaining.startswith('purchase [asin'):
        # purchase [asin | product_name] (attributes)
        match = re.match(r'purchase \[asin \| ([^\]]+)\](.*)', remaining)
        if match:
            event['action_type'] = 'purchase'
            event['product_name'] = match.group(1).strip()
            event['asin'] = f"ASIN_{hash(event['product_name']) % 10000000:07d}"
            attributes = match.group(2).strip()
            parse_attributes(event, attributes)
            return event
    
    return None


def parse_attributes(event: Dict[str, Any], attributes_str: str) -> None:
    """Parse product attributes from the parentheses."""
    if not attributes_str or attributes_str == '()':
        return
    
    # Remove outer parentheses
    attributes_str = attributes_str.strip('() ')
    
    # Parse key-value pairs
    patterns = {
        'brand': r'brand:\s*([^,]+)',
        'product_type': r'product type:\s*([^,]+)',
        'price': r'price:\s*([\d.]+)',
        'prime': r'prime:\s*(True|False)',
        'review_rating': r'review rating:\s*([\d.]+)',
        'review_count': r'review count:\s*(\d+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, attributes_str, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key == 'prime':
                event[key] = value == 'True'
            elif key in ['price', 'review_rating']:
                try:
                    event[key] = float(value)
                except:
                    event[key] = None
            elif key == 'review_count':
                try:
                    event[key] = int(value)
                except:
                    event[key] = None
            else:
                event[key] = value


def parse_trajectory_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse entire trajectory file into list of events."""
    events = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            event = parse_trajectory_line(line)
            if event:
                events.append(event)
    
    return events


def events_to_table(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert list of events to NeoBinder table format with exact schema."""
    # Define table headers matching the schema
    headers = [
        'row_id',
        'timestamp',
        'action_type',
        'asin',
        'product_name',
        'brand',
        'product_type',
        'price',
        'prime',
        'review_rating',
        'review_count',
        'search_query'
    ]
    
    # Convert events to rows
    rows = []
    for idx, event in enumerate(events):
        row = [
            str(idx),
            event.get('timestamp', ''),
            event.get('action_type', ''),
            event.get('asin', '') or '',
            event.get('product_name', '') or '',
            event.get('brand', '') or '',
            event.get('product_type', '') or '',
            str(event.get('price', '')) if event.get('price') is not None else '',
            str(event.get('prime', '')) if event.get('prime') is not None else '',
            str(event.get('review_rating', '')) if event.get('review_rating') is not None else '',
            str(event.get('review_count', '')) if event.get('review_count') is not None else '',
            event.get('search_query', '') or ''
        ]
        rows.append(row)
    
    table = {
        'page_title': 'Customer Shopping Trajectory',
        'header': headers,
        'rows': rows
    }
    
    return table


def convert_trajectory_to_neobinder(
    input_file: str,
    output_file: str,
    trajectory_id: str = None,
    questions: List[Dict[str, str]] = None
):
    """
    Convert trajectory file to NeoBinder format with custom questions.
    
    Args:
        input_file: Path to trajectory text file
        output_file: Path to output JSON file
        trajectory_id: Optional ID for the trajectory
        questions: List of dicts with 'question' and 'answer' keys
    """
    # Parse trajectory
    events = parse_trajectory_file(input_file)
    
    print(f"Parsed {len(events)} events from trajectory")
    
    # Convert to table
    table = events_to_table(events)
    
    # Create NeoBinder format data items
    data_items = []
    
    if questions:
        for idx, q in enumerate(questions):
            data_item = {
                'id': f"{trajectory_id or 'trajectory'}_{idx}",
                'question': q['question'],
                'table': table,
                'answer_text': [q['answer']]
            }
            data_items.append(data_item)
    else:
        # Create a placeholder
        data_item = {
            'id': trajectory_id or 'trajectory_0',
            'question': 'placeholder question',
            'table': table,
            'answer_text': ['placeholder answer']
        }
        data_items.append(data_item)
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_items, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(data_items)} data items to {output_file}")
    print(f"\nTable has {len(table['rows'])} rows and {len(table['header'])} columns")
    
    return data_items, events


def main():
    parser = argparse.ArgumentParser(
        description='Convert trajectory data to NeoBinder format with exact schema'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input trajectory file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path'
    )
    parser.add_argument(
        '--trajectory-id',
        type=str,
        default='trajectory',
        help='Trajectory ID prefix'
    )
    parser.add_argument(
        '--questions-file',
        type=str,
        help='Optional JSON file with custom questions'
    )
    
    args = parser.parse_args()
    
    # Load questions if provided
    questions = None
    if args.questions_file:
        with open(args.questions_file, 'r') as f:
            questions = json.load(f)
    
    convert_trajectory_to_neobinder(
        input_file=args.input,
        output_file=args.output,
        trajectory_id=args.trajectory_id,
        questions=questions
    )


if __name__ == '__main__':
    main()
