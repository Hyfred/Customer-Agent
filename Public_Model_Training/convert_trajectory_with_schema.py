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
    Parse a single trajectory line into structured format matching actual schema.
    """

    line = line.strip()
    if not line:
        return None

    # Extract timestamp
    timestamp_match = re.match(r'^([\d-]+\s+[\d:]+)', line)
    if not timestamp_match:
        return None
    timestamp = timestamp_match.group(1)
    remaining = line[len(timestamp):].strip()

    event = {
        'timestamp': timestamp,
        'action_type': None,
        'asin': None,           # product_id
        'product_name': None,
        'brand': None,
        'color': None,
        'product_types': None,
        'price': None,
        'search_query': None
    }

    # type action
    if remaining.lower().startswith('type [search amazon'):
        match = re.match(r'type\s*\[Search Amazon\s*\|\s*([^\]]+)\]', remaining, re.IGNORECASE)
        if match:
            event['action_type'] = 'type'
            event['search_query'] = match.group(1).strip()
            return event

    # click, Add to Cart, purchase
    action_patterns = [
        ('click', r'click\s*\[([^\]|]+)\s*\|\s*([^\]]+)\]\s*(\(.*\))?'),
        ('Add to Cart', r'click\s*\[Add to Cart\s*\|\s*([^\]]+)\]\s*(\(.*\))?'),
        ('purchase', r'purchase\s*\[([^\]|]+)\s*\|\s*([^\]]+)\]\s*(\(.*\))?')
    ]

    for action_type, pattern in action_patterns:
        match = re.match(pattern, remaining, re.IGNORECASE)
        if match:
            event['action_type'] = action_type
            if action_type == 'Add to Cart':
                event['product_name'] = match.group(1).strip()
                event['asin'] = None
                attributes_str = match.group(2)
            else:
                event['asin'] = match.group(1).strip()  # product_id
                event['product_name'] = match.group(2).strip()
                attributes_str = match.group(3) if len(match.groups()) > 2 else None

            parse_attributes(event, attributes_str)
            return event

    return None


def parse_attributes(event: Dict[str, Any], attributes_str: Optional[str]) -> None:
    """Parse optional product attributes from parentheses."""
    if not attributes_str:
        return

    attributes_str = attributes_str.strip('() ')
    if not attributes_str:
        return

    patterns = {
        'brand': r'brand:\s*([^,]+)',
        'color': r'color:\s*([^,]+)',
        'product_types': r'product_types:\s*([^,]+)',
        'price': r'price:\s*\$?([\d.]+)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, attributes_str, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key == 'price':
                try:
                    event[key] = float(value)
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
    """
    Convert list of events to a structured table format.
    
    Updated schema:
    - row_id
    - timestamp
    - action_type
    - asin / product_id
    - product_name
    - brand
    - color
    - product_types
    - price
    - search_query (only for type)
    """

    headers = [
        'row_id',
        'timestamp',
        'action_type',
        'asin',
        'product_name',
        'brand',
        'color',
        'product_types',
        'price',
        'search_query'
    ]

    rows = []
    for idx, event in enumerate(events):
        row = [
            str(idx),
            event.get('timestamp', '') or '',
            event.get('action_type', '') or '',
            event.get('asin', '') or '',
            event.get('product_name', '') or '',
            event.get('brand', '') or '',
            event.get('color', '') or '',
            event.get('product_types', '') or '',
            str(event.get('price', '')) if event.get('price') is not None else '',
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

