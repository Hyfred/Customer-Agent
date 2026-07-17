#!/usr/bin/env python3
"""
Robust benchmark question generator that ensures we get exactly the target number of questions.
"""

import json
import argparse
import time
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import sys
import os

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_structured_benchmark import (
    create_structured_benchmark_prompt,
    validate_sql_query,
    calculate_evidence_depth,
    analyze_question_distribution
)
from openai import OpenAI
from convert_trajectory_with_schema import parse_trajectory_line, events_to_table


def robust_call_gpt_oss_api(prompt: str, model_id: str = "openai/gpt-oss-120b", max_retries: int = 3) -> str:
    """Call GPT-OSS API with robust error handling and JSON enforcement."""
    # client = OpenAI(
    #     base_url="http://localhost:8000/v1",
    #     api_key="EMPTY"
    # )
    client = OpenAI(
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1", 
        api_key=os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "") # Replace with actual API key
    )
    
    # Enhanced system prompt to enforce JSON-only output
    system_prompt = """You are a JSON generation expert. Your ONLY job is to return valid JSON arrays.

CRITICAL RULES:
1. Return ONLY the JSON array - no explanations, no reasoning, no markdown fences
2. Do not include any text before or after the JSON
3. Ensure the JSON is properly formatted and valid
4. If you cannot generate valid JSON, return an empty array: []

Example of correct output:
[{"question": "What was the first product?", "sql": "SELECT * FROM events LIMIT 1", "answer": "Product A", "difficulty": "Easy"}]

Do NOT include:
- ```json or ``` fences
- Explanations like "Here are the questions:"
- Any text outside the JSON array
- Reasoning or analysis

Return ONLY the JSON array now:"""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openai.gpt-oss-120b-1:0",#model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.1,  # Lower temperature for more consistent JSON
                top_p=0.9
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
            else:
                print(f"All {max_retries} API attempts failed")
                return ""
    
    return ""



def robust_parse_json_response(response: str) -> List[Dict[str, Any]]:
    """Robustly parse JSON response with multiple fallback strategies."""
    if not response or not response.strip():
        return []
    
    # Clean the response
    response_clean = response.strip()
    
    # Remove markdown code fences
    import re
    response_clean = re.sub(r'^```(?:json)?\s*', '', response_clean)
    response_clean = re.sub(r'\s*```\s*$', '', response_clean)
    
    # Strategy 1: Direct JSON parsing
    try:
        parsed = json.loads(response_clean)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            # Check for common wrapper keys
            for key in ['questions', 'items', 'data', 'results']:
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON array using regex patterns
    json_patterns = [
        r'(\[.*?\])',  # Simple array
        r'(\{.*?"questions".*?\})',  # Object with questions key
        r'(\[.*?\])(?=\s*$)',  # Array at end of string
        r'(\[.*?\])(?=\s*\n)',  # Array followed by newline
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response_clean, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list) and len(parsed) > 0:
                    print(f"✓ Extracted JSON array with {len(parsed)} items using pattern: {pattern[:20]}...")
                    return parsed
                elif isinstance(parsed, dict):
                    for key in ['questions', 'items', 'data', 'results']:
                        if key in parsed and isinstance(parsed[key], list):
                            print(f"✓ Extracted questions from object with {len(parsed[key])} items")
                            return parsed[key]
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Try to fix common JSON issues
    try:
        # Remove trailing commas
        fixed_response = re.sub(r',\s*}', '}', response_clean)
        fixed_response = re.sub(r',\s*]', ']', fixed_response)
        
        # Try parsing the fixed version
        parsed = json.loads(fixed_response)
        if isinstance(parsed, list):
            print(f"✓ Fixed and parsed JSON array with {len(parsed)} items")
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Extract individual JSON objects and combine
    try:
        # Look for individual objects in the response
        object_pattern = r'\{[^{}]*"question"[^{}]*\}'
        matches = re.findall(object_pattern, response_clean, re.DOTALL)
        if matches:
            objects = []
            for match in matches:
                try:
                    obj = json.loads(match)
                    objects.append(obj)
                except json.JSONDecodeError:
                    continue
            if objects:
                print(f"✓ Extracted {len(objects)} individual JSON objects")
                return objects
    except Exception:
        pass
    
    print(f"✗ Could not parse JSON from response (first 200 chars): {response[:200]}")
    return []


def detect_question_attributes(question: Dict[str, Any]) -> List[str]:
    """Detect which attributes a question primarily queries."""
    import re
    sql = question.get('sql', '').lower()
    q_text = question.get('question', '').lower()
    
    attributes = []
    
    # Check SQL SELECT and WHERE clauses for attributes
    if 'product_name' in sql or 'product name' in q_text:
        attributes.append('product_name')
    if 'brand' in sql or 'brand' in q_text:
        attributes.append('brand')
    if 'timestamp' in sql or 'time' in q_text or 'when' in q_text:
        attributes.append('timestamp')
    if 'search_query' in sql or 'search' in q_text or 'query' in q_text:
        attributes.append('search_query')
    if 'price' in sql or 'price' in q_text or 'expensive' in q_text or 'cost' in q_text:
        attributes.append('price')
    if 'review_rating' in sql or 'rating' in q_text:
        attributes.append('review_rating')
    if 'review_count' in sql or 'review count' in q_text or 'reviews' in q_text:
        attributes.append('review_count')
    if 'prime' in sql or 'prime' in q_text:
        attributes.append('prime')
    if 'product_type' in sql or 'product type' in q_text or 'category' in q_text:
        attributes.append('product_type')
    if 'asin' in sql:
        attributes.append('asin')
    if 'action_type' in sql or 'click' in q_text or 'purchase' in q_text or 'cart' in q_text:
        attributes.append('action_type')
    if 'row_id' in sql or 'row id' in q_text:
        attributes.append('row_id')
    
    return attributes if attributes else ['unknown']


def slice_events_by_depth(events: List[Dict], depth_bucket: str) -> List[Dict]:
    """Slice events based on evidence depth bucket."""
    total = len(events)
    if total == 0:
        return events
    
    if depth_bucket == 'early':
        # First 1/3 of events
        end_idx = max(1, total // 3)
        return events[:end_idx]
    elif depth_bucket == 'middle':
        # Middle 1/3 of events
        start_idx = total // 3
        end_idx = 2 * total // 3
        return events[start_idx:end_idx] if start_idx < end_idx else events[start_idx:start_idx+1]
    elif depth_bucket == 'late':
        # Last 1/3 of events
        start_idx = 2 * total // 3
        return events[start_idx:]
    else:
        return events


def robust_generate_questions_for_trajectory(
    trajectory_data: Dict[str, Any],
    trajectory_index: int,
    num_questions: int,
    difficulty_distribution: Dict[str, float],
    trajectory_mode: str,
    model_id: str = "openai/gpt-oss-120b",
    max_retries: int = 5,
    selection_state: Dict[str, Any] | None = None,
    balanced_depth: bool = False,
    batch_depth_quota: Dict[str, int] | None = None,
) -> List[Dict[str, Any]]:
    """Generate questions for a single trajectory with retry logic."""
    
    print(f"\n{'='*60}")
    print(f"Processing Trajectory {trajectory_index + 1}")
    print(f"Customer ID: {trajectory_data.get('cid', 'unknown')}")
    print(f"Trajectory Mode: {trajectory_mode}")
    print(f"{'='*60}")
    
    # Select trajectory text based on mode
    if trajectory_mode == "128k":
        trajectory_text = trajectory_data.get('trajectory_full', '')
    else:
        trajectory_text = trajectory_data.get('trajectory', '')
    
    print(f"Trajectory length: {len(trajectory_text)} characters")
    
    # Convert trajectory to table format (full trajectory for context/validation)
    lines = trajectory_text.strip().split('\n')
    events = []
    for line in lines:
        event = parse_trajectory_line(line)
        if event:
            events.append(event)
    
    full_table = events_to_table(events)
    print(f"✓ Converted trajectory to table with {len(full_table['rows'])} rows")
    
    # Generate questions with retry logic
    for attempt in range(max_retries):
        print(f"\nAttempt {attempt + 1}/{max_retries}")
        
        try:
            # If balanced_depth, generate separately for each bucket using sliced trajectory
            if balanced_depth and batch_depth_quota is not None:
                all_batch_questions = []
                
                for bucket, count in batch_depth_quota.items():
                    if count <= 0:
                        continue
                    
                    # Slice events to the bucket's portion
                    sliced_events = slice_events_by_depth(events, bucket)
                    sliced_table = events_to_table(sliced_events)
                    
                    print(f"  Generating {count} question(s) for '{bucket}' bucket ({len(sliced_table['rows'])} rows)")
                    
                    # Create prompt with sliced trajectory
                    prompt_sliced = create_structured_benchmark_prompt(sliced_table, count, difficulty_distribution)
                    prompt_sliced += f"""

EVIDENCE DEPTH REQUIREMENT:
===========================
Generate questions whose evidence appears in this slice of the trajectory.
This slice represents the {bucket.upper()} portion of the customer's journey.
"""
                    
                    # Generate for this bucket
                    response_sliced = robust_call_gpt_oss_api(prompt_sliced, model_id=model_id, max_retries=2)
                    if response_sliced:
                        bucket_questions = robust_parse_json_response(response_sliced)
                        if bucket_questions:
                            all_batch_questions.extend(bucket_questions)
                            print(f"  ✓ Parsed {len(bucket_questions)} question(s) for '{bucket}' bucket")
                        else:
                            print(f"  ✗ Could not parse questions for '{bucket}' bucket")
                    else:
                        print(f"  ✗ Empty response for '{bucket}' bucket")
                
                questions = all_batch_questions
                print(f"✓ Total parsed: {len(questions)} questions across all buckets")
            else:
                # No balanced depth - use full trajectory
                prompt = create_structured_benchmark_prompt(full_table, num_questions, difficulty_distribution)
                response = robust_call_gpt_oss_api(prompt, model_id=model_id, max_retries=3)
                print("✓ Generated questions via LLM")
                
                if not response:
                    print("✗ Error: Empty response from LLM")
                    if attempt < max_retries - 1:
                        print("Retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                    else:
                        return []
                
                questions = robust_parse_json_response(response)
            
            if not questions:
                print(f"✗ Could not parse valid JSON from response")
                if attempt < max_retries - 1:
                    print("Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    return []
            
            print(f"✓ Parsed {len(questions)} questions")
            
            if len(questions) == 0:
                print("✗ No questions generated")
                if attempt < max_retries - 1:
                    print("Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    return []
            
            # Validate SQL queries against FULL trajectory (even though questions were generated from slices)
            print("Validating SQL queries against full trajectory...")
            import sqlite3
            import tempfile
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
                temp_db = f.name
            
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()

            # breakpoint()
            
            # Create events table with correct schema (12 columns including row_id)
            cursor.execute('''
                CREATE TABLE events (
                    row_id TEXT,
                    timestamp TEXT,
                    action_type TEXT,
                    asin TEXT,
                    product_name TEXT,
                    brand TEXT,
                    color TEXT,
                    product_type TEXT,
                    price REAL,
                    search_query TEXT
                )
            ''')

            # breakpoint()
            
            # Insert FULL trajectory data for validation
            for row in full_table['rows']:
                cursor.execute('''
                    INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', row)
            
            validated_questions = []
            for i, q in enumerate(questions):
                try:
                    is_valid, result = validate_sql_query(conn, q['sql'])
                    
                    if is_valid:
                        q['sql_valid'] = True
                        q['sql_result'] = str(result)
                        # Use SQL execution result as the ground truth answer
                        if result and len(result) > 0:
                            # Extract the actual value from SQL result
                            if len(result) == 1 and len(result[0]) == 1:
                                # Single value result
                                sql_answer = str(result[0][0])
                            else:
                                # Multiple values or complex result - use the first value
                                sql_answer = str(result[0][0]) if result[0] else ""
                            
                            # Update the answer with SQL result
                            q['answer'] = sql_answer
                            q['validated_answer'] = sql_answer
                            print(f"  ✓ Question {i+1}: SQL valid, answer updated to: {sql_answer}")
                            validated_questions.append(q)
                        else:
                            # SQL returned no results - DROP this question as it's invalid
                            print(f"  ✗ Question {i+1}: SQL valid but no results - DROPPING question")
                            print(f"    Original question: {q.get('question', '')[:80]}...")
                            print(f"    Original answer: {q.get('answer', '')}")
                            # Don't add to validated_questions - question is dropped
                    else:
                        print(f"  ✗ Question {i+1}: SQL invalid - {result}")
                except Exception as sql_error:
                    print(f"  ✗ Question {i+1}: SQL validation error - {sql_error}")
                    # Still add the question but mark it as having SQL issues
                    q['sql_valid'] = False
                    q['sql_error'] = str(sql_error)
                    q['validated_answer'] = q['answer']
                    validated_questions.append(q)
            
            conn.close()
            os.unlink(temp_db)
            
            print(f"✓ Validated {len(validated_questions)}/{len(questions)} questions")
            
            # Helper functions for selection
            def depth_bucket(d: float) -> str:
                if d < 0.33:
                    return 'early'
                if d < 0.67:
                    return 'middle'
                return 'late'

            # No hard type categorization/caps – we aim for natural diversity from the generator

            # Calculate evidence depth early for selection and attach metadata
            customer_id = trajectory_data.get('cid', f'customer_{trajectory_index}')
            for q in validated_questions:
                q['customer_id'] = customer_id
                q['trajectory_index'] = trajectory_index
                q['original_pgeval_index'] = trajectory_index
                q['trajectory_mode'] = trajectory_mode
                q['evidence_depth'] = calculate_evidence_depth(q, trajectory_data)

            # If no balancing/diversity requested, fast path
            if not balanced_depth:
                selected = validated_questions[:num_questions]
                print(f"✓ Generated {len(selected)} questions for trajectory {trajectory_index + 1}")
                return selected

            # Otherwise, select respecting quotas from selection_state and this batch's depth quota
            selected: List[Dict[str, Any]] = []
            batch_counts = {'early': 0, 'middle': 0, 'late': 0}
            for q in validated_questions:
                if len(selected) >= num_questions:
                    break
                ok = True
                if balanced_depth and selection_state is not None:
                    b = depth_bucket(q.get('evidence_depth', 0.0))
                    # Enforce global remaining quota
                    if selection_state['depth_counts'][b] >= selection_state['depth_quota'][b]:
                        ok = False
                    # Enforce per-batch quota if provided
                    if ok and batch_depth_quota is not None:
                        if batch_counts[b] >= batch_depth_quota.get(b, 0):
                            ok = False
                    
                    # Enforce attribute diversity limits
                    # Relax attribute limits when we're very close to target (last 5% of questions)
                    current_progress = selection_state.get('current_progress', 0)
                    total_target = sum(selection_state['depth_quota'].values())
                    enforce_attr_limits = current_progress < (total_target * 0.95)
                    if ok and enforce_attr_limits and 'attribute_counts' in selection_state:
                        q_attrs = detect_question_attributes(q)
                        for attr in q_attrs:
                            if attr in selection_state['attribute_limits']:
                                if selection_state['attribute_counts'].get(attr, 0) >= selection_state['attribute_limits'][attr]:
                                    ok = False
                                    break
                
                if ok:
                    selected.append(q)
                    if selection_state is not None:
                        if balanced_depth:
                            b = depth_bucket(q.get('evidence_depth', 0.0))
                            selection_state['depth_counts'][b] += 1
                        # Update attribute counts
                        if 'attribute_counts' in selection_state:
                            q_attrs = detect_question_attributes(q)
                            for attr in q_attrs:
                                if attr in selection_state['attribute_counts']:
                                    selection_state['attribute_counts'][attr] += 1
                    if batch_depth_quota is not None:
                        batch_counts[b] += 1

            if selected:
                print(f"✓ Selected {len(selected)} questions for trajectory {trajectory_index + 1} after depth balancing")
                return selected
            
            # If depth balancing is enabled and nothing fit the quotas, skip this trajectory
            if balanced_depth:
                print("✗ No questions fit remaining quotas; continuing to next trajectory")
                return []

            # Fallback only when not balancing depth
            elif len(validated_questions) >= 1:
                print(f"✓ Got {len(validated_questions)} valid questions (target was {num_questions}), accepting (no depth balance)...")
                return validated_questions
            
            else:
                print(f"✗ No valid questions generated")
                if attempt < max_retries - 1:
                    print("Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    return []
                    
        except Exception as e:
            print(f"✗ Error generating questions: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
                continue
            else:
                return []
    
    return []


def main():
    parser = argparse.ArgumentParser(description='Robust benchmark question generation')
    parser.add_argument('--input', default='../datasets/data/pgeval_benchmark/pgeval.json',
                       help='Input pgeval JSON file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--trajectory-mode', choices=['32k', '128k'], required=True,
                       help='Trajectory mode (32k or 128k)')
    parser.add_argument('--num-trajectories', type=int, default=10,
                       help='Number of trajectories to process')
    parser.add_argument('--questions-per-trajectory', type=int, default=5,
                       help='Questions per trajectory')
    parser.add_argument('--total-questions', type=int, default=0,
                       help='If >0, generate until this total is reached (overrides num-trajectories * questions-per-trajectory)')
    parser.add_argument('--easy-ratio', type=float, default=0.5,
                       help='Ratio of easy questions')
    parser.add_argument('--medium-ratio', type=float, default=0.3,
                       help='Ratio of medium questions')
    parser.add_argument('--hard-ratio', type=float, default=0.2,
                       help='Ratio of hard questions')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Starting trajectory index')
    parser.add_argument('--model-id', default='openai/gpt-oss-120b',
                       help='Model ID for generation')
    parser.add_argument('--max-retries', type=int, default=5,
                       help='Maximum retries per trajectory')
    parser.add_argument('--balanced-depth', action='store_true',
                       help='Enforce balanced evidence depth distribution (early/middle/late)')
    
    args = parser.parse_args()
    
    # Calculate difficulty distribution
    difficulty_distribution = {
        'Easy': args.easy_ratio,
        'Medium': args.medium_ratio,
        'Hard': args.hard_ratio
    }
    
    print("=" * 80)
    print("ROBUST BENCHMARK QUESTION GENERATION")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Processing {args.num_trajectories} trajectories starting from index {args.start_index}")
    print(f"Questions per trajectory: {args.questions_per_trajectory}")
    print(f"Trajectory mode: {args.trajectory_mode}")
    print(f"Difficulty distribution: Easy {args.easy_ratio*100:.1f}%, Medium {args.medium_ratio*100:.1f}%, Hard {args.hard_ratio*100:.1f}%")
    print(f"Max retries per trajectory: {args.max_retries}")
    print("=" * 80)
    
    # Load data
    print("Loading pgeval data...")
    with open(args.input, 'r') as f:
        trajectories = json.load(f)
    print(f"✓ Loaded {len(trajectories)} trajectories")
    
    # Generate questions with target-based approach
    all_questions = []
    successful_trajectories = 0
    target_questions = args.total_questions if args.total_questions and args.total_questions > 0 else args.num_trajectories * args.questions_per_trajectory
    processed_trajectories = set()
    successful_trajectory_indices = []  # Track successful trajectories for resampling
    trajectory_question_counts = defaultdict(int)  # Track questions per trajectory to enforce max limit
    max_questions_per_trajectory = args.questions_per_trajectory * 2  # Allow up to 2x during resampling
    trajectory_index = args.start_index
    resampling_round = 0
    
    print(f"Target: {target_questions} questions from {args.num_trajectories} trajectories")
    
    # Set up selection state for balancing/diversity across the whole run
    selection_state = {
        'current_progress': 0,  # Track progress for relaxing limits near end
        'depth_counts': {'early': 0, 'middle': 0, 'late': 0},
        'depth_quota': {
            'early': target_questions // 3,
            'middle': target_questions // 3,
            'late': target_questions - 2 * (target_questions // 3)
        },
        # Track attribute usage to enforce diversity
        'attribute_counts': {
            'product_name': 0,
            'brand': 0,
            'timestamp': 0,
            'search_query': 0,
            'price': 0,
            'review_rating': 0,
            'review_count': 0,
            'prime': 0,
            'product_type': 0,
            'asin': 0,
            'action_type': 0,
            'row_id': 0
        },
        # Max allowed per attribute (scaled to target)
        # Note: action_type is important for distinguishing click vs purchase vs cart, so allow more
        'attribute_limits': {
            'product_name': max(5, target_questions // 6),    # ~17% max (was 10%)
            'brand': max(5, target_questions // 6),            # ~17% max (was 10%)
            'timestamp': target_questions,                     # unlimited
            'search_query': target_questions // 3,            # ~33% max
            'price': target_questions // 3,                   # ~33% max (was 25%)
            'review_rating': target_questions // 4,           # ~25% max (was 20%)
            'review_count': target_questions // 4,            # ~25% max (was 20%)
            'prime': target_questions // 4,                   # ~25% max (was 20%)
            'product_type': target_questions // 3,            # ~33% max (was 25%)
            'asin': target_questions // 4,                    # ~25% max (was 20%)
            'action_type': int(target_questions * 0.6),       # ~60% max (was 33%) - important for click/purchase distinction
            'row_id': target_questions // 3                   # ~33% max
        }
    }

    output_jsonl = args.output + ".jsonl"
    jsonl_file = open(output_jsonl, "a", encoding="utf-8")

    while len(all_questions) < target_questions:
        # Check if we've exhausted all trajectories in the current pass
        if trajectory_index >= len(trajectories):
            if not successful_trajectory_indices:
                print(f"\n✗ No successful trajectories to resample from. Stopping at {len(all_questions)}/{target_questions} questions.")
                break
            
            # Start resampling from successful trajectories
            resampling_round += 1
            print(f"\n{'='*80}")
            print(f"RESAMPLING ROUND {resampling_round}")
            print(f"{'='*80}")
            print(f"Exhausted all {len(trajectories)} trajectories. Resampling from {len(successful_trajectory_indices)} successful ones.")
            print(f"Current: {len(all_questions)}/{target_questions} questions")
            print(f"Remaining quotas - early: {selection_state['depth_quota']['early'] - selection_state['depth_counts']['early']}, "
                  f"middle: {selection_state['depth_quota']['middle'] - selection_state['depth_counts']['middle']}, "
                  f"late: {selection_state['depth_quota']['late'] - selection_state['depth_counts']['late']}")
            
            # Reset for resampling: clear processed set and restart from successful trajectories
            processed_trajectories.clear()
            trajectory_index = 0
            # Use a shuffled order to avoid always hitting the same trajectories
            import random
            random.shuffle(successful_trajectory_indices)
            continue
        
        # Skip if we've already processed this trajectory in current pass
        if trajectory_index in processed_trajectories:
            trajectory_index += 1
            continue
        
        # If resampling, only process trajectories that were previously successful
        if resampling_round > 0 and trajectory_index not in successful_trajectory_indices:
            trajectory_index += 1
            continue
        
        # Skip if this trajectory already has max questions
        if trajectory_question_counts[trajectory_index] >= max_questions_per_trajectory:
            trajectory_index += 1
            continue
            
        trajectory_data = trajectories[trajectory_index]
        processed_trajectories.add(trajectory_index)
        
        # Calculate how many questions we still need
        remaining_questions = target_questions - len(all_questions)
        questions_to_generate = min(args.questions_per_trajectory, remaining_questions)
        
        print(f"\n{'='*60}")
        print(f"Processing Trajectory {trajectory_index + 1}")
        print(f"Customer ID: {trajectory_data.get('cid', 'unknown')}")
        print(f"Need {questions_to_generate} questions (remaining: {remaining_questions})")
        print(f"Current progress: {len(all_questions)}/{target_questions} questions")
        print(f"{'='*60}")
        
        # Determine per-trajectory batch quotas for evidence depth to keep strict control
        remaining_early = selection_state['depth_quota']['early'] - selection_state['depth_counts']['early']
        remaining_middle = selection_state['depth_quota']['middle'] - selection_state['depth_counts']['middle']
        remaining_late = selection_state['depth_quota']['late'] - selection_state['depth_counts']['late']

        # Allocate this batch's quotas greedily, capped by remaining and by questions_to_generate
        batch_quota = {'early': 0, 'middle': 0, 'late': 0}
        # Start with the bucket with the largest remaining to maximize success
        remaining_list = sorted([
            ('early', max(0, remaining_early)),
            ('middle', max(0, remaining_middle)),
            ('late', max(0, remaining_late)),
        ], key=lambda x: x[1], reverse=True)
        to_assign = questions_to_generate
        for name, rem in remaining_list:
            if to_assign <= 0:
                break
            take = min(rem, to_assign)
            batch_quota[name] += take
            to_assign -= take
        # If still unassigned (all rem were zero), just distribute evenly (will be filtered later)
        idx = 0
        order = ['early', 'middle', 'late']
        while to_assign > 0:
            batch_quota[order[idx % 3]] += 1
            to_assign -= 1
            idx += 1

        questions = robust_generate_questions_for_trajectory(
            trajectory_data, trajectory_index, questions_to_generate, 
            difficulty_distribution, args.trajectory_mode, 
            args.model_id, args.max_retries,
            selection_state=selection_state,
            balanced_depth=args.balanced_depth,
            batch_depth_quota=batch_quota if args.balanced_depth else None
        )
        
        if questions:
            all_questions.extend(questions)
            
            for q in questions:
                jsonl_file.write(json.dumps(q, ensure_ascii=False) + "\n")
                
            # Update current progress in selection_state
            selection_state['current_progress'] = len(all_questions)
            
            # Track questions per trajectory
            trajectory_question_counts[trajectory_index] += len(questions)
            
            # Track this trajectory as successful (only once, not on resampling)
            if resampling_round == 0 and trajectory_index not in successful_trajectory_indices:
                successful_trajectory_indices.append(trajectory_index)
                successful_trajectories += 1
            
            print(f"✓ Successfully generated {len(questions)} questions for trajectory {trajectory_index + 1} (total from this traj: {trajectory_question_counts[trajectory_index]})")
            
            # If we got fewer questions than requested, try to retry this trajectory
            if len(questions) < questions_to_generate and len(all_questions) < target_questions:
                print(f"⚠ Got {len(questions)} questions, needed {questions_to_generate}. Retrying trajectory...")
                # Remove from processed set so we can retry
                processed_trajectories.remove(trajectory_index)
                continue
        else:
            print(f"✗ Failed to generate questions for trajectory {trajectory_index + 1}")
        
        trajectory_index += 1
        print(f"Progress: {len(all_questions)}/{target_questions} questions from {len(successful_trajectory_indices)} unique successful trajectories")
    
    # If balanced-depth is enabled, selection has been enforced online per-trajectory using selection_state quotas.


    # Trim to exact target if we have more
    if len(all_questions) > target_questions:
        all_questions = all_questions[:target_questions]
    
    print(f"\n{'='*80}")
    print("BENCHMARK ANALYSIS")
    print(f"{'='*80}")
    print(f"Total questions: {len(all_questions)}")
    print(f"Successful trajectories: {successful_trajectories}/{args.num_trajectories}")
    
    if len(all_questions) > 0:
        # Analyze distribution
        analysis = analyze_question_distribution(all_questions)
        print(f"By difficulty: {analysis['by_difficulty']}")
        print(f"Evidence depth stats: min={analysis['evidence_depth_stats']['min']:.2f}, max={analysis['evidence_depth_stats']['max']:.2f}, avg={analysis['evidence_depth_stats']['avg']:.2f}")
        print(f"Evidence depth distribution: {analysis['evidence_depth_stats']['distribution']}")
        
        # Save results
        output_data = {
            'metadata': {
                'total_questions': len(all_questions),
                'num_trajectories': successful_trajectories,
                'questions_per_trajectory': args.questions_per_trajectory,
                'trajectory_mode': args.trajectory_mode,
                'difficulty_distribution': difficulty_distribution,
                'analysis': analysis
            },
            'questions': all_questions
        }
        
        with open(args.output+'_full', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved benchmark to {args.output}"+"_full")
    else:
        print("✗ No questions generated!")
        return 1
    
    jsonl_file.close()

    return 0


if __name__ == '__main__':
    exit(main())
