import pandas as pd
import re
import os
from convert_trajectory_with_schema import parse_trajectory_line, events_to_table
import sqlite3

def load_trajectory(qid: str, base_dir: str = "/root/sandbox/trajectory_files") -> sqlite3.Connection:
    """
    read trajectory based on qid, parse, and obtain sqlite3 database。
    """
    filepath = os.path.join(base_dir, f"trajectory_{qid}.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trajectory file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    events = []
    skip_count=0
    
    for line in lines:
        event = parse_trajectory_line(line)
        if event:
            events.append(event)
        else:
            skip_count+=1
            # print(f"Skipped line (unmatched): {line}")

    table = events_to_table(events)
    # print(f"In total {len(lines)} line, parse {len(table['rows'])} line, skip {skip_count} line")
    # print(f"✓ Converted trajectory to table with {len(table['rows'])} rows")
    
    # Create SQLite database for validation
    conn = sqlite3.connect(':memory:')
    df = pd.DataFrame(table['rows'], columns=table['header'])
    df.to_sql('events', conn, index=False, if_exists='replace')

    return conn, df

def sql_query(conn: sqlite3.Connection, sql: str) -> tuple:
    """ SQL query against the database."""
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        return True, result
    except Exception as e:
        return False, str(e)



import json

# qid = "25"
# df = load_trajectory(qid)
# _,_ =sql_query(sql)

def save_trajectory_file(qid, trajectory, folder="/root/sandbox/trajectory_files"):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"trajectory_{qid}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(trajectory)
    return file_path

## test benchmark ###
# input_path="data/merged_output3.jsonl"
# with open(input_path, "r", encoding="utf-8") as fin:
#     data = [json.loads(line) for line in fin]

# valida_true = 0
# wrong_answer =0
# no_result = 0
# for single_data in data:
#     save_trajectory_file(single_data["qid"], single_data["trajectory"])
#     qid = single_data.get("qid")
#     conn, df = load_trajectory(qid)
#     sql_query_single = single_data.get("sql_query")
#     validate_true, sql_result = sql_query(conn, sql_query_single)

#     if validate_true==True and sql_result!=[]:
#         if str(single_data.get("reference_answer"))==str(sql_result[0][0]):
#             valida_true+=1
#         else:
#             wrong_answer+=1
#             # print(f'correct one: {single_data.get("reference_answer")}, predict: {sql_result[0][0]}')
#             # print(single_data.get('question'))
#     elif validate_true==True and sql_result==[]:
#         validate_true, sql_result = sql_query(conn, sql_query_single)
#         no_result+=1

# print(valida_true)
# print(wrong_answer)
# print(no_result)
## test benchmark ###

# from data_loader import load_trajectory
# from data_loader import sql_query

# def load_trajectory(qid: str, base_dir: str = "trajectory_files") -> sqlite3.Connection:
#     """
#     read trajectory based on qid, parse, and obtain sqlite3 database。
#     """

# def sql_query(conn: sqlite3.Connection, sql: str) -> tuple:
#     """ SQL query against the database."""

# qid = "d5f17bc494"
# conn, df = load_trajectory(qid)
# question = 'What was the first search query made by the customer?'
# single_query = "SELECT search_query FROM events WHERE action_type='type' ORDER BY timestamp LIMIT 1;"
# validate_true, sql_result =sql_query(conn, single_query)
# print(validate_true)
# print(sql_result)

