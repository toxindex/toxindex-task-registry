#!/usr/bin/env python3
"""
Script to seed default workflows from JSON file into the database.
This should be run after the Flyway migrations have created the schema.
"""

import json
import os
import sys
import datastore as ds

def load_workflows_from_json(json_path):
    """Load workflows from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('workflows', [])
    except FileNotFoundError:
        print(f"Error: {json_path} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []

def insert_workflows(workflows):
    """Insert workflows into the database."""
    for workflow in workflows:
        # Insert workflow with all routing fields
        ds.execute("""
            INSERT INTO workflows (workflow_id, title, description, initial_prompt, celery_task, task_name, queue)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (workflow_id) DO UPDATE SET
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                initial_prompt = EXCLUDED.initial_prompt,
                celery_task = EXCLUDED.celery_task,
                task_name = EXCLUDED.task_name,
                queue = EXCLUDED.queue
        """, (
            workflow['workflow_id'],
            workflow['title'],
            workflow['description'],
            workflow.get('initial_prompt', ''),
            workflow.get('celery_task'),
            workflow.get('task_name'),
            workflow.get('queue')
        ))
        
        print(f"Inserted/updated workflow: {workflow['title']} (ID: {workflow['workflow_id']})")

def setup_workflow_access():
    """Set up default workflow access for user groups."""
    # Admin gets access to all workflows
    ds.execute("""
        INSERT INTO workflow_group_access (workflow_id, group_id)
        SELECT w.workflow_id, '00000000-0000-0000-0000-000000000001'
        FROM workflows w
        ON CONFLICT (workflow_id, group_id) DO NOTHING
    """)
    
    # Researchers get access to analysis workflows (1, 4, 5)
    ds.execute("""
        INSERT INTO workflow_group_access (workflow_id, group_id) VALUES
            (1, '00000000-0000-0000-0000-000000000002'),
            (4, '00000000-0000-0000-0000-000000000002'),
            (5, '00000000-0000-0000-0000-000000000002')
        ON CONFLICT (workflow_id, group_id) DO NOTHING
    """)
    
    # Basic users get access to simple workflows (2, 3)
    ds.execute("""
        INSERT INTO workflow_group_access (workflow_id, group_id) VALUES
            (2, '00000000-0000-0000-0000-000000000003'),
            (3, '00000000-0000-0000-0000-000000000003')
        ON CONFLICT (workflow_id, group_id) DO NOTHING
    """)
    
    print("Workflow access permissions configured")

def main():
    """Main function."""
    # Use path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "default_workflows.json")
    
    # Load workflows from JSON
    workflows = load_workflows_from_json(json_path)
    if not workflows:
        print("No workflows found or error loading JSON file")
        sys.exit(1)
    
    try:
        # Insert workflows
        insert_workflows(workflows)
        
        # Set up access permissions
        setup_workflow_access()
        
        print("Workflow seeding completed successfully")
        
    except Exception as e:
        print(f"Database error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 