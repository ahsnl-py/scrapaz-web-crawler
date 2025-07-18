#!/usr/bin/env python3
"""
Schema management script for the scraper service
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

def list_schemas(schema_dir: str = "./schemas") -> List[str]:
    """List all available schemas"""
    schema_path = Path(schema_dir)
    if not schema_path.exists():
        print(f"Schema directory {schema_dir} does not exist")
        return []
    
    schemas = []
    for schema_file in schema_path.glob("*.json"):
        schemas.append(schema_file.name)
    
    return sorted(schemas)

def show_schema_info(schema_name: str, schema_dir: str = "./schemas") -> Dict:
    """Show information about a specific schema"""
    schema_path = Path(schema_dir) / schema_name
    
    if not schema_path.exists():
        return {"error": f"Schema {schema_name} not found"}
    
    try:
        with open(schema_path, 'r') as f:
            schema_data = json.load(f)
        
        stat = schema_path.stat()
        return {
            "name": schema_name,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
            "fields": len(schema_data.get("fields", [])),
            "base_selector": schema_data.get("baseSelector", "N/A")
        }
    except Exception as e:
        return {"error": f"Failed to read schema {schema_name}: {str(e)}"}

def backup_schemas(schema_dir: str = "./schemas", backup_dir: str = "./schema_backups"):
    """Backup all schemas to a backup directory"""
    schema_path = Path(schema_dir)
    backup_path = Path(backup_dir)
    
    if not schema_path.exists():
        print(f"Schema directory {schema_dir} does not exist")
        return
    
    backup_path.mkdir(exist_ok=True)
    
    backed_up = 0
    for schema_file in schema_path.glob("*.json"):
        backup_file = backup_path / schema_file.name
        try:
            with open(schema_file, 'r') as src, open(backup_file, 'w') as dst:
                json.dump(json.load(src), dst, indent=2)
            backed_up += 1
        except Exception as e:
            print(f"Failed to backup {schema_file.name}: {e}")
    
    print(f"Backed up {backed_up} schemas to {backup_dir}")

def restore_schemas(backup_dir: str = "./schema_backups", schema_dir: str = "./schemas"):
    """Restore schemas from backup directory"""
    backup_path = Path(backup_dir)
    schema_path = Path(schema_dir)
    
    if not backup_path.exists():
        print(f"Backup directory {backup_dir} does not exist")
        return
    
    schema_path.mkdir(exist_ok=True)
    
    restored = 0
    for backup_file in backup_path.glob("*.json"):
        schema_file = schema_path / backup_file.name
        try:
            with open(backup_file, 'r') as src, open(schema_file, 'w') as dst:
                json.dump(json.load(src), dst, indent=2)
            restored += 1
        except Exception as e:
            print(f"Failed to restore {backup_file.name}: {e}")
    
    print(f"Restored {restored} schemas to {schema_dir}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python manage_schemas.py <command> [options]")
        print("\nCommands:")
        print("  list                    - List all schemas")
        print("  info <schema_name>      - Show info about a specific schema")
        print("  backup [backup_dir]     - Backup all schemas")
        print("  restore [backup_dir]    - Restore schemas from backup")
        print("  stats                   - Show schema statistics")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        schemas = list_schemas()
        if schemas:
            print("Available schemas:")
            for schema in schemas:
                print(f"  - {schema}")
        else:
            print("No schemas found")
    
    elif command == "info" and len(sys.argv) > 2:
        schema_name = sys.argv[2]
        info = show_schema_info(schema_name)
        if "error" in info:
            print(f"Error: {info['error']}")
        else:
            print(f"Schema: {info['name']}")
            print(f"Size: {info['size_bytes']} bytes")
            print(f"Fields: {info['fields']}")
            print(f"Base Selector: {info['base_selector']}")
    
    elif command == "backup":
        backup_dir = sys.argv[2] if len(sys.argv) > 2 else "./schema_backups"
        backup_schemas(backup_dir=backup_dir)
    
    elif command == "restore":
        backup_dir = sys.argv[2] if len(sys.argv) > 2 else "./schema_backups"
        restore_schemas(backup_dir=backup_dir)
    
    elif command == "stats":
        schemas = list_schemas()
        if schemas:
            total_size = 0
            for schema in schemas:
                info = show_schema_info(schema)
                if "error" not in info:
                    total_size += info["size_bytes"]
            
            print(f"Total schemas: {len(schemas)}")
            print(f"Total size: {total_size} bytes")
            print(f"Schema directory: {Path('./schemas').absolute()}")
        else:
            print("No schemas found")
    
    else:
        print("Invalid command. Use 'python manage_schemas.py' for help.")

if __name__ == "__main__":
    main() 