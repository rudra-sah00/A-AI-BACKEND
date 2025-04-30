import json
import os
import uuid
from typing import Optional, Dict, List, Any
from datetime import datetime

from app.core.config import settings
from app.models.rule import Rule, RuleCreate, RuleUpdate


class RuleService:
    def __init__(self):
        # Ensure the data directory exists
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        self.rules_file = settings.DATA_DIR / "rules.json"
        
        # Initialize rules.json if it doesn't exist
        if not os.path.exists(self.rules_file):
            with open(self.rules_file, "w") as f:
                json.dump({}, f)
            print(f"Initialized empty rules file at {self.rules_file}")
    
    def _read_rules(self) -> Dict:
        """Read rules from JSON file"""
        try:
            with open(self.rules_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is empty/invalid JSON, return empty dict
            return {}
    
    def _write_rules(self, rules_data: Dict) -> None:
        """Write rules to JSON file"""
        try:
            # Make sure directory exists
            os.makedirs(os.path.dirname(self.rules_file), exist_ok=True)
            
            # Write using a temporary file first
            temp_file = f"{self.rules_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump(rules_data, f, indent=4)
                # Ensure data is written to disk before closing file
                os.fsync(f.fileno())
            
            # Safely rename the temp file to the actual file
            os.replace(temp_file, self.rules_file)
            
            print(f"Saved rules data to {self.rules_file}")
                    
        except Exception as e:
            print(f"Error writing rules file: {str(e)}")
            raise
    
    def create_rule(self, rule_data: RuleCreate) -> Rule:
        """Create a new rule"""
        # Read existing rules
        rules = self._read_rules()
        
        # Generate a unique rule ID
        rule_id = str(uuid.uuid4())
        
        # Convert rule_data to dict to include any extra fields
        rule_dict_data = rule_data.model_dump()
        
        # Prepare rule data with condition as dict
        rule_dict = {
            "id": rule_id,
            "name": rule_dict_data["name"],
            "event": rule_dict_data["event"],
            "condition": rule_dict_data["condition"],
            "enabled": rule_dict_data.get("enabled", True),
            "cameraId": rule_dict_data["cameraId"],
            "cameraName": rule_dict_data["cameraName"],
            "created_at": datetime.now().isoformat(),
            "updated_at": None
        }
        
        # Add days if present in the request
        if "days" in rule_dict_data:
            rule_dict["days"] = rule_dict_data["days"]
        else:
            rule_dict["days"] = []
        
        # Add any additional fields from the request
        for key, value in rule_dict_data.items():
            if key not in rule_dict and key != "condition":
                rule_dict[key] = value
                
        # Add rule to rules dictionary
        rules[rule_id] = rule_dict
        
        # Save changes
        self._write_rules(rules)
        
        # Return rule data
        return Rule(**rule_dict)
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by ID"""
        rules = self._read_rules()
        if rule_id not in rules:
            return None
        
        return Rule(**rules[rule_id])
    
    def update_rule(self, rule_id: str, rule_data: RuleUpdate) -> Optional[Rule]:
        """Update a rule by ID"""
        rules = self._read_rules()
        if rule_id not in rules:
            return None
        
        # Update rule fields
        rule_dict = rules[rule_id]
        
        # Convert rule_data to dict to include any extra fields
        if rule_data:
            rule_update_dict = rule_data.model_dump(exclude_unset=True)
            
            # Handle special case for condition
            if "condition" in rule_update_dict:
                rule_dict["condition"] = rule_update_dict["condition"]
                
            # Update all other fields
            for key, value in rule_update_dict.items():
                if value is not None and key != "condition":
                    rule_dict[key] = value
        
        # Update timestamp
        rule_dict["updated_at"] = datetime.now().isoformat()
        
        # Save changes
        self._write_rules(rules)
        
        # Return updated rule
        return Rule(**rule_dict)
    
    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by ID"""
        rules = self._read_rules()
        if rule_id not in rules:
            return False
        
        # Delete rule from rules.json
        del rules[rule_id]
        self._write_rules(rules)
        
        return True
    
    def list_rules(self, camera_id: Optional[str] = None, event_type: Optional[str] = None) -> List[Rule]:
        """List rules with optional filtering by camera ID or event type"""
        rules = self._read_rules()
        result = []
        
        for rule_data in rules.values():
            # Filter by camera ID if provided
            if camera_id and rule_data.get("cameraId") != camera_id:
                continue
                
            # Filter by event type if provided
            if event_type and rule_data.get("event") != event_type:
                continue
                
            result.append(Rule(**rule_data))
        
        return result

# Create a singleton instance for global use
rule_service = RuleService()