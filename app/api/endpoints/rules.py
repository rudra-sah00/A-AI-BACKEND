from fastapi import APIRouter, HTTPException, Body, Query, Path
from typing import List, Optional

from app.models.rule import Rule, RuleCreate, RuleUpdate, RuleResponse
from app.services.rule_service import rule_service
from app.services.camera_service import camera_service

router = APIRouter()

@router.post("/", response_model=RuleResponse)
async def create_rule(rule_data: RuleCreate = Body(...)):
    """
    Create a new attendance rule.
    
    Args:
        rule_data: Rule data containing name, event type, condition details, etc.
        
    Returns:
        The newly created rule with ID and timestamps
    """
    try:
        # Verify camera exists
        camera = camera_service.get_camera(rule_data.cameraId)
        if not camera:
            raise HTTPException(
                status_code=404,
                detail=f"Camera with ID {rule_data.cameraId} not found"
            )
        
        # Create the rule
        new_rule = rule_service.create_rule(rule_data)
        
        return RuleResponse(
            id=new_rule.id,
            name=new_rule.name,
            event=new_rule.event,
            condition=new_rule.condition,
            enabled=new_rule.enabled,
            days=new_rule.days,
            cameraId=new_rule.cameraId,
            cameraName=new_rule.cameraName,
            created_at=new_rule.created_at,
            updated_at=new_rule.updated_at,
            message="Rule created successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating rule: {str(e)}")


@router.get("/", response_model=List[RuleResponse])
async def get_all_rules(
    camera_id: Optional[str] = Query(None, description="Filter rules by camera ID"),
    event_type: Optional[str] = Query(None, description="Filter rules by event type")
):
    """
    Get all rules with optional filtering by camera ID or event type.
    
    Args:
        camera_id: Optional camera ID to filter rules
        event_type: Optional event type to filter rules
        
    Returns:
        List of rules matching the filter criteria
    """
    try:
        rules = rule_service.list_rules(camera_id=camera_id, event_type=event_type)
        
        return [
            RuleResponse(
                id=rule.id,
                name=rule.name,
                event=rule.event,
                condition=rule.condition,
                enabled=rule.enabled,
                days=rule.days,
                cameraId=rule.cameraId,
                cameraName=rule.cameraName,
                created_at=rule.created_at,
                updated_at=rule.updated_at,
                message="Rule retrieved successfully"
            ) for rule in rules
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving rules: {str(e)}")


@router.get("/{rule_id}", response_model=RuleResponse)
async def get_rule(rule_id: str = Path(..., description="The ID of the rule to retrieve")):
    """
    Get a specific rule by ID.
    
    Args:
        rule_id: ID of the rule to retrieve
        
    Returns:
        The requested rule details
    """
    try:
        rule = rule_service.get_rule(rule_id)
        if not rule:
            raise HTTPException(status_code=404, detail=f"Rule with ID {rule_id} not found")
        
        return RuleResponse(
            id=rule.id,
            name=rule.name,
            event=rule.event,
            condition=rule.condition,
            enabled=rule.enabled,
            days=rule.days,
            cameraId=rule.cameraId,
            cameraName=rule.cameraName,
            created_at=rule.created_at,
            updated_at=rule.updated_at,
            message="Rule retrieved successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving rule: {str(e)}")


@router.put("/{rule_id}", response_model=RuleResponse)
async def update_rule(
    rule_id: str = Path(..., description="The ID of the rule to update"),
    rule_data: RuleUpdate = Body(...)
):
    """
    Update a rule by ID.
    
    Args:
        rule_id: ID of the rule to update
        rule_data: New rule data
        
    Returns:
        The updated rule details
    """
    try:
        # Check if rule exists
        rule = rule_service.get_rule(rule_id)
        if not rule:
            raise HTTPException(status_code=404, detail=f"Rule with ID {rule_id} not found")
        
        # If camera ID is being updated, verify the new camera exists
        if rule_data.cameraId is not None and rule_data.cameraId != rule.cameraId:
            camera = camera_service.get_camera(rule_data.cameraId)
            if not camera:
                raise HTTPException(
                    status_code=404,
                    detail=f"Camera with ID {rule_data.cameraId} not found"
                )
        
        # Update the rule
        updated_rule = rule_service.update_rule(rule_id, rule_data)
        
        return RuleResponse(
            id=updated_rule.id,
            name=updated_rule.name,
            event=updated_rule.event,
            condition=updated_rule.condition,
            enabled=updated_rule.enabled,
            days=updated_rule.days,
            cameraId=updated_rule.cameraId,
            cameraName=updated_rule.cameraName,
            created_at=updated_rule.created_at,
            updated_at=updated_rule.updated_at,
            message="Rule updated successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating rule: {str(e)}")


@router.delete("/{rule_id}", response_model=dict)
async def delete_rule(rule_id: str = Path(..., description="The ID of the rule to delete")):
    """
    Delete a rule by ID.
    
    Args:
        rule_id: ID of the rule to delete
        
    Returns:
        Success message
    """
    try:
        # Check if rule exists
        rule = rule_service.get_rule(rule_id)
        if not rule:
            raise HTTPException(status_code=404, detail=f"Rule with ID {rule_id} not found")
        
        # Delete the rule
        success = rule_service.delete_rule(rule_id)
        
        if success:
            return {
                "rule_id": rule_id,
                "message": "Rule deleted successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete rule")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting rule: {str(e)}")


@router.patch("/{rule_id}", response_model=RuleResponse)
async def toggle_rule_status(
    rule_id: str = Path(..., description="The ID of the rule to update"),
    update_data: dict = Body(..., description="The update data containing enabled status")
):
    """
    Toggle a rule's enabled status.
    
    Args:
        rule_id: ID of the rule to update
        update_data: Dictionary containing enabled field (boolean)
        
    Returns:
        The updated rule details
    """
    try:
        # Check if rule exists
        rule = rule_service.get_rule(rule_id)
        if not rule:
            raise HTTPException(status_code=404, detail=f"Rule with ID {rule_id} not found")
        
        # Validate the request body has the required field
        if "enabled" not in update_data:
            raise HTTPException(
                status_code=400,
                detail="Request body must contain 'enabled' field"
            )
        
        if not isinstance(update_data["enabled"], bool):
            raise HTTPException(
                status_code=400,
                detail="'enabled' field must be a boolean value"
            )
        
        # Create a RuleUpdate with only the enabled field
        rule_update = RuleUpdate(enabled=update_data["enabled"])
        
        # Update the rule's enabled status
        updated_rule = rule_service.update_rule(rule_id, rule_update)
        
        status_message = "Rule enabled" if updated_rule.enabled else "Rule disabled"
        
        return RuleResponse(
            id=updated_rule.id,
            name=updated_rule.name,
            event=updated_rule.event,
            condition=updated_rule.condition,
            enabled=updated_rule.enabled,
            days=updated_rule.days,
            cameraId=updated_rule.cameraId,
            cameraName=updated_rule.cameraName,
            created_at=updated_rule.created_at,
            updated_at=updated_rule.updated_at,
            message=status_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating rule status: {str(e)}")