def calculate_damage_score(object_area: int, damage_area: int) -> float:
    """Calculate damage as simple percentage of object area.
    
    damage_score = (damage_area / object_area) * 100.0
    
    Returns percentage (0.0 to 100.0)
    Clamp to [0, 100]
    """
    if object_area <= 0:
        return 0.0
    
    damage_score = (damage_area / object_area) * 100.0
    return max(0.0, min(100.0, damage_score))
