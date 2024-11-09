import os
from typing import Any
import numpy as np
from langchain_community.graphs import Neo4jGraph

def _get_current_hospitals() -> list[str]:
    """Fetch a list of current hospital names from a Neo4j database."""
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )

    current_hospitals = graph.query(
        """
        MATCH (h:Hospital)
        RETURN h.name AS hospital_name
        """
    )

    return [d["hospital_name"].lower() for d in current_hospitals]

def _get_current_wait_time_minutes(hospital: str) -> int:
    """Get the current wait time at a hospital in minutes."""
    return np.random.randint(low=10, high=600)

def stringify(total_minutes: int) -> str:
    """Convert minutes to a string representation."""
    hours, minutes = int(total_minutes/60), total_minutes%60 
    return f"{hours} hours {minutes} minutes"

def get_current_wait_times(hospital: str) -> str:
    """Get the current wait time at a hospital formatted as a string."""
    current_hospitals = _get_current_hospitals()

    if hospital.lower() not in current_hospitals:
        return f"Hospital '{hospital}' does not exist."

    wait_time_in_minutes = _get_current_wait_time_minutes(hospital)
    return stringify(wait_time_in_minutes)

def get_most_available_hospital(_: Any) -> dict[str, str]:
    """Find the hospital with the shortest wait time."""
    current_hospitals = _get_current_hospitals()

    current_wait_times = [
        _get_current_wait_time_minutes(h) for h in current_hospitals
    ]

    best_time_idx = np.argmin(current_wait_times)
    best_hospital = current_hospitals[best_time_idx]
    best_wait_time = current_wait_times[best_time_idx]

    return {best_hospital: stringify(best_wait_time)}