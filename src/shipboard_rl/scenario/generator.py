#!/usr/bin/env python3
"""
SHIPBOARD FIRE RESPONSE SCENARIO GENERATOR
==========================================

Creates realistic fire emergency scenarios specifically for shipboard 
operations in various maritime environments.

Generates scenarios with varying:
- Fire locations within ship compartments (1-12 spaces)
- Fire types common in maritime operations
- Shipboard-specific conditions
- Crew readiness levels during various operational periods
- Equipment status during maintenance and operations
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import random


class ShipboardCompartment(Enum):
    """Ship compartments layout"""
    COMP_01 = "compartment_01"  # Forward engineering
    COMP_02 = "compartment_02"  # Forward berthing
    COMP_03 = "compartment_03"  # Forward mess
    COMP_04 = "compartment_04"  # Forward storage
    COMP_05 = "compartment_05"  # Mid-ship engineering
    COMP_06 = "compartment_06"  # Mid-ship operations
    COMP_07 = "compartment_07"  # Mid-ship berthing
    COMP_08 = "compartment_08"  # Mid-ship storage
    COMP_09 = "compartment_09"  # Aft engineering
    COMP_10 = "compartment_10"  # Aft berthing
    COMP_11 = "compartment_11"  # Aft mess
    COMP_12 = "compartment_12"  # Aft storage


class ShipFireType(Enum):
    """Fire types common during shipboard operations"""
    WELDING_SPARK = "welding_spark"           # Hot work incidents
    ELECTRICAL_SHORT = "electrical_short"     # Electrical maintenance
    PAINT_VAPOR = "paint_vapor"              # Painting operations
    CUTTING_TORCH = "cutting_torch"          # Metal cutting work
    FUEL_SPILL = "fuel_spill"               # Fuel system maintenance
    CHEMICAL_REACTION = "chemical_reaction"   # Cleaning/preservation
    INSULATION_FIRE = "insulation_fire"     # Lagging work
    MACHINERY_OVERHEAT = "machinery_overheat" # Equipment testing


class ShipCondition(Enum):
    """Environmental conditions during ship operations"""
    NORMAL_OPERATIONS = "normal_operations"   # Standard operations
    HEAVY_MAINTENANCE = "heavy_maintenance"   # Major maintenance period
    PRESERVATION_WORK = "preservation_work"   # Painting/coating
    SYSTEM_TESTING = "system_testing"        # Equipment trials
    NIGHT_SHIFT = "night_shift"              # Reduced manning


@dataclass
class ShipboardFireScenario:
    """Represents a shipboard fire emergency scenario"""
    scenario_id: str
    
    # Shipboard-specific parameters
    compartment: ShipboardCompartment
    fire_type: ShipFireType
    ship_condition: ShipCondition
    
    # Fire characteristics
    fire_location: str  # Specific location within compartment
    initial_intensity: float  # 0.0 to 1.0
    initial_size: float      # Square meters
    spread_rate: float       # Per minute
    smoke_production: float  # Visibility reduction rate
    
    # Shipyard-specific factors
    maintenance_activity: str      # What work was being done
    hot_work_permit: bool         # Was hot work authorized
    ventilation_status: float     # 0.0 to 1.0 (reduced during maintenance)
    access_restrictions: List[str] # Blocked routes due to maintenance
    
    # Shipboard crew readiness (maintenance manning levels)
    duty_section: int             # Which duty section (1, 2, or 3)
    crew_experience: float        # Experience level (0.0 to 1.0)
    crew_size: int               # Available personnel
    response_time: float         # Expected response time in minutes
    
    # Equipment status during maintenance period
    fire_main_pressure: float    # Reduced pressure during maintenance
    equipment_condition: float   # Some systems may be down for maintenance
    foam_system_status: bool     # May be isolated
    sprinkler_status: bool       # May be impaired
    
    # Shipyard environment
    civilian_workers: int        # Contractor personnel count
    escape_route_status: List[bool]  # Which routes are available
    communication_status: float # May be degraded during maintenance
    
    # Mission impact
    availability_schedule: bool  # Is this affecting scheduled completion
    critical_system_threat: bool # Does fire threaten critical systems
    
    def to_environment_state(self) -> np.ndarray:
        """Convert Shipboard scenario to environment state vector"""
        
        # Convert compartment to index (1-12)
        compartment_index = list(ShipboardCompartment).index(self.compartment) + 1
        
        return np.array([
            compartment_index,           # Which of 12 compartments
            self.initial_intensity,      # Fire intensity
            self.initial_size / 100.0,   # Normalized fire size
            self.smoke_production,       # Smoke/visibility impact
            self.crew_experience,        # Crew capability
            self.fire_main_pressure,     # Water system status
            0.0,                        # Time elapsed (starts at 0)
            self.civilian_workers / 50.0, # Civilian risk (normalized)
            1.0 - self.ventilation_status, # Ventilation impairment
            self.communication_status,   # Communications capability
            float(self.foam_system_status), # Foam system availability
            self.equipment_condition     # Overall equipment readiness
        ])


class ShipboardScenarioGenerator:
    """Generates Shipboard maintenance fire scenarios"""
    
    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_scenario(self, difficulty: str = "medium") -> ShipboardFireScenario:
        """
        Generate a single Shipboard maintenance fire scenario
        
        Args:
            difficulty: "easy", "medium", "hard", "extreme"
        """
        scenario_id = f"shipboard_maintenance_{random.randint(10000, 99999)}"
        
        # Shipboard compartment (1-12)
        compartment = random.choice(list(ShipboardCompartment))
        
        # Shipyard fire type
        fire_type = random.choice(list(ShipFireType))
        
        # Shipyard working conditions
        ship_condition = random.choice(list(ShipCondition))
        
        # Fire characteristics based on type and difficulty
        fire_params = self._generate_fire_parameters(fire_type, difficulty)
        
        # Shipyard-specific parameters
        maintenance_params = self._generate_maintenance_parameters(ship_condition, difficulty)
        
        # Shipboard crew parameters (maintenance manning)
        crew_params = self._generate_crew_parameters(difficulty)
        
        # Equipment status during maintenance availability
        equipment_params = self._generate_equipment_parameters(difficulty)
        
        return ShipboardFireScenario(
            scenario_id=scenario_id,
            compartment=compartment,
            fire_type=fire_type,
            ship_condition=ship_condition,
            **fire_params,
            **maintenance_params,
            **crew_params,
            **equipment_params
        )
    
    def generate_scenario_batch(self, count: int, difficulty_mix: bool = True) -> List[ShipboardFireScenario]:
        """Generate multiple Shipboard maintenance scenarios with optional difficulty variation"""
        scenarios = []
        
        difficulties = ["easy", "medium", "hard", "extreme"]
        
        for i in range(count):
            if difficulty_mix:
                # Weight towards medium difficulty, but include all types
                weights = [0.2, 0.4, 0.3, 0.1]
                difficulty = np.random.choice(difficulties, p=weights)
            else:
                difficulty = "medium"
            
            scenarios.append(self.generate_scenario(difficulty))
        
        return scenarios
    
    def _generate_fire_parameters(self, fire_type: ShipFireType, difficulty: str) -> Dict:
        """Generate fire-specific parameters"""
        
        # Base parameters by fire type
        fire_configs = {
            ShipFireType.ELECTRICAL_SHORT: {
                "base_intensity": 0.4,
                "base_size": 0.2,
                "spread_rate": 0.6,
                "smoke_production": 0.8,
                "locations": ["electrical_panel", "motor_control", "switchboard", "cable_way"]
            },
            ShipFireType.FUEL_SPILL: {
                "base_intensity": 0.8,
                "base_size": 0.5,
                "spread_rate": 0.9,
                "smoke_production": 0.9,
                "locations": ["fuel_tank", "pump_room", "transfer_station", "bilge"]
            },
            ShipFireType.WELDING_SPARK: {
                "base_intensity": 0.3,
                "base_size": 0.1,
                "spread_rate": 0.4,
                "smoke_production": 0.6,
                "locations": ["work_area", "hull_section", "pipe_joint", "structural_work"]
            },
            ShipFireType.MACHINERY_OVERHEAT: {
                "base_intensity": 0.7,
                "base_size": 0.4,
                "spread_rate": 0.7,
                "smoke_production": 0.8,
                "locations": ["engine_room", "machinery_space", "generator", "compressor"]
            },
            ShipFireType.PAINT_VAPOR: {
                "base_intensity": 0.6,
                "base_size": 0.3,
                "spread_rate": 0.8,
                "smoke_production": 0.95,
                "locations": ["paint_locker", "work_space", "ventilation_duct", "preservation_area"]
            },
            ShipFireType.CUTTING_TORCH: {
                "base_intensity": 0.5,
                "base_size": 0.2,
                "spread_rate": 0.5,
                "smoke_production": 0.7,
                "locations": ["cutting_station", "removal_work", "modification_area", "repair_site"]
            },
            ShipFireType.CHEMICAL_REACTION: {
                "base_intensity": 0.6,
                "base_size": 0.3,
                "spread_rate": 0.8,
                "smoke_production": 0.95,
                "locations": ["chemical_storage", "mixing_area", "cleaning_station", "treatment_room"]
            },
            ShipFireType.INSULATION_FIRE: {
                "base_intensity": 0.4,
                "base_size": 0.6,
                "spread_rate": 0.9,
                "smoke_production": 0.9,
                "locations": ["pipe_lagging", "insulation_work", "thermal_barrier", "acoustic_treatment"]
            }
        }
        
        config = fire_configs[fire_type]
        
        # Difficulty multipliers
        difficulty_mults = {
            "easy": 0.7,
            "medium": 1.0,
            "hard": 1.3,
            "extreme": 1.6
        }
        
        mult = difficulty_mults[difficulty]
        
        return {
            "fire_location": random.choice(config["locations"]),
            "initial_intensity": min(1.0, config["base_intensity"] * mult * np.random.uniform(0.8, 1.2)),
            "initial_size": min(1.0, config["base_size"] * mult * np.random.uniform(0.8, 1.2)),
            "spread_rate": min(1.0, config["spread_rate"] * mult * np.random.uniform(0.9, 1.1)),
            "smoke_production": min(1.0, config["smoke_production"] * np.random.uniform(0.9, 1.1))
        }
    
    def _generate_maintenance_parameters(self, ship_condition: ShipCondition, difficulty: str) -> Dict:
        """Generate maintenance-specific parameters"""
        
        activities = {
            ShipCondition.NORMAL_OPERATIONS: ["routine_maintenance", "system_check", "minor_repair"],
            ShipCondition.HEAVY_MAINTENANCE: ["major_overhaul", "system_replacement", "structural_work"],
            ShipCondition.PRESERVATION_WORK: ["painting", "coating_application", "surface_preparation"],
            ShipCondition.SYSTEM_TESTING: ["equipment_testing", "system_integration", "performance_trials"],
            ShipCondition.NIGHT_SHIFT: ["watchstanding", "security_rounds", "emergency_maintenance"]
        }
        
        return {
            "maintenance_activity": random.choice(activities[ship_condition]),
            "hot_work_permit": random.choice([True, False]),
            "ventilation_status": np.random.uniform(0.3, 0.9),
            "access_restrictions": random.sample(
                ["main_passage", "ladder_well", "escape_trunk", "access_hatch"],
                random.randint(0, 2)
            )
        }
    
    def _generate_crew_parameters(self, difficulty: str) -> Dict:
        """Generate crew readiness parameters"""
        
        difficulty_ranges = {
            "easy": {"experience": (0.7, 0.9), "size": (8, 12), "response": (1, 3)},
            "medium": {"experience": (0.5, 0.8), "size": (5, 8), "response": (2, 5)},
            "hard": {"experience": (0.3, 0.6), "size": (3, 6), "response": (3, 7)},
            "extreme": {"experience": (0.1, 0.4), "size": (2, 4), "response": (5, 10)}
        }
        
        ranges = difficulty_ranges[difficulty]
        
        return {
            "duty_section": random.randint(1, 3),
            "crew_experience": np.random.uniform(*ranges["experience"]),
            "crew_size": random.randint(*ranges["size"]),
            "response_time": np.random.uniform(*ranges["response"])
        }
    
    def _generate_equipment_parameters(self, difficulty: str) -> Dict:
        """Generate equipment status parameters"""
        
        difficulty_ranges = {
            "easy": {"pressure": (0.8, 1.0), "condition": (0.8, 1.0), "systems": 0.9},
            "medium": {"pressure": (0.6, 0.9), "condition": (0.6, 0.9), "systems": 0.7},
            "hard": {"pressure": (0.4, 0.7), "condition": (0.4, 0.7), "systems": 0.5},
            "extreme": {"pressure": (0.2, 0.5), "condition": (0.2, 0.5), "systems": 0.3}
        }
        
        ranges = difficulty_ranges[difficulty]
        
        return {
            "fire_main_pressure": np.random.uniform(*ranges["pressure"]),
            "equipment_condition": np.random.uniform(*ranges["condition"]),
            "foam_system_status": random.random() < ranges["systems"],
            "sprinkler_status": random.random() < ranges["systems"],
            "civilian_workers": random.randint(0, 20),
            "escape_route_status": [random.choice([True, False]) for _ in range(4)],
            "communication_status": np.random.uniform(0.3, 1.0),
            "availability_schedule": random.choice([True, False]),
            "critical_system_threat": random.choice([True, False])
        }


def save_scenarios(scenarios: List[ShipboardFireScenario], filename: str):
    """Save scenarios to JSON file"""
    scenario_dicts = []
    for scenario in scenarios:
        scenario_dict = {
            "scenario_id": scenario.scenario_id,
            "compartment": scenario.compartment.value,
            "fire_type": scenario.fire_type.value,
            "ship_condition": scenario.ship_condition.value,
            "fire_location": scenario.fire_location,
            "initial_intensity": scenario.initial_intensity,
            "initial_size": scenario.initial_size,
            "spread_rate": scenario.spread_rate,
            "smoke_production": scenario.smoke_production,
            "maintenance_activity": scenario.maintenance_activity,
            "hot_work_permit": scenario.hot_work_permit,
            "ventilation_status": scenario.ventilation_status,
            "access_restrictions": scenario.access_restrictions,
            "duty_section": scenario.duty_section,
            "crew_experience": scenario.crew_experience,
            "crew_size": scenario.crew_size,
            "response_time": scenario.response_time,
            "fire_main_pressure": scenario.fire_main_pressure,
            "equipment_condition": scenario.equipment_condition,
            "foam_system_status": scenario.foam_system_status,
            "sprinkler_status": scenario.sprinkler_status,
            "civilian_workers": scenario.civilian_workers,
            "escape_route_status": scenario.escape_route_status,
            "communication_status": scenario.communication_status,
            "availability_schedule": scenario.availability_schedule,
            "critical_system_threat": scenario.critical_system_threat
        }
        scenario_dicts.append(scenario_dict)
    
    with open(filename, 'w') as f:
        json.dump(scenario_dicts, f, indent=2)


if __name__ == "__main__":
    # Demo scenario generation
    generator = ShipboardScenarioGenerator(seed=42)
    
    print("🔥 SHIPBOARD FIRE SCENARIO GENERATOR DEMO")
    print("=" * 50)
    
    # Generate sample scenarios
    easy_scenario = generator.generate_scenario("easy")
    hard_scenario = generator.generate_scenario("hard")
    extreme_scenario = generator.generate_scenario("extreme")
    
    scenarios = [easy_scenario, hard_scenario, extreme_scenario]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n📋 SCENARIO {i+1}:")
        print(f"   ID: {scenario.scenario_id}")
        print(f"   Compartment: {scenario.compartment.value}")
        print(f"   Fire Type: {scenario.fire_type.value.title()}")
        print(f"   Location: {scenario.fire_location}")
        print(f"   Condition: {scenario.ship_condition.value.title()}")
        print(f"   Intensity: {scenario.initial_intensity:.2f}")
        print(f"   Crew Experience: {scenario.crew_experience:.2f}")
        print(f"   Equipment: {scenario.equipment_condition:.2f}")
        print(f"   Civilians: {scenario.civilian_workers}")
    
    # Generate batch
    print(f"\n🎲 Generating 100 mixed-difficulty scenarios...")
    batch = generator.generate_scenario_batch(100, difficulty_mix=True)
    
    print(f"✅ Generated {len(batch)} scenarios")
    
    # Stats
    fire_types = {}
    ship_conditions = {}
    for scenario in batch:
        fire_types[scenario.fire_type.value] = fire_types.get(scenario.fire_type.value, 0) + 1
        ship_conditions[scenario.ship_condition.value] = ship_conditions.get(scenario.ship_condition.value, 0) + 1
    
    print(f"\n📊 SCENARIO STATISTICS:")
    print(f"   Fire Types: {fire_types}")
    print(f"   Ship Conditions: {ship_conditions}")
