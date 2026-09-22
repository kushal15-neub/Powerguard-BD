import itertools

# ============================================================
# POWERGUARD BD - ENERGY MANAGEMENT OPTIMIZER
# ============================================================

devices = [
    {"name": "Router", "power": 8, "priority": 1},
    {"name": "LED Light", "power": 10, "priority": 1},
    {"name": "Fan", "power": 25, "priority": 2},
    {"name": "Laptop", "power": 45, "priority": 2},
    {"name": "Extra Light", "power": 10, "priority": 3},
]


# Battery information
battery_capacity_wh = 100

# Predicted outage duration
outage_duration_hours = 3


# ============================================================
# CALCULATE BEST DEVICE COMBINATION
# ============================================================

best_solution = None

for combination in itertools.product([0, 1], repeat=len(devices)):

    selected_devices = []

    total_power = 0
    total_priority = 0

    for device, selected in zip(devices, combination):

        if selected == 1:
            selected_devices.append(device)
            total_power += device["power"]

            # Higher priority = more important
            total_priority += (4 - device["priority"]) * 100

    # No device selected
    if total_power == 0:
        continue

    # Required energy
    required_energy = total_power * outage_duration_hours

    # Check battery constraint
    if required_energy <= battery_capacity_wh:

        solution = {
            "devices": selected_devices,
            "power": total_power,
            "energy": required_energy,
            "priority_score": total_priority,
        }

        # Select solution with highest priority score
        if (
            best_solution is None
            or solution["priority_score"] > best_solution["priority_score"]
        ):
            best_solution = solution


# ============================================================
# DISPLAY RESULT
# ============================================================

print("=" * 60)
print("POWERGUARD BD - ENERGY MANAGEMENT RESULT")
print("=" * 60)

print(f"Battery capacity: {battery_capacity_wh} Wh")
print(f"Expected outage:  {outage_duration_hours} hours")

print("\nRecommended devices:")

for device in best_solution["devices"]:
    print(
        f"  ON  -> {device['name']}"
        f" | {device['power']} W"
        f" | Priority {device['priority']}"
    )

print("\nTotal power:")
print(f"{best_solution['power']} W")

print("\nRequired energy:")
print(f"{best_solution['energy']} Wh")

print("\nRemaining battery:")
print(f"{battery_capacity_wh - best_solution['energy']} Wh")

print("\n" + "=" * 60)
