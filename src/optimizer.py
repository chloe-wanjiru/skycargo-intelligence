from scipy.optimize import linprog

def optimize_payload(max_allowable_payload_kg: float, pass_weight_kg: float, bag_weight_kg: float) -> dict:
    #Calculates remaining available capacity for commercial cargo to maximize revenue without exceeding MTOW margins.
    remaining_capacity = max_allowable_payload_kg - (pass_weight_kg + bag_weight_kg)
    
    if remaining_capacity <= 0:
        return {
            "status": "OVERWEIGHT WARNING",
            "available_cargo_capacity_kg": 0.0,
            "cargo_revenue_est": 0.0,
            "weight_status_pct": 100.0
        }
    
    # Objective: Maximize revenue per kg ($2.50 per kg standard freight)
    c = [-2.50]  
    A = [[1.0]]  
    b = [remaining_capacity] 
    
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, remaining_capacity))
    
    allocated_cargo = res.x[0] if res.success else 0.0
    total_used = pass_weight_kg + bag_weight_kg + allocated_cargo
    
    return {
        "status": "OPTIMAL",
        "available_cargo_capacity_kg": round(allocated_cargo, 2),
        "cargo_revenue_est": round(allocated_cargo * 2.50, 2),
        "weight_status_pct": round((total_used / max_allowable_payload_kg) * 100, 2)
    }
def calculate_delay_penalties(dep_delay_mins: float, passenger_count: int) -> dict:
    #Calculates operational loss liabilities based on departure delay length.
    food_voucher_cost = 0.0
    hotel_cost = 0.0
    compensation_cost = 0.0
    
    # 2+ Hours (120 mins): Food & Beverage Vouchers ($20/passenger)
    if dep_delay_mins >= 120:
        food_voucher_cost = passenger_count * 20.0
        
    # 4+ Hours (240 mins): Cash Compensation ($400 average regulatory payout/passenger)
    if dep_delay_mins >= 240:
        compensation_cost = passenger_count * 400.0
        
    # 6+ Hours (360 mins) or Overnight: Hotel + Transport ($150/passenger)
    if dep_delay_mins >= 360:
        hotel_cost = passenger_count * 150.0

    total_delay_loss = food_voucher_cost + compensation_cost + hotel_cost
    
    return {
        "food_voucher_cost": food_voucher_cost,
        "hotel_cost": hotel_cost,
        "compensation_cost": compensation_cost,
        "total_delay_loss": total_delay_loss
    }