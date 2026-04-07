def generate_report(label: str, probs: dict, drift: dict) -> str:
    """
    Generate medical report with safety checks.
    
    Args:
        label: Predicted class name
        probs: Dictionary of class probabilities
        drift: Data drift monitoring results
        
    Returns:
        Formatted report string
    """
    if not probs or not isinstance(probs, dict):
        return "ERROR: Invalid probability data"
    
    # Sort confidence scores
    top = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    # Build confidence ranking safely
    confidence_lines = []
    for i, (cls, prob) in enumerate(top[:3]):  # Top 3 predictions
        confidence_lines.append(f"{i+1}. {cls}: {prob:.3f}")
    confidence_text = "\n".join(confidence_lines)
    
    # Drift warning
    drift_note = ""
    if drift and drift.get("alert"):
        drift_note = f"\n⚠️ Data shift alert: {drift['alert']} (score={drift.get('score', 0):.2f})"
    
    return (
        "OCT AI Preliminary Report\n"
        "=========================\n"
        f"Predicted class: {label}\n"
        f"Confidence ranking:\n{confidence_text}\n\n"
        "Clinical note:\n"
        "- This output is for research/triage support only and is NOT a final diagnosis.\n"
        "- Please correlate with symptoms, fundus, and physician assessment.\n"
        f"{drift_note}\n"
    )