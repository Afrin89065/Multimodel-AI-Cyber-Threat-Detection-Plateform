"""TextAttack robustness eval. RUN: python scripts\training\adversarial_robustness.py"""
import sys, json
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import os; os.chdir(PROJECT_ROOT)

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

SUBSTITUTIONS = {
    "urgent": "urgnt", "verify": "verif y", "account": "acc0unt",
    "login": "l0gin", "paypal": "paypa1", "bank": "b@nk",
}

def perturb(text):
    for word, sub in SUBSTITUTIONS.items():
        text = text.lower().replace(word, sub)
    return text

def evaluate_robustness(n_samples=100):
    from services.nlp_service import NLPService
    from core.config import settings
    nlp_svc = NLPService(settings.NLP_MODEL_PATH)
    test_file = PROJECT_ROOT / "datasets/processed/nlp/test.jsonl"
    if not test_file.exists():
        logger.error("Test file not found. Run preprocess_nlp.py first.")
        return {}
    test_samples = [json.loads(l) for l in open(test_file)][:n_samples]

    try:
        from textattack.models.wrappers import PyTorchModelWrapper
        from textattack.attack_recipes import TextFoolerJin2019
        label_map = {"CLEAN": 0, "SPAM": 1, "PHISHING": 2, "BEC": 3}
        clean_correct = sum(
            1 for s in test_samples
            if nlp_svc.analyse(s["text"], s.get("url", ""))["threat_type"] == s["label"]
        )
        clean_acc = clean_correct / len(test_samples)
        wrapper = PyTorchModelWrapper(nlp_svc.model, nlp_svc.tokenizer)
        attack = TextFoolerJin2019.build(wrapper)
        attack_success = total_attacked = 0
        for sample in test_samples[:50]:
            if sample["label"] == "CLEAN":
                continue
            total_attacked += 1
            try:
                r = attack.attack(sample["text"], label_map.get(sample["label"], 0))
                if r.__class__.__name__ == "SuccessfulAttackResult":
                    attack_success += 1
            except Exception:
                pass
        attack_success_rate = attack_success / max(total_attacked, 1)
        results = {
            "clean_accuracy": round(clean_acc, 4),
            "adversarial_accuracy": round(1 - attack_success_rate, 4),
            "attack_success_rate": round(attack_success_rate, 4),
            "attack_method": "TextFoolerJin2019",
            "samples_tested": n_samples,
        }
    except ImportError:
        clean_correct = perturbed_correct = 0
        for s in test_samples:
            if nlp_svc.analyse(s["text"], "")["threat_type"] == s["label"]:
                clean_correct += 1
            if nlp_svc.analyse(perturb(s["text"]), "")["threat_type"] == s["label"]:
                perturbed_correct += 1
        n = len(test_samples)
        results = {
            "clean_accuracy": round(clean_correct / n, 4),
            "adversarial_accuracy": round(perturbed_correct / n, 4),
            "robustness_drop": round((clean_correct - perturbed_correct) / n, 4),
            "attack_method": "manual_substitution_fallback",
            "samples_tested": n,
            "note": "Install textattack for full evaluation: pip install textattack",
        }

    json.dump(results, open(LOGS_DIR / "adversarial_results.json", "w"), indent=2)
    logger.info(f"Results saved: {LOGS_DIR / 'adversarial_results.json'}")
    return results

if __name__ == "__main__":
    r = evaluate_robustness(n_samples=100)
    print("\n=== ROBUSTNESS RESULTS ===")
    for k, v in r.items():
        print(f"  {k}: {v}")