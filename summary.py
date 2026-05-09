from collections import Counter
from preprocess import clean_text

positive_keywords = [
    "good", "great", "excellent", "amazing", "fast",
    "beautiful", "premium", "sharp", "stylish"
]

negative_keywords = [
    "bad", "poor", "broken", "damage", "slow",
    "heat", "scratched", "terrible", "drain"
]

product_features = [
    "battery", "camera", "display", "screen",
    "speaker", "design", "performance", "color",
    "case", "back"
]


def generate_feature_summary(feature_results):

    summaries = {}

    # ✅ FIX: ensure we are working with dict
    if not isinstance(feature_results, dict):
        return {"error": "feature_results must be a dictionary"}

    for feature, result in feature_results.items():

        pos = result.get("positive", 0)
        neg = result.get("negative", 0)
        total = pos + neg

        if total == 0:
            summaries[feature] = (
                f"There was very little discussion around {feature}, "
                f"so no clear pattern could be identified."
            )
            continue

        positive_ratio = pos / total

        if positive_ratio >= 0.75:
            summaries[feature] = (
                f"The {feature} received strong appreciation in most reviews. "
                f"Users frequently highlighted its good performance and reliability, "
                f"showing a clearly positive overall experience."
            )

        elif positive_ratio >= 0.55:
            summaries[feature] = (
                f"Feedback on the {feature} was mostly favorable. "
                f"Many users pointed out satisfying performance, though a few mentioned minor concerns."
            )

        elif positive_ratio >= 0.40:
            summaries[feature] = (
                f"Opinions on the {feature} were mixed. "
                f"Some users found it useful, while others reported drawbacks."
            )

        elif positive_ratio >= 0.20:
            summaries[feature] = (
                f"The {feature} showed noticeable concerns in several reviews. "
                f"Users mentioned performance limitations and improvement areas."
            )

        else:
            summaries[feature] = (
                f"The {feature} was one of the weakest aspects in user feedback. "
                f"Recurring problems and dissatisfaction were reported."
            )

    return summaries