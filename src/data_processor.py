"""
Data processing module: Extracts, cleans, and reconstructs customer-support
conversation pairs from the Twitter Customer Support dataset archive.
"""

import os
import re
import sys
import zipfile
import logging
from pathlib import Path

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import Optional, Tuple, Dict, List
import pandas as pd

from config import RAW_DATA_DIR, BRAND_HANDLE, KB_PATH

logger = logging.getLogger(__name__)


def clean_tweet_text(text: str) -> str:
    """
    Cleans raw tweet text by removing mentions, normalizing URLs, and trimming whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Decode HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    # Remove @mentions at start or throughout
    text = re.sub(r"@\w+", "", text)
    # Normalize URLs
    text = re.sub(r"https?://t\.co/\w+", "[URL]", text)
    text = re.sub(r"https?://\S+", "[URL]", text)
    # Normalize multiple whitespace and newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_apple_conversations(
    zip_path: str = r"D:\archive.zip",
    csv_filename_in_zip: str = "twcs/twcs.csv",
    output_csv_path: Optional[Path] = None,
    max_chunks: int = 10,
    chunk_size: int = 100000,
) -> pd.DataFrame:
    """
    Streams through twcs.csv inside the zip file, extracts AppleSupport conversation pairs,
    and returns a cleaned DataFrame.
    """
    output_csv = output_csv_path or (RAW_DATA_DIR / "sample_apple.csv")
    if output_csv.exists() and output_csv.stat().st_size > 1000:
        logger.info(f"Loading existing cleaned dataset from {output_csv}")
        return pd.read_csv(output_csv)

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Archive file not found at {zip_path}")

    logger.info(f"Extracting AppleSupport conversations from {zip_path}...")

    # We will accumulate inbound tweets and outbound Apple tweets across chunks
    inbound_tweets: Dict[int, Dict] = {}
    outbound_apple: List[Dict] = []

    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_filename_in_zip) as f:
            for chunk_idx, chunk in enumerate(pd.read_csv(f, chunksize=chunk_size)):
                if chunk_idx >= max_chunks:
                    break
                logger.info(f"Processing chunk {chunk_idx + 1}/{max_chunks}...")

                # Filter Apple outbound
                apple_replies = chunk[
                    (chunk["author_id"] == BRAND_HANDLE)
                    & (chunk["inbound"] == False)
                    & (chunk["in_response_to_tweet_id"].notna())
                ]
                for _, row in apple_replies.iterrows():
                    outbound_apple.append({
                        "apple_tweet_id": int(row["tweet_id"]),
                        "apple_text": row["text"],
                        "in_response_to_tweet_id": int(row["in_response_to_tweet_id"]),
                        "created_at": row["created_at"],
                    })

                # Filter inbound customer tweets that might be referenced
                cust_inbound = chunk[chunk["inbound"] == True]
                for _, row in cust_inbound.iterrows():
                    tid = int(row["tweet_id"])
                    # Store only if it might be an Apple conversation
                    text_lower = str(row["text"]).lower()
                    if "apple" in text_lower or "@applesupport" in text_lower:
                        inbound_tweets[tid] = {
                            "customer_tweet_id": tid,
                            "customer_text": row["text"],
                            "author_id": row["author_id"],
                        }

    # Match outbound Apple replies to customer tweets
    matched_pairs = []
    for reply in outbound_apple:
        parent_id = reply["in_response_to_tweet_id"]
        if parent_id in inbound_tweets:
            cust = inbound_tweets[parent_id]
            clean_cust = clean_tweet_text(cust["customer_text"])
            clean_reply = clean_tweet_text(reply["apple_text"])
            # Filter out non-informative single word or empty tweets
            if len(clean_cust) >= 15 and len(clean_reply) >= 15 and clean_cust != "[URL]":
                matched_pairs.append({
                    "customer_tweet_id": cust["customer_tweet_id"],
                    "customer_text": clean_cust,
                    "raw_customer_text": cust["customer_text"],
                    "apple_tweet_id": reply["apple_tweet_id"],
                    "apple_reply": clean_reply,
                    "raw_apple_reply": reply["apple_text"],
                    "created_at": reply["created_at"],
                })

    df_pairs = pd.DataFrame(matched_pairs).drop_duplicates(subset=["customer_text"])
    logger.info(f"Successfully reconstructed {len(df_pairs)} unique conversation pairs.")

    df_pairs.to_csv(output_csv, index=False)
    logger.info(f"Saved cleaned conversation pairs to {output_csv}")
    return df_pairs


def build_knowledge_base(
    df_pairs: pd.DataFrame,
    output_kb_path: Optional[Path] = None,
    max_entries: int = 500,
) -> List[Dict]:
    """
    Builds a curated retrieval knowledge base of historical AppleSupport resolutions.
    """
    kb_path = output_kb_path or KB_PATH
    if kb_path.exists():
        import json
        with open(kb_path, "r", encoding="utf-8") as f:
            return json.load(f)

    kb_entries = []
    # Keywords to categorize resolution patterns
    patterns = [
        ("battery", "battery_power_hardware"),
        ("drain", "battery_power_hardware"),
        ("charge", "battery_power_hardware"),
        ("heat", "battery_power_hardware"),
        ("update", "software_update_glitch"),
        ("ios", "software_update_glitch"),
        ("freeze", "software_update_glitch"),
        ("crash", "software_update_glitch"),
        ("bluetooth", "software_update_glitch"),
        ("wifi", "software_update_glitch"),
        ("password", "account_security_icloud"),
        ("apple id", "account_security_icloud"),
        ("icloud", "account_security_icloud"),
        ("locked", "account_security_icloud"),
        ("subscription", "billing_subscription"),
        ("refund", "billing_subscription"),
        ("charged", "billing_subscription"),
        ("purchase", "billing_subscription"),
        ("screen", "physical_damage_repair"),
        ("cracked", "physical_damage_repair"),
        ("repair", "physical_damage_repair"),
        ("genius bar", "physical_damage_repair"),
        ("water", "physical_damage_repair"),
    ]

    for _, row in df_pairs.iterrows():
        c_text = str(row["customer_text"]).lower()
        a_text = str(row["apple_reply"])

        assigned_intent = "general_feedback_inquiry"
        for kw, intent in patterns:
            if kw in c_text:
                assigned_intent = intent
                break

        kb_entries.append({
            "id": int(row["apple_tweet_id"]),
            "intent": assigned_intent,
            "customer_query": row["customer_text"],
            "historical_resolution": a_text,
        })
        if len(kb_entries) >= max_entries:
            break

    import json
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb_entries, f, indent=2, ensure_ascii=False)
    logger.info(f"Built retrieval knowledge base with {len(kb_entries)} items at {kb_path}")
    return kb_entries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = extract_apple_conversations()
    print("Dataset extracted shape:", df.shape)
    kb = build_knowledge_base(df)
    print("Knowledge base entries:", len(kb))
