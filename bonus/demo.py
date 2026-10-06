"""Demo script running 5 distinct query scenarios for HybridMemoryAgent."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent


def main() -> int:
    print("=" * 70)
    print("  BONUS CHALLENGE DEMO — Hybrid Memory Agent for Personal AI Assistant")
    print("=" * 70)

    agent = HybridMemoryAgent(user_id="u_001")

    # Seed episodic memories for u_001
    print("\n[Step 1] Seeding episodic memories into vector store...")
    seed_memories = [
        ("Tôi đã đọc xong cuốn sách Kubernetes in Action, học cách deployment và rolling update HPA.", "cloud"),
        ("Tài liệu hướng dẫn tự động mở rộng (Auto-scaling) hạ tầng khi traffic tăng đột biến.", "cloud"),
        ("Ghi chú về mã hóa dữ liệu AES-256 và bảo mật xác thực OAuth2 / JWT cho REST API.", "security"),
        ("Tóm tắt thảo luận về việc di chuyển database từ PostgreSQL sang Qdrant vector database.", "database"),
        ("Nhật ký nghiên cứu mô hình ngôn ngữ lớn LLM và chiến lược RAG agentic retrieval.", "ai_ml"),
    ]
    for mem_text, topic in seed_memories:
        agent.remember(text=mem_text, topic=topic)
        print(f"  + Remembered: {mem_text[:60]}...")

    queries = [
        (
            "1. Pure Vector Hit",
            "Tôi đã đọc gì về Kubernetes?",
            "Hỏi cụ thể tài liệu đã lưu — kỳ vọng Vector hit chính xác k8s memory.",
        ),
        (
            "2. Profile Context Needed",
            "Recommend nội dung gì nên đọc tiếp theo?",
            "Cần thông tin topic_affinity và preferred_language từ Feast profile.",
        ),
        (
            "3. Fresh Activity Needed",
            "Tôi đang quan tâm tới lĩnh vực nào gần đây?",
            "Cần thông tin queries_last_hour và distinct_topics từ Feast streaming.",
        ),
        (
            "4. Paraphrase Vector Win",
            "Tài liệu về giải pháp tự động mở rộng hạ tầng máy chủ?",
            "Dùng từ diễn đạt lại (không trùng keyword verbatim) — Vector search chiếm ưu thế.",
        ),
        (
            "5. Mixed (Hybrid + Profile Context)",
            "Cho tôi tóm tắt kinh nghiệm về cloud security và kiến trúc hạ tầng",
            "Kết hợp cả Episodic Memory (OAuth2/Auto-scaling) và User Profile.",
        ),
    ]

    print("\n" + "=" * 70)
    print("  EXECUTING 5 DEMO QUERIES")
    print("=" * 70)

    for title, q, desc in queries:
        print(f"\n>>> QUERY SCENARIO: [{title}]")
        print(f"    Description: {desc}")
        print(f"    User Query : {q!r}")
        print("-" * 50)
        context = agent.recall(query=q, top_k=2)
        print(context)
        print("-" * 50)

    print("\n[SUCCESS] All 5 queries completed. Exit 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
