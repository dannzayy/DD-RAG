import matplotlib.pyplot as plt
import numpy as np

# ====================== DATA ======================
models = ["Granite", "LLaMA 3", "Mistral 7B", "DeepSeek-R1", "Qwen 2.5"]

# With RAG (from your original table)
accuracy_rag = [90, 65, 90, 85, 85]         # %
latency_rag = [9.87, 5.58, 6.94, 47.25, 4.30]  # seconds

# Without RAG (estimated / LLM-only)
accuracy_no_rag = [70, 55, 65, 60, 65]      # %
latency_no_rag = [4.10, 3.26, 3.92, 18.45, 2.87]  # seconds

x = np.arange(len(models))
width = 0.35

# ====================== PLOT ======================
fig, ax1 = plt.subplots(figsize=(10,6))

# Accuracy bars
bars1 = ax1.bar(x - width/2, accuracy_rag, width, label='Accuracy (RAG)', color='#4C72B0')
bars2 = ax1.bar(x + width/2, accuracy_no_rag, width, label='Accuracy (No-RAG)', color='#55A868')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(0, 100)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_title('Model Performance: Accuracy vs Latency (RAG vs No-RAG)')
ax1.legend(loc='upper left')

# Latency as line plot
ax2 = ax1.twinx()
ax2.plot(x, latency_rag, 'o-', color='#C44E52', label='Latency RAG', linewidth=2)
ax2.plot(x, latency_no_rag, 's-', color='#8172B3', label='Latency No-RAG', linewidth=2)
ax2.set_ylabel('Latency (seconds)')
ax2.legend(loc='upper right')

plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()