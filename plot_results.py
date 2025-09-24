import numpy as np
import matplotlib.pyplot as plt

data = np.load('results.npz')
score_history = data['score_history']
avg_score_history = data['avg_score_history']

plt.figure(figsize=(10,5))
plt.plot(score_history, label='Score per Episode')
plt.plot(avg_score_history, label='Average Score (last 100)')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Training Progress')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('training_progress.png')
plt.show()