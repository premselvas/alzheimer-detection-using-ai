import pickle
import matplotlib.pyplot as plt

with open("training_history.pkl", "rb") as f:
    history = pickle.load(f)

plt.figure()
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Graph of Hybrid Architecture')
plt.legend()
plt.grid(True)

plt.savefig("loss_graph_hybrid_architecture.png", dpi=300, bbox_inches="tight")
plt.show()
