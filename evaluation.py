import matplotlib.pyplot as plt

# Model names
models = ["Logistic Regression", "Random Forest"]

# Accuracy values from your results
accuracy = [81.69, 79.77]

plt.figure(figsize=(6,4))

bars = plt.bar(models, accuracy)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0,100)

# Display values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f"{height:.2f}%",
        ha="center"
    )

plt.tight_layout()

# Save image
plt.savefig("model_accuracy.png", dpi=300)

# Show chart
plt.show()

print("Chart saved as model_accuracy.png")