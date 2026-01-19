import matplotlib.pyplot as plt

# --- DONNÉES EXTRAITES DE NOS LOGS ---
epochs = list(range(1, 31))

train_loss = [
    2.6566, 1.0279, 0.6827, 0.5426, 0.4472, 0.3792, 0.3229, 0.2738, 0.2331, 0.1992,
    0.1599, 0.1284, 0.1146, 0.0911, 0.0543, 0.0332, 0.0240, 0.0193, 0.0206, 0.0250,
    0.0124, 0.0072, 0.0061, 0.0056, 0.0053, 0.0053, 0.0047, 0.0041, 0.0039, 0.0037
]

val_loss = [
    1.5508, 0.8292, 0.6817, 0.6204, 0.5573, 0.5518, 0.5051, 0.4944, 0.4997, 0.5071,
    0.5196, 0.5434, 0.5264, 0.5571, 0.5506, 0.5716, 0.5950, 0.6327, 0.6691, 0.6883,
    0.6839, 0.7036, 0.7225, 0.7398, 0.7538, 0.7701, 0.7842, 0.7894, 0.8051, 0.8138
]

plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid') 

plt.plot(epochs, train_loss, label='Train Loss', color='#2ecc71', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', color='#e74c3c', linewidth=2) 


best_epoch = 8
best_val = 0.4944
plt.axvline(x=best_epoch, color='black', linestyle='--', alpha=0.5, label='Arrêt Optimal (Epoch 8)')
plt.scatter(best_epoch, best_val, color='black', zorder=5)
plt.text(best_epoch + 1, best_val, f'Min Val Loss: {best_val}', fontsize=10, verticalalignment='bottom')


plt.title("Évolution de l'Apprentissage (Overfitting après Epoch 8)", fontsize=14, pad=15)
plt.xlabel("Époques", fontsize=12)
plt.ylabel("CTC Loss", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)


plt.savefig("courbe_loss.png", dpi=300)
print("Graphique généré : courbe_loss.png")
plt.show()