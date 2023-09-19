import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

c = np.array([5.6,5.2,4.8,4.5,4.4,2.9,2.7])
idade = np.array([1,2,3,4,5,6,7])
ex_1 = pd.DataFrame({"Conversão alimentar":c, "Idade":idade,})
print(ex_1.corr())
print(np.std(ex_1['Conversão alimentar'],ddof=1)) # 1 (Amostral) || 2 (Populacional)
plt.plot(ex_1["Conversão alimentar"],ex_1["Idade"])
plt.xlabel("Conversão alimentar")
plt.ylabel("Idade")
plt.title("Correlação entre alimentação e idade")
plt.show()