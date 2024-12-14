import numpy as np

# Lista de strings com os tempos de execução
dados = np.loadtxt('data_mpi.txt')
data_proc = {}
for line in dados:
    key = int(line[0])
    value = line[1]
    if key not in data_proc:
        data_proc[key] = []
    data_proc[key].append(value)  # Adiciona o valor à lista
print(data_proc)

for chave, valores in data_proc.items():
    menores_cinco = np.sort(valores)[:9]

    media = np.mean(menores_cinco)
    desvio_padrao = np.std(menores_cinco)
    
    # Exibe o resultado
    print(f"{chave}\t{media:.2f} +- {desvio_padrao:.2f}")
