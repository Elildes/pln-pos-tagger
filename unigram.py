from itertools import count
from optparse import Values
import funcoes
import operator  # max()
from sklearn.metrics import confusion_matrix, classification_report

# Observação: antes de rodar esta aplicação, veja as dependências no final deste arquivo.

# 1ª PARTE: 'Treino' (usando o corpus de treino; no caso do Penn Treebank usar Seções 0-18):

# abre arquivo de texto: corpus de treino
dir = r'/home/elildes/Git/ufrn-2022.2-pln/trab1/Secs0-18-training.txt'
aqq_treino = open(dir, 'r')

# Teste: imprime lista de word_tag + valor
#funcoes.print_normal(aqq_treino)

# Teste: imprime lista ordenada de word_tag + valor
#funcoes.print_sorted(aqq_treino)

"""
gera dicionário de ocorrências de palavra_tag = valor
Formato de saída: d = = {'876_cd': 1, '3.34_cd': 1, '2.59_cd': 1, 'kuse_nnp': 1, '46.125_cd': 1, 'temblor-prone_jj': 1}
"""
d = funcoes.count_word_tag(aqq_treino)

# fecha arquivo
aqq_treino.close()

# Teste: imprimir dicionário de ocorrências
#funcoes.print_dic(d)