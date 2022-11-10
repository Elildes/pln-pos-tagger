from itertools import count
from optparse import Values
import funcoes
import operator  # max()
from sklearn.metrics import confusion_matrix, classification_report

# Observação: antes de rodar esta aplicação, veja as dependências no final deste arquivo.

# 1ª PARTE: 'Treino' (usando o corpus de treino; no caso do Penn Treebank usar Seções 0-18):

# abre arquivo de texto: corpus de treino
dir = r'Secs0-18-training.txt'
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

"""
gera Dicionário de Palavras, Tags e Ocorrências extraindo a sequência de correspondentes tags e valores de cada palavra.
Formato de saída: Ex.: dicio_tag_val = {'a': {'DT': 100, 'AR': 20}, 'the': {'AR': 2, 'PP': 25}}
"""
dicio_tag_val = funcoes.split_key_tag_value(d)

# Teste: imprimir dicionário
#print(dicio_tag_val)
#funcoes.print_dic(dicio_tag_val)

# Teste: imprime valores do dicionário pela chave
#print('Valores de a:')
#funcoes.print_values('a', dicio_tag_val)
#print('Valores de the:')
#funcoes.print_values('the', dicio_tag_val)

# Teste: contar chave no dicionário
#funcoes.count_key('a', dicio_tag_val)

"""
salva o Dicionário de Palavras, Tags e Ocorrências
Formato de saída: Ex.: dicio_tag_val = {'a': {'DT': 100, 'AR': 20}, 'the': {'AR': 2, 'PP': 25}}
"""
funcoes.save_dic_arq(dicio_tag_val, 'dicio_tag_val.txt')

"""
cria Dicionário de Treino: com as estimativas de probabilidades por contagem de ocorrências.
Formato de saída: Ex.: dicio_treino = {'pierre': {'nnp': 6, 'prob': 1.0}, ',': {',': 46522, 'prob': 1.0}, (...)}
"""
dicio_treino = funcoes.dicio_treino(dicio_tag_val)

# Teste: imprime dicionário com chaves, tags, valores/máximo e probabilidades
#print(dicio_treino)

"""
salva o Dicionário de Treino em um arquivo
Formato de saída: Ex.: dicio_treino = {'pierre': {'nnp': 6, 'prob': 1.0}, ',': {',': 46522, 'prob': 1.0}, (...)}
"""
funcoes.save_dic_arq(dicio_treino, 'dicio_treino.txt')

# Teste:
print('\n1ª PARTE ok!')

# 2ª PARTE: 'Teste de desenvolvimento ou validação' (usando o corpus de desenvolvimento, validação ou heldout;  no caso do Penn Treebank usar Seções 19-21):

# abre arquivo de texto: corpus de desenvolvimento
dir = r'Secs19-21 - development.txt'
arq_develop = open(dir, 'r')

"""
Gera Dicionário/Lista de Testes: extrair a sequências de palavras de um corpus
Formato de saída: Ex.: dicio_teste = ['the', 'Arizona', 'Corporations', 'Commission', (...)]
"""
dicio_teste = funcoes.dicio_teste(arq_develop)
# Teste:
#print("\nDicionário de Testes - dicio_teste:", dicio_teste)

# fecha arquivo
arq_develop.close()

"""
salva O Dicionário de Testes em um arquivo
Formato de saída: Ex.: lista = ['the', 'Arizona', 'Corporations', 'Commission', (...)]
"""
funcoes.save_dic_arq(dicio_teste, 'dicio_teste.txt')

"""
Entradas: lê Dicionário de Testes e Dicionário de Treino
Saída: gera arquivo de Lista Real (lista_real) com os pares word e tag
Formato de saída: Ex.: lista_real = ['the_dt', 'arizona_nnp', 'corporations_nns', 'commission_nnp', (...)]
"""
lista_real = funcoes.driver_pos_tagger(dicio_treino, dicio_teste)

"""
Entrada: Arquivo corpus de desenvolvimento
Saída: Lista de Tag Preditiva contendo as tags de cada palavra na sequência
Formato de saída: Ex.: lista_tag_pred = [['dt', 'nnp', 'nns', (...)]
"""
dir = r'Secs19-21 - development.txt'
arq_develop = open(dir, 'r')

"""
Entrada: Arquivo corpus
Saída: Lista de Tag Preditiva contendo as tags de cada palavra na sequência
Formato de saída: Ex.: lista_tag_pred = [['dt', 'nnp', 'nns', (...)]
"""
lista_tag_pred_treino = funcoes.listar_tag_pred(arq_develop)

# fecha arquivo
arq_develop.close()

"""
Entrada: Lista Real (lista_real)
Saída: Lista de Tag Real de Treino (lista_tag_real_treino)
Formato de saída: Ex.: lista_tag_real_treino = ['dt', 'nnp', 'nns, 'nnp]
"""
lista_tag_real_treino = funcoes.listar_tag_real(lista_real)

# Teste
print('\nLista tag_real tamanho:', len(lista_tag_real_treino))
print('\nLista tag_pred tamanho:', len(lista_tag_pred_treino))



"""
Comparar a Lista de Tag Real de Treino (lista_tag_real_treino) e Lista de Tag Preditiva  de Treino (lista_tag_pred_treino)
Entradas: Lista de tags Real de Treino, Lista de tags Preditiva de Treino, Lista de Labels
Saída: Matriz de Confusão
"""
list_labels = funcoes.listar_labels(lista_tag_pred_treino)  # gera Lista de Rótulos
print('\nMatriz de confusão:\n', confusion_matrix(lista_tag_real_treino, lista_tag_pred_treino, labels=list_labels))


# Teste:
print('\n2ª PARTE ok!')

# 3ª PARTE: 'Avaliação final' (usando o corpus de teste final; no caso do Penn Treebank usar Seções 22-24):



# idem para o arquivo avalaçãi final--------------------????????????


# Quando os testes acima estiverem ok:
# comparação dos pares words/tags do arquivo real (lista_word_tag_real.txt) com o arquivo preditivo (Secs22-24 - testing.txt)

# Os acertos e erros da comparação deverão ser adicionados na Matriz de Confusão

# ------------------------------------ INICIO DO CÓDIGO -----------------------------

# abre arquivo de texto: corpus de desenvolvimento
#dir = r'Secs22-24 - testing.txt'
#arq_testing = open(dir, 'r')

# extrair a sequência de palavras de um corpus
# formato: lista [w1, w2, w3, ...]
#lista_preditivos = funcoes.dicio_teste(arq_testing)
#lista_reais = ???
# Teste:
#print("\nLista preditivo:", lista_preditivos)
#print('\nLista preditivo len:', len(lista_preditivos))

# fecha arquivo
#arq_develop.close()

# Teste:
print('\n3ª PARTE ok!')

# Teste: final
print('\n\nOK, fim código!!!')


"""
Lista de dependências:
Linux: 
. Python: sudo apt-get install python3: 
. Pip3: sudo apt-get install python3-pip
. Scikit-learn: pip install -U scikit-learn
Windows:
. Python: https://www.python.org/downloads/
. Pip 22.3: já incluso no python 3.11
. Scikit-learn: https://scikit-learn.org/dev/install.html
Obs.: instale nova dependência, caso seja solicitado 
"""
