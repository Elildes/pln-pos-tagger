# PLN - Processamento de linguagem natural: Modelagem de Part-of-speech tagger  

## Dependências  

Antes de rodar esta aplicação, instale as dependências abaixo:  

**Lista de dependências:**  

Linux:  
1. Python: sudo apt-get install python3  
2. Pip3: sudo apt-get install python3-pip  
3. Scikit-learn: pip install -U scikit-learn   

Windows:  
1. Python: https://www.python.org/downloads/  
2. Pip 23.2.1: já incluso no python 3.12  
3. Scikit-learn: https://scikit-learn.org/dev/install.html   

**Obs.**: instale nova dependência, caso seja solicitado.  

## Rodar o programa  

Abrir o terminal e executar o seguinte comando:  

1. `python3 unigram.py`  
2. Aguarde, que vai demorar um pouco  
3. Resultado esperado:  

```
1ª PARTE ok!  

Lista tag_real tamanho: 131768  

Lista tag_pred tamanho: 131768  

Matriz de confusão:  
 [[11003    10     0 ...     1     0     0]  
 [    1  9376     1 ...     0     0     0]  
 [    0     5  3867 ...     0     0     0]  
 ...
 [    0     0     0 ...     5     0     0]  
 [    0     1     0 ...     0     4     0]  
 [    0     0     0 ...     0     0     0]]  

2ª PARTE ok!  

3ª PARTE ok!  


OK, fim código!!!  
```



