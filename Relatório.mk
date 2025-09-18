Após o treinamento e a avaliação dos modelos de regressão linear, Ridge e Lasso no dataset California Housing Prices, obtivemos os seguintes resultados:

 Linear Regression: RMSE ≈ 67.110, R² ≈ 0.671, MAE ≈ 48.465, MAPE ≈ 28.13%
 Ridge Regression: RMSE ≈ 67.066, R² ≈ 0.671, MAE ≈ 48.440, MAPE ≈ 28.12%
 Lasso Regression: RMSE ≈ 67.110, R² ≈ 0.671, MAE ≈ 48.464, MAPE ≈ 28.13%

Esses valores indicam que o modelo explica aproximadamente 67% da variabilidade dos preços das casas. O RMSE (Erro Quadrático Médio) mostra que, em média, os erros de previsão ficam na casa de 67 mil dólares. Já o MAE (Erro Absoluto Médio) confirma que o desvio absoluto médio das previsões em relação ao valor real é de cerca de 48 mil dólares.

A métrica MAPE revela que o erro relativo médio é de aproximadamente 28%, ou seja, as previsões tendem a se afastar em média 28% do valor real. Para traduzir isso em uma métrica mais interpretável, definimos a Accuracy (%) ≈ 72%, que mostra que o modelo acerta com boa proximidade em torno de 7 em cada 10 previsões.

Entre os três algoritmos, o Ridge Regression apresentou o melhor desempenho, ainda que por uma diferença pequena em relação ao modelo linear simples. Isso era esperado, já que a regularização L2 (Ridge) é mais eficaz em casos onde existe correlação entre as variáveis preditoras — algo comum neste dataset, onde atributos como população, quartos e domicílios têm relação entre si.

O Lasso Regression obteve resultados muito próximos ao modelo linear clássico, o que sugere que, neste caso, a regularização L1 não trouxe ganhos expressivos. Ainda assim, o Lasso é importante por sua capacidade de reduzir coeficientes irrelevantes a zero, realizando uma espécie de seleção automática de variáveis.

Em resumo:

 O modelo linear já fornece uma boa aproximação.
 O Ridge é a melhor escolha prática neste cenário, pois melhora ligeiramente o ajuste e ajuda a evitar sobreajuste (overfitting).
 O Lasso, apesar de não melhorar as métricas, é útil em contextos com muitos atributos redundantes.



Assim, podemos concluir que a utilização de técnicas de regularização (Ridge e Lasso) é fundamental para garantir que os modelos sejam mais robustos e não fiquem excessivamente ajustados aos dados de treino. O modelo Ridge se mostrou o mais adequado para o problema proposto.

