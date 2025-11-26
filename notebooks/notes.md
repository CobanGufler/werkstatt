##### TimesFM

YT: 	https://youtu.be/7QqUw29bJmc?si=yjIx2Zm90zLehIto  
https://youtu.be/265Mpaj8O1U?si=b4lZTLMO-qRR7GVO (bis Hälfte, dann multivariat)

Hugging Face: 	https://huggingface.co/google/timesfm-2.5-200m-pytorch


- Wandelt die Zeitreihe zuerst in Patches (Chunks) um → ähnlich wie Vision Transformers
- Nutzt einen Decoder-Only Transformer (ähnlich GPT-Architektur)
- Modell lernt: „Gib mir Kontext-Patches → ich generiere zukünftige Patches.“
- Optimiert für univariate Zero-Shot Forecasting
- Kein explizites Handling von Kovariaten oder Multivariate Dynamik
- Forecasten erfolgt autoregressiv: Output-Patch wird als Input für den nächsten Schritt genutzt



##### 

##### Chronos-Bolt
- Werte werden erst quantisiert und in Diskrete Token umgewandelt (ähnlich NLP Tokenizer)
- Transformer verarbeitet Sequenzen aus Tokens → nicht direkte Zahlenwerte
- Modell sagt Wahrscheinlichkeitsverteilungen von Token voraus, nicht konkrete Werte
- Unterstützt probabilistische Vorhersagen (Quantile, full distribution)
- Stärker auf Effizienz getrimmt (Bolt-Variante = optimierter und schneller)
- Funktioniert gut für Zero-Shot, Few-Shot und Multi-Step Forecasts

##### 

##### Moirai
- Modell nimmt Zeitreihen als kontinuierliche numerische Sequenzen, kein Tokenizer
- Transformer-Encoder (kein autoregressiver Decoder wie bei TimesFM)
- Unterstützt beliebige Feature-Strukturen: multivariate, verschieden Frequenzen, fehlende Werte etc.
- Designed für Multivariate Forecasting und Exogene Variablen
- Zielt auf universelle Forecast-Generalisation über Domains hinweg
- Moirai ist eher ein Representation Learning Modell, nicht rein autoregressiv

Also: 

TimesFM = GPT-ähnlicher Patch-Decoder,

Chronos-Bolt = Token-Forecasting wie ein Sprachmodell,

Moirai = universelles Encoder-Foundation-Model für multivariate Strukturen



andere Modelle: Chronos-Bolt oder Chronos-Classic, UniTS, PatchTST-ZS

#####
##### Metriken

Library *statsforecast* (hat z.B. MAPE, sMAPE, MASE...)

oder

github.com/SalesforceAIResearch/uni2ts

uni2ts, ist state-of-the-art
Offizielles Zuhause: Es ist die offizielle Bibliothek von Salesforce Research für ihre Moirai-Modelle (Moirai 1.0, 1.1 und das ganz neue 2.0).
Unified Evaluation: PLUSPUNKT, Salesforce hat Code veröffentlicht, um TimesFM und Chronos einheitlich zu evaluieren.
hat MASE und sMAPE

