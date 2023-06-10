# Neural_network_SBU_Predicting_price_LSTM
In this project the objective was to learn RNN.
Data is extracted from a CSV file provided by our professor. It's price market of a single name from 2006-01-03 until 2018-01-01
<br>
Data is given such that the model receives 60 days prior price and then model will predict the day 61 result.
The model was developed by experimenting different settings such as comparing two models that have different number of units.
Then by comparing MSE one model was choosed.
<p align="center">
  <img width="460" height="300" src="https://github.com/smrh1379/Neural_network_SBU/blob/Predicting_price_using_RNN_LSTM/val1.jpg">
</p>
<p align="center">
  <img width="460" height="300" src="https://github.com/smrh1379/Neural_network_SBU/blob/Predicting_price_using_RNN_LSTM/predict1.jpg">
</p>
The above image shows that the model was able to closely predict the price.
After that in the choosen model was compared with a model that had one more layer of LSTM.
The latter model showed less val_loss and MSE than the former Hence it was picked to continue the process.
<p align="center">
  <img width="460" height="300" src="https://github.com/smrh1379/Neural_network_SBU/blob/Predicting_price_using_RNN_LSTM/val2.jpg">
</p>
Then 2 copy of the architecture of the choosen model was trained on the dataset to see if changing batch size can affect the results or even make it better
<p align="center">
  <img width="460" height="300" src="https://github.com/smrh1379/Neural_network_SBU/blob/Predicting_price_using_RNN_LSTM/val3.jpg">
</p>
model 4 was trained with batch size 4 and the other model with batch size 2
<br>
model 4 results outcompete the others, making us to choose it as the final model.
<br>
<p align="center">
  <img width="460" height="300" src="https://github.com/smrh1379/Neural_network_SBU/blob/Predicting_price_using_RNN_LSTM/predict2.jpg">
</p>
The above picture compare model 4 prediction of the prices and actual prices. It shows that model 4 has the highest accuracy and lowest MSE among all of the models
<p align="center">
  <img width="460" height="300" src="https://github.com/smrh1379/Neural_network_SBU/blob/Predicting_price_using_RNN_LSTM/mse_final.jpg">
</p>
