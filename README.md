程式檔案內有部分註解，若有不清楚的部分再問


image_capture用於建立資料庫   用對比的方式擷取出物件，可以用素色的布或牆壁當作背景


取出來的28x28彩色圖片 資料處理(ex:灰階、二值化、降維、tag)後可以丟入你們之前的神經網路進行訓練


dataset_tag 直接將圖片檔名更改當作tag


classifier用於訓練及實際測試


裡面有些跟檔案路徑有關的部分需自行更改


---------------------------------------------------------------------------------------------


操作說明：


將資料放在同一個資料夾並創建新的cap及training_data資料夾


![t](https://user-images.githubusercontent.com/72076184/96996406-63c89480-1562-11eb-9e0d-d1d3a492ee96.png)


1.先將一種物品放在拍攝區，用image_capture抓取影像，若是無法抓到可以將內部的canny參數(ex:30,50)、area閥值調低


![hi](https://user-images.githubusercontent.com/72076184/96996986-75f70280-1563-11eb-958e-838cc32cbc7a.png)


注意，標記前請將cap內抓取不正確的影像移到別的資料夾或刪除(可以將其用作訓練資料的最後一個類別)


2.用dataset_tag將其標註，再將tag的sort加1，並刪除標記前的資料(在cap資料夾中)(有多種類別的物品就重複拍攝並標記)


![註解 2020-10-23 185555](https://user-images.githubusercontent.com/72076184/96996024-bce3f880-1561-11eb-9337-c31e4574f084.png)


(也可以用手動標記及擷取)


3.用training的程式訓練並測試(內部的網路架構及參數(學習率、bias等)可自行調整)


同樣，若抓不到就將canny參數、area閥值調低
