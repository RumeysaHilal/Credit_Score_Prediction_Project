# Credit Score Prediction Project

| Field                     | Description                                           |
|---------------------------|-------------------------------------------------------|
| ID                        | Unique ID of the record                               |
| Customer_ID               | Unique ID of the customer                             |
| Month                     | Month of the year                                     |
| Name                      | The name of the person                                 |
| Age                       | The age of the person                                  |
| SSN                       | Social Security Number of the person                   |
| Occupation                | The occupation of the person                           |
| Annual_Income             | The Annual Income of the person                        |
| Monthly_Inhand_Salary     | Monthly in-hand salary of the person                   |
| Num_Bank_Accounts         | The number of bank accounts of the person              |
| Num_Credit_Card           | Number of credit cards the person is having            |
| Interest_Rate             | The interest rate on the credit card of the person     |
| Num_of_Loan               | The number of loans taken by the person from the bank  |
| Type_of_Loan              | The types of loans taken by the person from the bank   |
| Delay_from_due_date       | The average number of days delayed by the person from the date of payment |
| Num_of_Delayed_Payment    | Number of payments delayed by the person               |
| Changed_Credit_Card       | The percentage change in the credit card limit of the person |
| Num_Credit_Inquiries      | The number of credit card inquiries by the person      |
| Credit_Mix                | Classification of Credit Mix of the customer           |
| Outstanding_Debt          | The outstanding balance of the person                  |
| Credit_Utilization_Ratio  | The credit utilization ratio of the credit card of the customer |
| Credit_History_Age        | The age of the credit history of the person            |
| Payment_of_Min_Amount     | Yes if the person paid the minimum amount to be paid only, otherwise no. |
| Total_EMI_per_month        | The total EMI per month of the person                  |
| Amount_invested_monthly   | The monthly amount invested by the person              |
| Payment_Behaviour         | The payment behaviour of the person                    |
| Monthly_Balance            | The monthly balance left in the account of the person   |
| Credit_Score              | The credit score of the person                          |


Kredi skor sınıf tahminin yapıldığı bu projede önişleme, görselleştirmeler ve deneme yapılan modeller ayrı ayrı fonksiyonlaştırılarak python dosyaları içerisinde parametrelerle çağrılabilir şekilde hazırlandı. Kredi skorunu etkileyen birçok etken maddenin yer aldığı bu büyük veri setinden yeni özellikler oluşturuldu, bazı özellikler direkt olarak yeni veri seti için hazır hale getirildi ve ataması yapıldı. Aşağıda ilk veri setinin işleme alınma işlem yapılması için çalıştırılması gereken kod parçası yer almakta. Çağrım sonrası fonksiyon içerisinde new_cs.csv isminde tanımlanan csv dosyasının oluşturulması sağlanır.

```python
import credit_func

url = "credit-score.csv"
credit_data = pd.read_csv(url)
credit_func.create_dataFrame(credit_data, csv=True)
```
### Keşifsel Veri Analizi

Keşifsel veri analizi aşamasında ana veri seti içerisinde eksik bir veri kontrolüyle birlikte çok aykırı veriler olmadığı için outlier temizliği yapılmadı. Credi_Score alanı object türünde olduğu için LabelEncoder tekniğiyle sınıfların sayısal hale getirilmesi, atanması yapıldı. 

Görselleştirmeler kısmında ilk olarak görselleştirme denemeleri yapıldı. Anlamlı sonuç çıkaran plotlar fonksiyonlara çevrilerek başka bir python dosyası içerisine yerleştirildi (data_visualization.py). Bu şekilde daha dinamik bir yapı oluştrulması hedeflendi. 

Aşağıdaki örnekte dağılım yüzdesinin plot haline getirilmesinin fonksiyonunun çağrılmış örnek hali yer almakta. Çıktısı da altında gösterilmektedir.

```python
import data_visualization

data_visualization.plot_percentage_distribution(credit_data, "Payment_Behaviour")
```
![Payment Behaviour Distribution](resimler/distribution_of_paymentB.png)


Yeni belirlenen özellikler aylık ödenmesi gereken taksit miktarının ne kadar ödendiği, kart başına düşen borç miktarı, kredi kartı kullanım sıklığı, günlük gecikme oranı, yüksek kredi kullanımı özellikleri oluşturuldu. Heatmap üzerinden özellikler seçilerek işlenmeden direkt olarak yeni veri setine eklendi. 

Hedef sınıf değerlerinin dengesiz olduğunu belirlendi ve bu dengesizliği gidermek için SMOTE(Synthetic Minority Over-sampling Technique) işlemi uygulandı. Azınlık sınıfındaki veri sayısını artırmak için sentetik örnekler oluşturmak için kullanılır. Aşağıda işlem uygulanmadan önceki verilerin durumu ve işlem uygulandıktan sonraki durumun pie grafiğindeki halleri verilmiştir.

![Before Resample](resimler/before_resample.png) ![Before Resample](resimler/resample_data.png) 

Modelleri kurgulayarak denemeler yaparak yüksek accuracy değerine sahip modeli ana model olma seçme yolu izlendi. Birden çok parametre ile denenebilir modellere farklı yaklaşımlar ile hangi parametrenin nasıl uygun olacağına karar verildi. Örnek olarak aşağıda KNN algoritmasına cross-validation ve test accuracy değerlerinin k değerlerine göre değişimlerini gösteren grafik verilmiştir. Ana modelde k değeri 4 olarak seçildi, en yüksek oran 2 değerinde olmasına rağmen. Bunun sebebi düşük değerlerdeki k değerleri overfit durumuna yakın olabilme ihtimalidir.

![KNN Comparasion](resimler/accuracy-comparison.png)