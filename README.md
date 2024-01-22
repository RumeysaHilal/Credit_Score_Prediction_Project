# Credit Score Prediction Project

| Alan                    | Açıklama                                               |
|---------------------------|--------------------------------------------------------|
| ID                        | Kaydın benzersiz kimliği                              |
| Customer_ID               | Müşterinin benzersiz kimliği                          |
| Month                     | Yılın ayı                                             |
| Name                      | Kişinin adı                                           |
| Age                       | Kişinin yaşı                                          |
| SSN                       | Kişinin Sosyal Güvenlik Numarası                      |
| Occupation                | Kişinin mesleği                                        |
| Annual_Income             | Kişinin yıllık geliri                                 |
| Monthly_Inhand_Salary     | Kişinin aylık eline geçen maaşı                       |
| Num_Bank_Accounts         | Kişinin sahip olduğu banka hesaplarının sayısı        |
| Num_Credit_Card           | Kişinin sahip olduğu kredi kartı sayısı               |
| Interest_Rate             | Kişinin kredi kartındaki faiz oranı                  |
| Num_of_Loan               | Kişinin bankadan aldığı kredi sayısı                  |
| Type_of_Loan              | Kişinin bankadan aldığı kredi türleri                 |
| Delay_from_due_date       | Kişinin ödeme tarihinden ortalama kaç gün geciktiği  |
| Num_of_Delayed_Payment    | Kişinin yaptığı gecikmiş ödeme sayısı                 |
| Changed_Credit_Card       | Kişinin kredi kartı limitindeki değişim yüzdesi   |
| Num_Credit_Inquiries      | Kişinin kredi kartı sorgularının sayısı              |
| Credit_Mix                | Müşterinin Kredi Karışımı Sınıflandırması             |
| Outstanding_Debt          | Kişinin borç durumu                                   |
| Credit_Utilization_Ratio  | Müşterinin kredi kartı kullanım oranı                |
| Credit_History_Age        | Kişinin kredi geçmişinin yaşı                        |
| Payment_of_Min_Amount     | Eğer kişi sadece asgari ödeme miktarını ödediyse "Evet", aksi takdirde "Hayır" |
| Total_EMI_per_month        | Kişinin aylık toplam EMI (Taksit) miktarı            |
| Amount_invested_monthly   | Kişinin aylık yatırım miktarı                         |
| Payment_Behaviour         | Kişinin ödeme davranışı                              |
| Monthly_Balance            | Kişinin hesabında kalan aylık bakiye                |
| Credit_Score              | Kişinin kredi skoru                                  |



Kredi skor sınıf tahminin yapıldığı bu projede önişleme, görselleştirmeler ve deneme yapılan modeller ayrı ayrı fonksiyonlaştırılarak python dosyaları içerisinde parametrelerle çağrılabilir şekilde hazırlandı. Projeyi kendi ortamınızda çalıştırmanız için öncelikle indirdiğiniz klasörde requirements.txt dosyası içerisinde belirtilen kütüphanelrin indirilmesi gerekir.

```bash
pip install -r requirements.txt
```

İndirme sonrası project_all.ipynb dosyasını hücre hücre çalıştırarak ilerlemeleri gerçekleştirebilirsiniz. Görselleştirme için gerekli fonksiyonlar data_visualization.py dosyası içerisinde, modeller için gerekli bilgiler model_func.py dosyası içerisinde, ilk veri setinin düzenlemesi credit_func.py içerisinde yer almaktadır. credit-preprocess.ipynb içerisinde bazı ön işlemler ve denemeler yer almaktadır incelemk isteyenler için bırakılmıştır, dirty koddur. 

Kredi skorunu etkileyen birçok etken maddenin yer aldığı bu büyük veri setinden yeni özellikler oluşturuldu, bazı özellikler direkt olarak yeni veri seti için hazır hale getirildi ve ataması yapıldı. Aşağıda ilk veri setinin işleme alınma işlem yapılması için çalıştırılması gereken kod parçası yer almakta. Çağrım sonrası fonksiyon içerisinde new_cs.csv isminde tanımlanan csv dosyasının oluşturulması sağlanır.

```python
import credit_func

url = "credit-score.csv"
credit_data = pd.read_csv(url)
credit_func.create_dataFrame(credit_data, csv=True)
```
### Keşifsel Veri Analizi

Keşifsel veri analizi aşamasında ana veri seti içerisinde eksik bir veri kontrolüyle birlikte çok aykırı veriler olmadığı için outlier temizliği yapılmadı. Credi_Score alanı object türünde olduğu için LabelEncoder tekniğiyle sınıfların sayısal hale getirilmesi, atanması yapıldı. 

Görselleştirmeler kısmında ilk olarak görselleştirme denemeleri yapıldı. Anlamlı sonuç çıkaran plotlar fonksiyonlara çevrilerek başka bir python dosyası içerisine yerleştirildi (data_visualization.py). Bu şekilde daha dinamik bir yapı hedeflendi. 

Aşağıdaki örnekte dağılım yüzdesinin plot haline getirilmesinin fonksiyonunun çağrılmış örnek hali yer almakta. Çıktısı da altında gösterilmektedir.

```python
import data_visualization

data_visualization.plot_percentage_distribution(credit_data, "Interest_Rate")
```
![distribution_of_paymentB](https://github.com/RumeysaHilal/Credit_Score_Prediction_Project/assets/66912242/a1b01076-7537-4140-b231-885c81842edd)

Yeni belirlenen özellikler aylık ödenmesi gereken taksit miktarının ne kadar ödendiği, kart başına düşen borç miktarı, kredi kartı kullanım sıklığı, günlük gecikme oranı, yüksek kredi kullanımı özellikleri oluşturuldu. Heatmap üzerinden özellikler seçilerek işlenmeden direkt olarak yeni veri setine eklendi. 

Hedef sınıf değerlerinin dengesiz olduğunu belirlendi ve bu dengesizliği gidermek için SMOTE(Synthetic Minority Over-sampling Technique) işlemi uygulandı. Azınlık sınıfındaki veri sayısını artırmak için sentetik örnekler oluşturmak için kullanılır. Aşağıda işlem uygulanmadan önceki verilerin durumu ve işlem uygulandıktan sonraki durumun pie grafiğindeki halleri verilmiştir.

![Merged_document](https://github.com/RumeysaHilal/Credit_Score_Prediction_Project/assets/66912242/12e4b726-9709-4093-abeb-a44d009115cb)

Modelleri kurgulayarak denemeler yaparak yüksek accuracy değerine sahip modeli ana model olma seçme yolu izlendi. Birden çok parametre ile denenebilir modellere farklı yaklaşımlar ile hangi parametrenin nasıl uygun olacağına karar verildi. Örnek olarak aşağıda KNN algoritmasına cross-validation ve test accuracy değerlerinin k değerlerine göre değişimlerini gösteren grafik verilmiştir. Ana modelde k değeri 4 olarak seçildi, en yüksek oran 2 değerinde olmasına rağmen. Bunun sebebi düşük değerlerdeki k değerleri overfit durumuna yakın olabilme ihtimalidir.

<<<<<<< HEAD
![KNN Comparasion](resimler/accuracy-comparison.png)

Çalışma boyunca veri daha iyi şekilde anlaşılmaya ve en uygun model seçilmeye çalışılmıştır. En yüksek doğruluk knn modelinde olduğu için ana model olarak daha sonrasında knn algoritması kullanılabilir. Projenin devamında müşteri etkileşimine eklenebilir şekilde FestAPI benzeri hazır araçlarla sunulabilir. Meslek grupları da içerisinde bulunduğu için meslek grupları üzerinden naaliz yapılır meslek gruplarındaki kişilerin alışkanlıkları hakkında yorumlar yapılabilir. Bu yaş grubu için de kullanılabilir bir değerlendirme aracı olabilir. 
=======
![accuracy-comparison](https://github.com/RumeysaHilal/Credit_Score_Prediction_Project/assets/66912242/ab7dce39-5d6b-48cc-9f0c-e9c64d3e388f)
>>>>>>> 9e58a165d4a779d5639536e1d1dfe3bb9f043b6c
