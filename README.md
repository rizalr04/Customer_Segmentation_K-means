# Customer_Segmentation_K-means
Customer Segmentation adalah proses membagi pelanggan atau calon pelanggan suatu bisnis menjadi kelompok-kelompok yang memiliki karakteristik atau perilaku serupa. Tujuan utama dari customer segmentation adalah memahami lebih baik kebutuhan, preferensi, dan perilaku pelanggan agar perusahaan dapat menyusun strategi pemasaran yang lebih efektif dan merancang pengalaman pelanggan yang lebih personal. 

Proyek ini bertujuan untuk melakukan pengelompokan pelanggan pada dataset mall customer menggunakan algoritma K-Means Clustering. Dataset ini berisi informasi tentang pelanggan seperti usia, jenis kelamin, pendapatan tahunan, dan skor pengeluaran. Tujuan dari proyek ini adalah mengelompokkan pelanggan berdasarkan perilaku pembelian dan karakteristik demografis mereka.

## Project Outline
1. Dataset
2. Data Preprocessing
3. Model Building
4. Conclusion

## 1. Dataset
Ini adalah data mall customer yang memiliki 5 kolom dan 200 baris.

### Loading Dataset
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```
```python
from google.colab import drive
drive.mount("/content/drive/")
```
```python
df = pd.read_csv('/content/drive/MyDrive/Data Portofolio/Mall_Customers.csv')
df.head()
```
   ![image](https://github.com/rizalr04/Customer_Segmentation_K-means/blob/da7a5a1ec71333e49d46634738b8b004df6924d2/Asset/data.PNG)

## 2. Data Preprocessing
- Mengecek informasi dalam data
```python
df.info()
```
   ![image](https://github.com/rizalr04/Customer_Segmentation_K-means/blob/da7a5a1ec71333e49d46634738b8b004df6924d2/Asset/df%20info.PNG)
- Mengecek nilai kosong pada data
```python
df.isnull().sum()
```
   ![image](https://github.com/rizalr04/Customer_Segmentation_K-means/blob/da7a5a1ec71333e49d46634738b8b004df6924d2/Asset/df%20isnull.PNG)
- Statistik Deskriptif
```python
df.describe()
```
   ![image](https://github.com/rizalr04/Customer_Segmentation_K-means/blob/da7a5a1ec71333e49d46634738b8b004df6924d2/Asset/df%20describe.PNG)
- Memilih kolom Annual Income (k$) dan Spending Score (1-100) untuk proses klastering bertujuan untuk fokus pada dua atribut kunci yang dianggap paling relevan dalam menganalisis dan membentuk kelompok pelanggan.
```python
X = df[['Annual Income (k$)','Spending Score (1-100)']]
```
   ![image](https://github.com/rizalr04/Customer_Segmentation_K-means/blob/da7a5a1ec71333e49d46634738b8b004df6924d2/Asset/x.PNG)
- Annual Income vs Spending Score
```python
plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'])
plt.title('Annual Income (k$) vs Spending Score (1-100)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```
   ![image](https://github.com/rizalr04/Customer_Segmentation_K-means/blob/da7a5a1ec71333e49d46634738b8b004df6924d2/Asset/bivariate.PNG)

## 3. Model Building
### KMeans Clustering
```python
from sklearn.cluster import KMeans
```
```python
k_means = KMeans()
k_means.fit(X)
```

#### - Elbow Method

Elbow Method adalah suatu metode yang digunakan untuk menentukan jumlah optimal dari klaster (clusters) dalam algoritma K-Means Clustering. 
```python
wcss=[]
for i in range(1,11):
  k_means = KMeans(n_clusters=i)
  k_means.fit(X)
  wcss.append(k_means.inertia_)
```
```python
plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()
```
   ![image](https://github.com/rizalr04/Customer_Segmentation_K-means/blob/da7a5a1ec71333e49d46634738b8b004df6924d2/Asset/elbow.PNG)
dari grafik diatas, dapat dilihat bahwa pada nilai 5 terdapat tikungan, dan itu dapat dianggap sebagai nilai k yang merupakan jumlah klaster. Oleh karena itu, k=5, artinya jumlah klaster adalah 5.

#### - Training Model

Berdasarkan hasil dari elbow method, akan dibangun model K-Means dengan jumlah cluster 5. Setelah itu, melakukan prediksi terhadap data dan menyimpannya dalam variabel y. Kemudian, kemudian data hasil prediksi pada variabel y akan ditambahkan ke dalam dataframe sebagai kolom baru bernama ‘Cluster’
```python
k_means = KMeans(n_clusters=5, random_state=42)
y = k_means.fit_predict(X)
```
```python
df['Cluster']=y
df.head()
```
   ![image](https://github.com/rizalr04/Customer_Segmentation_K-means/blob/da7a5a1ec71333e49d46634738b8b004df6924d2/Asset/kmeans.PNG)

Selanjutnya akan dibuatkan visualisasi berdasarkan hasil semua cluster
```python
plt.scatter(X.iloc[y==0,0],X.iloc[y==0,1],s=100,c='red',label="Cluster 1")
plt.scatter(X.iloc[y==1,0],X.iloc[y==1,1],s=100,c='yellow',label="Cluster 2")
plt.scatter(X.iloc[y==2,0],X.iloc[y==2,1],s=100,c='green',label="Cluster 3")
plt.scatter(X.iloc[y==3,0],X.iloc[y==3,1],s=100,c='blue',label="Cluster 4")
plt.scatter(X.iloc[y==4,0],X.iloc[y==4,1],s=100,c='black',label="Cluster 5")
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=100,c="magenta", label="Centorid")
plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
```
   ![image](https://github.com/rizalr04/Customer_Segmentation_K-means/blob/da7a5a1ec71333e49d46634738b8b004df6924d2/Asset/kmeans%20visual.PNG)

## 4. Conclusion
1. **Cluster 1 (Merah)** : Kelompok Pelanggan yang berpenghasilan rata-rata dengan skor pengeluaran rata-rata. ini merupakan kelompok pelanggan yang berhati-hati dengan pengeluaran mereka untuk belanja maka dari itu perlu untuk Identifikasi kategori produk yang paling relevan untuk pelanggan ini dan buat penawaran khusus atau bundel produk yang menarik bagi mereka.
2. **Cluster 2 (Kuning)** : Kelompok Pelanggan yang berpenghasilan lebih tinggi namun mereka tidak menghabiskan lebih banyak uang untuk belanja atau skor pengeluaran rendah. Untuk kelompok ini perlu diberi tawaran pelayanan konsultatif atau personal shopper untuk membantu pelanggan membuat keputusan yang lebih bijak tentang pembelian, mendorong mereka untuk mempertimbangkan produk atau layanan yang sesuai dengan kebutuhan mereka.
3. **Cluster 3 (Hijau)** :  Kelompok Pelanggan yang berpenghasilan rendah dengan skor pengeluaran rendah. Pada kelompok ini perlu diberi pertimbangan untuk memberikan opsi pembayaran yang fleksibel atau cicilan agar lebih sesuai dengan kemampuan finansial mereka..
4. **Cluster 4 (Biru)** : Kelompok Pelanggan berpenghasilan rendah dengan skor pengeluaran tinggi. pada kelompok ini berikan penawaran diskon dan paket bundel yang menggabungkan beberapa produk atau layanan dengan harga yang lebih terjangkau daripada membeli masing-masing.
5. **Cluster 5 (Hitam)** : Kelompok pelanggan yang berpenghasilan tinggi dengan skor pengeluaran tinggi. Diskon dan penawaran lain yang ditargetkan pada kelompok ini akan meningkatkan skor pembelanjaan mereka dan memaksimalkan keuntungan, serta Implementasikan program keanggotaan premium dengan manfaat tambahan seperti diskon eksklusif, pengiriman gratis, dan akses eksklusif ke acara atau lainnya.
