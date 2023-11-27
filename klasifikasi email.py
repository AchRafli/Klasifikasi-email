import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


data = {
    'email': ['Besok tanggal 27 November 2023 Praktikum Jaringan Komputer diliburkan.Untuk Praktikum 4 akan dirangkap minggu depannya.',
              'Selamat anda mendapatkan Uang tunai sebesar 500 juta dari undian berhadiah',
              'Batas waktu besok: Tugas 2 dan 3 - Praktikum Jaringan Komputer',
              'Menangkan hadiah besar dengan mengklik link dibawah ini!',
              'Materi baru: Modul Praktikum 2 Jaringan Komputer ',
              'Tugas baru: Laporan 3 dan Pemanasan 4'],
    'label': [0, 1, 0, 1, 0, 0]  # 0: Normal, 1: Spam
}


emails = np.array(data['email'])
labels = np.array(data['label'])


X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vectorized, y_train)


y_pred = model.predict(X_test_vectorized)



new_email = ['Ini adalah email dari achrafli098@gmail.com']
new_email_vectorized = vectorizer.transform(new_email)
prediction = model.predict(new_email_vectorized)

if prediction[0] == 0:
    print('Email ini tidak masuk kategori spam.')
else:
    print('Email ini masuk kategori spam.')
