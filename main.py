import pandas as pd
import numpy as np
import re
from collections import Counter

# Загрузка данных
df = pd.read_csv('train.csv')


# 1. Количество мужчин и женщин
male_count = (df['Sex'] == 'male').sum()
female_count = (df['Sex'] == 'female').sum()
print("1. Количество мужчин и женщин на параходе:")
print(f"   Мужчин: {male_count}, Женщин: {female_count}")


# 2. Количество пассажиров, загрузившихся в различных портах
embarked_counts = df['Embarked'].value_counts()
S_count = embarked_counts.get('S', 0)
C_count = embarked_counts.get('C', 0)
Q_count = embarked_counts.get('Q', 0)
print("2. Количество пассажиров, загрузившихся в различных портах:")
print(f"   Порт S (Саутгемптон): {S_count}")
print(f"   Порт C (Шербур): {C_count}")
print(f"   Порт Q (Квинстаун): {Q_count}")


# 3. Доля погибших
total_passengers = len(df)
died = (df['Survived'] == 0).sum()
died_percent = died / total_passengers * 100
print("3. Доля погибших на параходе:")
print(f"   Погибло: {died} человек из {total_passengers}")
print(f"   Процент погибших: {died_percent:.1f}%")


# 4. Доли пассажиров по классам (Pclass)
class_counts = df['Pclass'].value_counts().sort_index()  # ключи 1, 2, 3
class_shares = class_counts / total_passengers * 100
print("4. Доли пассажиров по классам:")
print(f"   Первый класс: {class_shares[1]:.1f}%")
print(f"   Второй класс: {class_shares[2]:.1f}%")
print(f"   Третий класс: {class_shares[3]:.1f}%")


# 5. Коэффициент корреляции Пирсона между SibSp и Parch
corr_sibsp_parch = df['SibSp'].corr(df['Parch'])
print("5. Коэффициент корреляции Пирсона между количеством супругов и детей:")
print(f"   Корреляция между SibSp и Parch: {corr_sibsp_parch:.3f}\n")

# 6. Корреляция между:
#    - возрастом (Age) и Survived
#    - полом (Sex) и Survived (преобразуем: male=1, female=0)
#    - классом (Pclass) и Survived
df['Sex_binary'] = df['Sex'].map({'male': 1, 'female': 0})
# Для возраста учитываем только непустые значения
age_survived_corr = df.dropna(subset=['Age'])['Age'].corr(df.dropna(subset=['Age'])['Survived'])
sex_survived_corr = df['Sex_binary'].corr(df['Survived'])
pclass_survived_corr = df['Pclass'].corr(df['Survived'])
print("6. Корреляция между разными параметрами и выживаемостью (Survived):")
print(f"   Возраст и выживаемость: {age_survived_corr:.2f}")
print(f"   Пол и выживаемость: {sex_survived_corr:.2f}")
print(f"   Класс и выживаемость: {pclass_survived_corr:.2f}\n")

# 7. Статистика по возрасту: среднее, медиана, минимум, максимум
age_mean = df['Age'].mean()
age_median = df['Age'].median()
age_min = df['Age'].min()
age_max = df['Age'].max()
print("7. Статистика по возрасту пассажиров:")
print(f"   Средний возраст: {age_mean:.1f} лет")
print(f"   Медианный возраст: {age_median:.0f} лет")
print(f"   Минимальный возраст: {age_min:.1f} лет")
print(f"   Максимальный возраст: {age_max:.0f} лет\n")

# 8. Статистика по цене билета (Fare): среднее, медиана, минимум, максимум
fare_mean = df['Fare'].mean()
fare_median = df['Fare'].median()
fare_min = df['Fare'].min()
fare_max = df['Fare'].max()
print("8. Статистика по цене билета:")
print(f"   Средняя цена: {fare_mean:.2f}")
print(f"   Медианная цена: {fare_median:.2f}")
print(f"   Минимальная цена: {fare_min:.2f}")
print(f"   Максимальная цена: {fare_max:.2f}\n")

# 9. Самое популярное мужское имя на корабле
male_names = df[df['Sex'] == 'male']['Name']
male_first_names = []
for name in male_names:
    # Ищем шаблон: "Mr. <Имя>"
    match = re.search(r'Mr\. ([A-Za-z]+)', name)
    if match:
        male_first_names.append(match.group(1))
male_name_counts = Counter(male_first_names)
most_common_male = male_name_counts.most_common(1)[0] if male_first_names else None
print("9. Самое популярное мужское имя на корабле:")
if most_common_male:
    print(f"   Имя: {most_common_male[0]}, Количество: {most_common_male[1]} человек\n")
else:
    print("   Данные отсутствуют\n")

# 10. Самые популярные имена среди пассажиров старше 15 лет
df_over15 = df[df['Age'] > 15]

# Для мужчин
male_names_over15 = df_over15[df_over15['Sex'] == 'male']['Name']
male_first_names_over15 = []
for name in male_names_over15:
    match = re.search(r'Mr\. ([A-Za-z]+)', name)
    if match:
        male_first_names_over15.append(match.group(1))
male_name_counts_over15 = Counter(male_first_names_over15)
most_common_male_over15 = male_name_counts_over15.most_common(1)[0] if male_first_names_over15 else None

# Для женщин
female_names_over15 = df_over15[df_over15['Sex'] == 'female']['Name']
female_first_names_over15 = []
for name in female_names_over15:
    # Для женщин часто встречается формат "Miss." или "Mrs.", иногда имя в скобках
    match = re.search(r'Miss\. ([A-Za-z]+)', name)
    if not match:
        # Попытка извлечь имя из скобок для "Mrs."
        match = re.search(r'Mrs\. [A-Za-z]+\s*\(([^)]+)\)', name)
        if match:
            # Берем первое слово внутри скобок
            candidate = match.group(1).split()[0]
            match = re.search(r'([A-Za-z]+)', candidate)
    if match:
        female_first_names_over15.append(match.group(1))
female_name_counts_over15 = Counter(female_first_names_over15)
most_common_female_over15 = female_name_counts_over15.most_common(1)[0] if female_first_names_over15 else None

print("10. Самые популярные имена среди людей старше 15 лет:")
if most_common_male_over15:
    print(f"   Мужское: {most_common_male_over15[0]}, Количество: {most_common_male_over15[1]} человек")
else:
    print("   Мужское: данные отсутствуют")
    
if most_common_female_over15:
    print(f"   Женское: {most_common_female_over15[0]}, Количество: {most_common_female_over15[1]} человек")
else:
    print("   Женское: данные отсутствуют")

