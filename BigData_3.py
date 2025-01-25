from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder \
    .appName("Dataset Analysis") \
    .master("local[*]") \
    .getOrCreate()

file_path = "NCHS_-_Leading_Causes_of_Death__United_States.csv"
data = spark.read.csv(file_path, header=True, inferSchema=True)

data.printSchema()

# Упражнение 1: Подсчет числа записей по каждому штату
data.groupBy("State").count().orderBy("count", ascending=False).show()

# Упражнение 2: Среднее значение количества смертей для каждого года
avg_deaths_per_year = data.groupBy("Year").avg("Deaths").orderBy("Year")
avg_deaths_per_year.show()

# Визуализация 1: Среднее количество смертей по годам
pandas_df = avg_deaths_per_year.toPandas()
plt.figure(figsize=(10, 6))
sns.barplot(x="Year", y="avg(Deaths)", data=pandas_df, palette="viridis")
plt.title("Среднее количество смертей по годам")
plt.xlabel("Год")
plt.ylabel("Среднее количество смертей")
plt.show()

# Упражнение 3: Топ-5 причин смертности в США
top_causes = data.filter(col("State") == "United States") \
    .groupBy("Cause Name") \
    .sum("Deaths") \
    .orderBy(col("sum(Deaths)").desc()) \
    .limit(5)
top_causes.show()

# Визуализация 2: Топ-5 причин смертности
pandas_df = top_causes.toPandas()
plt.figure(figsize=(10, 6))
sns.barplot(x="sum(Deaths)", y="Cause Name", data=pandas_df, palette="plasma")
plt.title("Топ-5 причин смертности в США")
plt.xlabel("Количество смертей")
plt.ylabel("Причина смерти")
plt.show()

# Упражнение 4: Сравнение уровня смертности между штатами
state_death_rate = data.groupBy("State").avg("Age-adjusted Death Rate").orderBy("avg(Age-adjusted Death Rate)", ascending=False)
state_death_rate.show()

# Упражнение 5: Динамика смертности в конкретном штате (например, Калифорния)
california_data = data.filter(col("State") == "California").groupBy("Year").sum("Deaths").orderBy("Year")
california_data.show()

# Визуализация 3: Динамика смертности в Калифорнии
pandas_df = california_data.toPandas()
plt.figure(figsize=(10, 6))
sns.lineplot(x="Year", y="sum(Deaths)", data=pandas_df, marker="o")
plt.title("Динамика смертности в Калифорнии")
plt.xlabel("Год")
plt.ylabel("Количество смертей")
plt.grid(True)
plt.show()

spark.stop()
