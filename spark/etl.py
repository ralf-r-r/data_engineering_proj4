import configparser
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek
from pyspark.sql.types import StringType, IntegerType, DoubleType, LongType, TimestampType, StructType, StructField

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config["KEYS"]['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config["KEYS"]['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    sc = spark.sparkContext
    sc._jsc.hadoopConfiguration().set("mapreduce.fileoutputcommitter.algorithm.version", "2")
    return spark


def create_song_schema():
    """
    creates the schema for the song data
    :return final_struct: pyspark.sql.types.StructType
    """
    data_schema = [StructField('num_songs', IntegerType(), True),
                   StructField('artist_id', StringType(), True),
                   StructField('artist_latitude', DoubleType(), True),
                   StructField('artist_longitude', StringType(), True),
                   StructField('artist_location', StringType(), True),
                   StructField('artist_name', StringType(), True),
                   StructField('song_id', StringType(), True),
                   StructField('title', StringType(), True),
                   StructField('duration', DoubleType(), True),
                   StructField('year', IntegerType(), True)
                   ]

    final_struct = StructType(fields=data_schema)
    return final_struct


def create_log_schema():
    """
    creates the schema for the log data
    :return final_struct: pyspark.sql.types.StructType
    """
    data_schema = [StructField('artist', StringType(), True),
                   StructField('auth', StringType(), True),
                   StructField('firstName', StringType(), True),
                   StructField('gender', StringType(), True),
                   StructField('itemInSession', IntegerType(), True),
                   StructField('lastName', StringType(), True),
                   StructField('length', DoubleType(), True),
                   StructField('level', StringType(), True),
                   StructField('location', StringType(), True),
                   StructField('method', StringType(), True),
                   StructField('page', StringType(), True),
                   StructField('registration', DoubleType(), True),
                   StructField('sessionId', LongType(), True),
                   StructField('song', StringType(), True),
                   StructField('status', LongType(), True),
                   StructField('ts', LongType(), True),
                   StructField('userAgent', StringType(), True),
                   StructField('userId', StringType(), True),
                   ]

    final_struct = StructType(fields=data_schema)
    return final_struct


def process_song_data(spark, input_data, output_data):
    """
    reads in song data from S3
    writes songs table and artists table to another S3 bucket as parquet files
    :param spark: SparkSession
    :param input_data: str,
    :param output_data: str
    :return:
    """
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"

    # read song data file
    df = spark.read.json(song_data, schema=create_song_schema())

    # extract columns to create songs table
    selected_columns = ["song_id", "title", "artist_id", "year", "duration"]
    songs_table = df.select(selected_columns)
    songs_table = songs_table.filter(songs_table.song_id.isNotNull())

    # write songs table to parquet files partitioned by year and artist_id
    path_write_songs = output_data + "songs_table/"
    songs_table.write.mode("overwrite").partitionBy("artist_id", "year").parquet(path_write_songs)

    # extract columns to create artists table
    selected_columns = ["artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude"]
    artists_table = df.select(selected_columns)
    artists_table = artists_table.filter(artists_table.artist_id.isNotNull())
    artists_table = artists_table.withColumnRenamed("artist_name", "name") \
        .withColumnRenamed("artist_location", "location") \
        .withColumnRenamed("artist_latitude", "latitude") \
        .withColumnRenamed("artist_longitude", "longitude")

    # write artists table to parquet files
    path_write_artits = output_data + "artists_table/"
    artists_table.repartition(1).write.mode("overwrite").parquet(path_write_artits)


def process_log_data(spark, input_data, output_data):
    """
    reads in song data and log datafrom S3
    writes users table, time table and songplay table to another S3 bucket as parquet files
    :param spark: SparkSession
    :param input_data: str,
    :param output_data: str
    :return:
    """
    # get filepath to log data file
    log_data = input_data + "log_data/*/*/*.json"

    # read log data file
    df = spark.read.json(log_data, schema=create_log_schema())

    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")

    # extract columns for users table
    selected_columns = ["userId", "firstName", "lastName", "gender", "level"]
    user_table = df.select(selected_columns)
    user_table = user_table.withColumnRenamed("firstName", "first_name") \
        .withColumnRenamed("lastName", "last_name") \
        .withColumnRenamed("userId", "user_id")

    # write users table to parquet files
    path_write_users = output_data + "users_table/"
    user_table.repartition(1).write.mode("overwrite").parquet(path_write_users)

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: int(x / 1000), IntegerType())
    df = df.withColumn('start_time_int', get_timestamp('ts'))
    df = df.withColumn("start_time", df.start_time_int.cast(TimestampType()))

    # extract columns to create time table
    selected_columns = ["start_time"]
    time_table = df.select(selected_columns)
    time_table = time_table.dropDuplicates() \
        .withColumn('hour', hour('start_time')) \
        .withColumn('day', dayofmonth('start_time')) \
        .withColumn('week', weekofyear('start_time')) \
        .withColumn('month', month('start_time')) \
        .withColumn('year', year('start_time')) \
        .withColumn('weekday', dayofweek('start_time'))

    time_table = time_table.filter(time_table.start_time.isNotNull())

    # write time table to parquet files partitioned by year and month
    path_write_time = output_data + "time_table/"
    time_table.write.partitionBy("year", "month").mode("overwrite").parquet(path_write_time)

    # read in song data to use for songplays table
    song_data = input_data + "song_data/*/*/*/*.json"

    # read song data file
    song_df = spark.read.json(song_data, schema=create_song_schema())

    # extract columns from joined song and log datasets to create songplays table
    cond = [
        (df.song == song_df.title),
        (df.length == song_df.duration),
        (df.artist == song_df.artist_name)
    ]

    songplays_table = df.join(song_df, cond, 'left')
    songplays_table = songplays_table.withColumn("songplay_id", monotonically_increasing_id())

    selected_columns = ["songplay_id", "start_time", "userId", "level", "song_id", "artist_id", "sessionId", "location",
                        "userAgent"]
    songplays_table = songplays_table.select(selected_columns)

    songplays_table = songplays_table.withColumnRenamed("userId", "user_id") \
        .withColumnRenamed("sessionId", "session_id") \
        .withColumnRenamed("userAgent", "user_agent")

    songplays_table = songplays_table.withColumn('year', year('start_time'))
    songplays_table = songplays_table.withColumn('month', month('start_time'))

    #write songplays table to parquet files partitioned by year and month
    path_write_songplays = output_data + "songplays_table/"
    songplays_table.write.mode("overwrite").partitionBy("year", "month").parquet(path_write_songplays)


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://rrrudacitybigdata/"

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
