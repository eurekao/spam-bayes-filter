import com.google.common.base.Predicates;
import com.google.common.collect.Collections2;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Created by Eureka on 2016-06-18.
 */
public class App {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf()
                .setMaster("spark://192.168.99.100:7077")
                .setJars(new String[]{"D:\\workspace\\java\\spamBayesFilter\\out\\artifacts\\spamBayesFilter_jar\\spamBayesFilter.jar"})
                .setAppName("SpamBayesFilter");
        JavaSparkContext sc = new JavaSparkContext(conf);
        JavaRDD<String> spamEmails = sc.textFile("hdfs://192.168.99.100:9000/user/root/emails/enron1/spam/*.txt");
        JavaRDD<String> hamEmails = sc.textFile("hdfs://192.168.99.100:9000/user/root/emails/enron1/ham/*.txt");
        final HashingTF tf = new HashingTF(100000);

        RDD<LabeledPoint> parsedSpamData = spamEmails.map(line -> {
            Pattern p = Pattern.compile("[^a-zA-Z]|\\d");
            String[] str = p.split(line);
            List<String> list = Stream.of(str).filter(item -> item != null && !"".equals(item)).collect(Collectors.toList());
            return new LabeledPoint(1.0, tf.transform(list));
        }).rdd();


        RDD<LabeledPoint> parsedHamData = hamEmails.map(line -> {
            Pattern p = Pattern.compile("[^a-zA-Z]|\\d");
            String[] str = p.split(line);
            List<String> list = Stream.of(str).filter(item -> item != null && !"".equals(item)).collect(Collectors.toList());
            return new LabeledPoint(0.0, tf.transform(list));
        }).rdd();

        //分隔为两个部分，60%的数据用于训练，40%的用于测试
		RDD<LabeledPoint> parsedData = parsedSpamData.union(parsedHamData);
		parsedData.cache();
        RDD<LabeledPoint>[] splits = parsedData.randomSplit(new double[]{0.6, 0.4}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].toJavaRDD();
        JavaRDD<LabeledPoint> test = splits[1].toJavaRDD();

        //训练模型， Additive smoothing的值为1.0（默认值）
        final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);


        JavaRDD<Double> prediction = test.map(p -> model.predict(p.features()));
        JavaPairRDD<Double, Double> predictionAndLabel = prediction.zip(test.map(LabeledPoint::label));
        //JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
        //用测试数据来验证模型的精度
        double accuracy = 1.0 * predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / test.count();
        System.out.println("Accuracy=" + accuracy);
    }
}
