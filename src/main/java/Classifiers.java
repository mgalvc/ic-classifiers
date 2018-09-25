import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class Classifiers {
    private Instances trainingData;
    private Instances testingData;
    private Classifier classifier;
    private String datasetPath;
    private String relation;
    private final int folds = 10;
    private final int executions = 3;
    private double[][] results;

    public Classifiers(String path, String relation) {
        // path specifies the root folder of the datasets
        this.datasetPath = path;
        this.relation = relation;

        // three executions - accuracy and kappa
        results = new double[3][3];
    }

    public Instances readData(String path) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(path));
        ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
        Instances data = arff.getData();
        data.setClassIndex(data.numAttributes() - 1);

        return data;
    }

    public void run() throws Exception {
        String baseNameTra;
        String baseNameTst;

        for (int i = 1; i <= this.executions; i++) {

            for (int j = 1; j <= this.folds; j++) {
                baseNameTra = String.format("%d_%s-%d-%dtra.arff", i, this.relation, this.folds, j);
                baseNameTst = String.format("%d_%s-%d-%dtst.arff", i, this.relation, this.folds, j);

                this.trainingData = this.readData(this.datasetPath + baseNameTra);
                this.testingData = this.readData(this.datasetPath + baseNameTst);

                this.runKNN(i-1);
                //this.runNaiveBayes(i-1);
                //this.runBayesNet(i-1);
                //this.runNeuralNet(i-1);
                System.out.println("Finished fold " + j);
            }
            System.out.println("Finished execution " + i);
            System.out.println("Took " + this.results[i-1][2] + " ms");
        }

        for (double[] result: results) {
            System.out.println(Arrays.toString(result));
        }
    }

    public void runNeuralNet(int executionIndex) throws Exception {
        this.classifier = new MultilayerPerceptron();

        classifier.buildClassifier(this.trainingData);

        Evaluation eval = new Evaluation(trainingData);
        eval.evaluateModel(this.classifier, this.testingData);

        this.results[executionIndex][0] += eval.pctCorrect()/this.folds;
        this.results[executionIndex][1] += eval.kappa()/this.folds;
    }

    public void runNaiveBayes(int executionIndex) throws Exception {
        this.classifier = new NaiveBayes();
        classifier.buildClassifier(this.trainingData);

        Evaluation eval = new Evaluation(trainingData);
        eval.evaluateModel(this.classifier, this.testingData);

        this.results[executionIndex][0] += eval.pctCorrect()/this.folds;
        this.results[executionIndex][1] += eval.kappa()/this.folds;
    }

    public void runBayesNet(int executionIndex) throws Exception {
        this.classifier = new BayesNet();
        classifier.buildClassifier(this.trainingData);

        Evaluation eval = new Evaluation(trainingData);
        eval.evaluateModel(this.classifier, this.testingData);

        this.results[executionIndex][0] += eval.pctCorrect()/this.folds;
        this.results[executionIndex][1] += eval.kappa()/this.folds;
    }

    public void runKNN(int executionIndex) throws Exception {
        this.classifier = new IBk();

        double startTime = System.currentTimeMillis();

        classifier.buildClassifier(this.trainingData);

        Evaluation eval = new Evaluation(trainingData);
        eval.evaluateModel(this.classifier, this.testingData);

        double stopTime = System.currentTimeMillis();

        this.results[executionIndex][0] += eval.pctCorrect()/this.folds;
        this.results[executionIndex][1] += eval.kappa()/this.folds;
        this.results[executionIndex][2] += stopTime - startTime;
    }
}
